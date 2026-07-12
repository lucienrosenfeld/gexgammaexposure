"""Constrained ridge, stacking discipline, blending, uncertainty."""

import numpy as np
import pandas as pd
import pytest

from espa.config import DEFAULT_CONSTANTS, SpecConstants
from espa.models.blend import LiveBlender
from espa.models.constrained_ridge import ConstrainedRidge
from espa.models.stage1 import Stage1Model, out_of_fold_predictions
from espa.models.stage2 import fit_stage2_stacked
from espa.models.uncertainty import ForecastUncertainty


def test_constrained_ridge_respects_signs():
    rng = np.random.default_rng(2)
    n = 500
    X = rng.normal(0, 1, (n, 3))
    # true coefficients violate the constraints on purpose
    y = -1.0 * X[:, 0] + 0.5 * X[:, 1] + 0.8 * X[:, 2] + rng.normal(0, 0.1, n)
    m = ConstrainedRidge(lam=1.0).fit(X, y, constraints=[+1, -1, 0])
    w = m.coef_
    assert w[0] >= 0  # data wants -1; constraint shrinks to zero instead
    assert w[0] == pytest.approx(0.0, abs=1e-8)
    assert w[1] <= 0
    assert w[1] == pytest.approx(0.0, abs=1e-8)
    assert w[2] == pytest.approx(0.8, abs=0.05)  # free coefficient recovered


def test_constrained_ridge_recovers_conforming_signs():
    rng = np.random.default_rng(3)
    n = 500
    X = rng.normal(0, 1, (n, 3))
    y = 0.7 * X[:, 0] - 0.4 * X[:, 1] - 0.6 * X[:, 2] + rng.normal(0, 0.1, n)
    m = ConstrainedRidge(lam=0.5).fit(X, y, constraints=[+1, -1, 0])
    assert m.coef_[0] == pytest.approx(0.7, abs=0.05)
    assert m.coef_[1] == pytest.approx(-0.4, abs=0.05)
    assert m.coef_[2] == pytest.approx(-0.6, abs=0.05)


def test_ridge_shrinks():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (100, 2))
    y = X[:, 0] + rng.normal(0, 1, 100)
    small = ConstrainedRidge(lam=0.01).fit(X, y, [0, 0]).coef_
    big = ConstrainedRidge(lam=1000.0).fit(X, y, [0, 0]).coef_
    assert abs(big[0]) < abs(small[0])


def test_out_of_fold_predictions_are_out_of_fold():
    """OOF prediction for a day must not depend on that day's own y."""
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.RangeIndex(n)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series(X["a"].to_numpy() + rng.normal(0, 0.5, n), index=idx)
    cons = {"a": 0, "b": 0}
    oof1 = out_of_fold_predictions(X, y, cons, lam=1.0)
    y2 = y.copy()
    y2.iloc[10] += 100.0  # poison one observation
    oof2 = out_of_fold_predictions(X, y2, cons, lam=1.0)
    # day 10 sits in fold 0; its own OOF prediction is fitted without it
    assert oof1.iloc[10] == pytest.approx(oof2.iloc[10])


def test_blender_starts_at_zero_and_earns_weight():
    b = LiveBlender()
    assert b.current_lambda() == 0.0
    # options block consistently improves the forecast
    rng = np.random.default_rng(6)
    for _ in range(200):
        y = rng.normal(0, 1)
        q = y + rng.normal(0, 1.0)  # bad stage 1
        o = (y - q) * 0.9  # options block corrects most of the error
        b.update(y, q, o, incremental_pnl=0.1)
    assert b.current_lambda() == 1.0


def test_utility_gate_forces_lambda_to_zero():
    b = LiveBlender()
    rng = np.random.default_rng(7)
    for _ in range(200):
        y = rng.normal(0, 1)
        q = y + rng.normal(0, 1.0)
        o = (y - q) * 0.9  # MSE says the block is great...
        b.update(y, q, o, incremental_pnl=-0.1)  # ...but it loses money
    assert b.current_lambda() == 0.0


def test_blender_warmup_prevents_day_one_full_weight():
    b = LiveBlender()
    rng = np.random.default_rng(60)
    for i in range(b.min_updates + 5):
        y = rng.normal(0, 1)
        q = y + rng.normal(0, 1.0)
        o = (y - q) * 0.9
        if i < b.min_updates:
            # one lucky comparison must not grant weight during warmup
            assert b.current_lambda() == 0.0
        b.update(y, q, o, incremental_pnl=0.1)
    assert b.current_lambda() > 0.0


def test_utility_gate_reopens_when_window_rolls():
    """The gate must not latch shut forever once forced to zero."""
    b = LiveBlender()
    rng = np.random.default_rng(61)
    win = b.constants.utility_gate_window
    # phase 1: block loses money -> gate shuts
    for _ in range(win):
        y = rng.normal(0, 1)
        q = y + rng.normal(0, 1.0)
        b.update(y, q, (y - q) * 0.9, incremental_pnl=-0.1)
    assert b.current_lambda() == 0.0
    # phase 2: counterfactual contribution turns positive; once the
    # trailing window rolls over, the gate reopens
    for _ in range(win + 1):
        y = rng.normal(0, 1)
        q = y + rng.normal(0, 1.0)
        b.update(y, q, (y - q) * 0.9, incremental_pnl=0.1)
    assert b.current_lambda() > 0.0


def test_stacking_stage2_sees_only_oof_stage1():
    """Stage 2 fitted against OOF predictions, not in-sample fits.

    With pure-noise features, in-sample Stage 1 fits correlate with y,
    so a leaky stack would hand Stage 2 a usable (spurious) interaction
    signal. We check the discipline structurally: the interaction
    feature fed to Stage 2 is built from OOF predictions that differ
    from the final refit model's in-sample predictions.
    """
    rng = np.random.default_rng(8)
    n = 400
    idx = pd.RangeIndex(n)
    feats = pd.DataFrame(
        {c: rng.normal(0, 1, n) for c in ("C_early", "C_late", "I_level")}, index=idx
    )
    cons = {"C_early": +1, "C_late": 0, "I_level": +1}
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    a = pd.Series(rng.normal(0, 1, n), index=idx)
    pgi = pd.Series(rng.normal(0, 1, n), index=idx)
    ce = pd.Series(rng.normal(0, 1, n), index=idx)
    cl = pd.Series(rng.normal(0, 1, n), index=idx)
    rd = pd.Series(rng.normal(0, 1, n), index=idx)
    p = pd.Series(rng.normal(0, 1, n), index=idx)
    fit = fit_stage2_stacked(feats, cons, y, a, pgi, ce, cl, rd, p, 1.0, 1.0)
    oof = out_of_fold_predictions(feats, y, cons, lam=1.0)
    insample = fit.stage1.predict(feats)
    # OOF and in-sample predictions must differ (else the stack leaked)
    diff = (oof - insample).abs().dropna()
    assert (diff > 1e-12).any()
    # and on pure noise the Stage 2 coefficients stay small
    assert fit.stage2.coef.abs().max() < 0.5


def test_forecast_uncertainty_scales_with_disagreement():
    u = ForecastUncertainty()
    for w in ([1.0, 0.0], [1.1, 0.0], [0.9, 0.0]):
        u.record(np.array(w))
    x = np.array([2.0, 0.0])
    s = u.sigma(x)
    assert np.isfinite(s) and s > 0
    # identical refits: the relative floor keeps sigma positive
    u2 = ForecastUncertainty()
    for _ in range(3):
        u2.record(np.array([1.0, 0.0]))
    assert u2.sigma(x) > 0
