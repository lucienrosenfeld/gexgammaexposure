"""R-IMB-BETA-01: frozen beta pipeline, screen, and builder admission gate."""

import numpy as np
import pandas as pd
import pytest

from espa.config import DEFAULT_CONSTANTS, RunConfig
from espa.features.core import Stage1Inputs, build_stage1_features
from espa.features.imbalance_beta import (
    ScreenStatus,
    beta_weighted_imbalance,
    cap_weighted_imbalance,
    redundancy_screen,
    rolling_es_betas,
)


def _panel(n_days=800, n_names=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    names = [f"S{i}" for i in range(n_names)]
    es = pd.Series(rng.normal(0, 0.01, n_days), index=idx)
    true_betas = np.linspace(0.5, 2.0, n_names)
    rets = pd.DataFrame(
        {nm: b * es + rng.normal(0, 0.005, n_days) for nm, b in zip(names, true_betas)},
        index=idx,
    )
    membership = pd.DataFrame(True, index=idx, columns=names)
    return rets, es, membership, true_betas


def test_beta_recovery_shrunk_and_clipped():
    rets, es, membership, true = _panel()
    betas = rolling_es_betas(rets, es, membership)
    est = betas.iloc[-1].to_numpy()
    # order preserved and values pulled toward 1 by the 0.5 shrinkage
    assert (np.diff(est) > 0).all()
    expected = 0.5 * true + 0.5
    assert np.allclose(est, expected, atol=0.15)
    lo, hi = DEFAULT_CONSTANTS.beta_clip
    assert (betas.stack() >= lo).all() and (betas.stack() <= hi).all()


def test_shrink_then_clip_order():
    """A raw beta of 6 shrinks to 3.5 then clips to 2.5; clip-then-shrink
    would give 1.75. The pipeline order is frozen — shrink first."""
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    es = pd.Series(rng.normal(0, 0.01, n), index=idx)
    rets = pd.DataFrame({"HOT": 6.0 * es + rng.normal(0, 0.002, n)}, index=idx)
    membership = pd.DataFrame(True, index=idx, columns=["HOT"])
    betas = rolling_es_betas(rets, es, membership)
    assert betas["HOT"].iloc[-1] == pytest.approx(2.5)  # hi clip, not 1.75


def test_fallback_below_min_obs():
    rets, es, membership, _ = _panel(n_days=100)  # < 120 paired obs everywhere
    betas = rolling_es_betas(rets, es, membership)
    assert (betas.iloc[-1] == DEFAULT_CONSTANTS.beta_fallback).all()


def test_no_same_day_contamination():
    """A constructed same-day leak must not move the beta: the window ends
    at t-1 by construction, so poisoning day T's returns changes nothing
    at day T."""
    rets, es, membership, _ = _panel()
    base = rolling_es_betas(rets, es, membership)
    rets2, es2 = rets.copy(), es.copy()
    rets2.iloc[-1] = 10.0  # absurd same-day move
    es2.iloc[-1] = 0.5
    poisoned = rolling_es_betas(rets2, es2, membership)
    pd.testing.assert_series_equal(base.iloc[-1], poisoned.iloc[-1])


def test_membership_masks_output():
    rets, es, membership, _ = _panel()
    membership.iloc[:, 0] = False
    betas = rolling_es_betas(rets, es, membership)
    assert betas.iloc[-1, 0] != betas.iloc[-1, 0]  # NaN for the non-member


def test_screen_admit_reject_defer():
    rng = np.random.default_rng(2)
    n = 600
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    i_w = pd.Series(rng.normal(0, 1, n), index=idx)
    # near-identical candidate -> REJECT
    r = redundancy_screen(i_w + rng.normal(0, 0.05, n), i_w)
    assert r.status is ScreenStatus.REJECT and abs(r.rho) > 0.9
    # independent candidate -> ADMIT
    r2 = redundancy_screen(pd.Series(rng.normal(0, 1, n), index=idx), i_w)
    assert r2.status is ScreenStatus.ADMIT
    # short history -> DEFER, not decided
    r3 = redundancy_screen(i_w.iloc[:200], i_w.iloc[:200])
    assert r3.status is ScreenStatus.DEFER and r3.rho is None
    # records serialise with the frozen rule and usable-day count
    rec = r.to_registry_record()
    assert "0.9" in rec["rule"] and rec["n_usable_z_days"] > 0


def test_builder_refuses_i1beta_without_admit():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.RangeIndex(n)
    s = lambda: pd.Series(rng.normal(0, 1, n), index=idx)  # noqa: E731
    inputs = Stage1Inputs(
        c_early=s(), c_late=s(), i_55=s(), i_50=s(), x_composite=s(),
        b_live=s(), vwap_deviation=s(), i_beta_55=s(), i_beta_50=s(),
    )
    cfg = RunConfig(name="t", imbalance_spec="I1beta")
    with pytest.raises(ValueError, match="ADMIT"):
        build_stage1_features(inputs, cfg)
    feats, cons = build_stage1_features(inputs, cfg, i1beta_admitted=True)
    assert len(feats.columns) == 7
    assert cons["I_level"] == +1  # level keeps its sign prior under all specs
    assert cons["I_change"] == 0


def test_aggregation_uses_fallback_for_missing_betas():
    idx = pd.date_range("2022-01-03", periods=3, freq="B")
    q = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.5, 0.5, 0.5]}, index=idx)
    w = pd.DataFrame({"A": [0.6, 0.6, 0.6], "B": [0.4, 0.4, 0.4]}, index=idx)
    betas = pd.DataFrame({"A": [2.0, 2.0, 2.0], "B": [np.nan] * 3}, index=idx)
    ib = beta_weighted_imbalance(q, w, betas)
    # B falls back to beta = 1: 0.6*2*1 + 0.4*1*0.5 = 1.4
    assert ib.iloc[0] == pytest.approx(1.4)
    iw = cap_weighted_imbalance(q, w)
    assert iw.iloc[0] == pytest.approx(0.8)
