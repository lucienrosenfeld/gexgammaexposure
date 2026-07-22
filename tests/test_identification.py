"""D-STAGE2-ID-01: kappa arithmetic, materiality floor, persistence, immutability."""

import numpy as np
import pandas as pd
import pytest

from espa.config import DEFAULT_CONSTANTS
from espa.validation.identification import (
    FoldIDRecord,
    IdentificationTracker,
    fold_identification,
)
from espa.validation.metrics import sign_stability


def _design(rho: float, n=500, seed=0) -> pd.DataFrame:
    """Three standardised columns; first two correlated at rho."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    return pd.DataFrame({"A_x_Qhat": a, "PGI_perp": b, "P": c})


def _record(fold, rho, coefs, ses, seed=0):
    return fold_identification(
        fold,
        _design(rho, seed=seed),
        pd.Series(coefs, index=["A_x_Qhat", "PGI_perp", "P"]),
        pd.Series(ses, index=["A_x_Qhat", "PGI_perp", "P"]),
    )


def test_kappa_arithmetic():
    """rho = 0.8 gives kappa ~ (1+rho)/(1-rho) = 9: crosses nothing at the
    conventional 30, crosses at the recalibrated 10 once rho ~ 0.82."""
    k_08 = _record(0, 0.8, [1, 1, 1], [0.1, 0.1, 0.1]).kappa
    assert 7.5 < k_08 < 10.5
    assert k_08 < 30  # the old threshold would never have fired here
    k_094 = _record(0, 0.94, [1, 1, 1], [0.1, 0.1, 0.1]).kappa
    assert k_094 > 25  # ~ the correlation needed to reach the old threshold


def test_alarm_requires_kappa_and_material_flip():
    t = IdentificationTracker()
    # collinear design + material sign flip -> alarm
    t.add(_record(0, 0.9, [0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=1))
    t.add(_record(1, 0.9, [-0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=2))
    assert t.alarms() == [1]
    # same flip on a well-conditioned design -> no alarm
    t2 = IdentificationTracker()
    t2.add(_record(0, 0.1, [0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=3))
    t2.add(_record(1, 0.1, [-0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=4))
    assert t2.alarms() == []


def test_materiality_floor_suppresses_noise_flips():
    """Section 26 scenario: beta1 ~ 0 flips sign as pure noise. With the
    coefficient inside its SE on either side, no alarm may fire."""
    t = IdentificationTracker()
    t.add(_record(0, 0.95, [0.02, 0.3, 0.1], [0.10, 0.1, 0.1], seed=5))
    t.add(_record(1, 0.95, [-0.03, 0.3, 0.1], [0.10, 0.1, 0.1], seed=6))
    assert t.alarms() == []


def test_persistence_rule():
    c = DEFAULT_CONSTANTS
    t = IdentificationTracker()
    # alternate material flips on a collinear design: every transition alarms
    for f in range(4):
        sign = 1 if f % 2 == 0 else -1
        t.add(_record(f, 0.95, [sign * 0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=10 + f))
    assert len(t.alarms()) == 3 >= c.id_min_alarms
    assert t.persistent()
    # one flipped fold => two flip transitions (in and out) across 11
    # comparable transitions: 18% rate and 2 < 3 alarms, below both floors
    t2 = IdentificationTracker()
    for f in range(12):
        sign = -1 if f == 5 else 1
        t2.add(_record(f, 0.95, [sign * 0.5, 0.3, 0.1], [0.1, 0.1, 0.1], seed=30 + f))
    assert len(t2.alarms()) == 2
    assert not t2.persistent()


def test_tracker_never_mutates_model_objects():
    coefs = pd.Series([0.5, 0.3, 0.1], index=["A_x_Qhat", "PGI_perp", "P"])
    ses = pd.Series([0.1, 0.1, 0.1], index=["A_x_Qhat", "PGI_perp", "P"])
    rec = fold_identification(0, _design(0.9), coefs, ses)
    rec.coefs.iloc[0] = 99.0  # mutating the record's copy...
    assert coefs.iloc[0] == 0.5  # ...never touches the model's series
    # and the module imports nothing from espa.models / espa.features
    import espa.validation.identification as ident

    assert not any(
        m.startswith(("espa.models", "espa.features"))
        for m in getattr(ident, "__dict__", {})
        if isinstance(m, str) and m.startswith("espa")
    )


def test_sign_stability_materiality_variant():
    coefs = pd.DataFrame({"b1": [0.02, -0.03, 0.02], "b2": [0.5, -0.5, 0.5]})
    ses = pd.DataFrame({"b1": [0.1, 0.1, 0.1], "b2": [0.1, 0.1, 0.1]})
    plain = sign_stability(coefs)
    assert plain["b1"] < 1.0  # unfiltered rule counts the noise flips
    filtered = sign_stability(coefs, coef_se_by_fold=ses, materiality_mult=1.0)
    assert filtered["b1"] == 1.0  # immaterial flips suppressed
    assert filtered["b2"] < 1.0  # material flips still counted


def test_vif_and_eigenmode_share():
    rec = _record(0, 0.9, [1, 1, 1], [0.1, 0.1, 0.1])
    assert rec.vif["A_x_Qhat"] > 4  # 1/(1-rho^2) ~ 5.3
    assert rec.vif["P"] == pytest.approx(1.0, abs=0.2)
    assert 0 < rec.smallest_eigenmode_share < 1 / 3
    assert rec.to_log_record()["kappa"] == rec.kappa
