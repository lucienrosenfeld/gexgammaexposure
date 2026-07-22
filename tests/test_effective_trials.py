"""R-NEFF-01: participation-ratio effective trials with validity fallbacks."""

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS
from espa.validation.metrics import effective_trials

RAW = 18


def _frame(cols: dict, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    return pd.DataFrame({k: v(rng, n) for k, v in cols.items()}, index=idx)


def test_orthogonal_streams_give_m():
    df = _frame({f"c{i}": (lambda r, n: r.normal(0, 1, n)) for i in range(4)})
    res = effective_trials(df, m_unrun=0, raw_count=RAW)
    assert not res.fallback
    assert abs(res.participation_ratio - 4) < 0.5


def test_identical_streams_give_one():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 600)
    df = pd.DataFrame({f"c{i}": base for i in range(4)})
    res = effective_trials(df, m_unrun=0, raw_count=RAW)
    assert not res.fallback
    assert abs(res.participation_ratio - 1) < 1e-6


def test_unrun_add_back():
    df = _frame({f"c{i}": (lambda r, n: r.normal(0, 1, n)) for i in range(3)})
    res = effective_trials(df, m_unrun=5, raw_count=RAW)
    assert res.n_eff == res.participation_ratio + 5


def test_zero_variance_exit_and_count():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "live1": rng.normal(0, 1, 600),
            "live2": rng.normal(0, 1, 600),
            "flat": np.zeros(600),  # all-flat: expected under flat-is-admissible
        }
    )
    res = effective_trials(df, m_unrun=0, raw_count=RAW)
    assert not res.fallback
    assert res.n_zero_variance == 1
    assert res.m_run == 2
    assert res.n_eff == res.participation_ratio + 1


def test_each_validity_condition_triggers_fallback():
    rng = np.random.default_rng(3)
    # (i) common intersection below the floor
    small = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=list("abc"))
    r1 = effective_trials(small, 0, RAW)
    assert r1.fallback and "500" in r1.reason and r1.effective() == RAW
    # (ii) too many configs for the observations
    wide = pd.DataFrame(rng.normal(0, 1, (510, 60)))
    wide.columns = [f"c{i}" for i in range(60)]
    r2 = effective_trials(wide, 0, RAW)
    assert r2.fallback and "configurations" in r2.reason
    # (iv) stability: a hard regime break in the correlation structure
    n = 600
    base = rng.normal(0, 1, n)
    a = base.copy()
    b = np.concatenate([base[: n // 2], rng.normal(0, 1, n - n // 2)])
    unstable = pd.DataFrame({"a": a, "b": b, "c": rng.normal(0, 1, n)})
    r4 = effective_trials(unstable, 0, RAW)
    if r4.fallback:
        assert "stability" in r4.reason
    # NaN days shrink the common intersection rather than corrupting it
    nanned = pd.DataFrame(rng.normal(0, 1, (600, 3)), columns=list("abc"))
    nanned.iloc[:150, 0] = np.nan
    r5 = effective_trials(nanned, 0, RAW)
    assert r5.n_common == 450
    assert r5.fallback  # 450 < 500 => condition (i)


def test_stability_check_arithmetic():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(0, 1, (1000, 4)), columns=list("abcd"))
    res = effective_trials(df, 0, RAW)
    assert not res.fallback
    k = int(np.floor(DEFAULT_CONSTANTS.neff_stability_frac * res.n_common))
    assert k == 800
    div = max(
        abs(res.neff_first_window - res.participation_ratio),
        abs(res.neff_last_window - res.participation_ratio),
    ) / res.participation_ratio
    assert div == res.stability_divergence
    assert div <= DEFAULT_CONSTANTS.neff_stability_tol
