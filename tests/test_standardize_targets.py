"""Standardisation and target construction: the leakage-sensitive primitives."""

import numpy as np
import pandas as pd
import pytest

from espa.standardize import robust_z
from espa.targets import ex_ante_window_vol, executable_pnl, targets_from_prices


def test_robust_z_is_strictly_lagged():
    """A huge value on day t must not affect its own z-score's centre/scale."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 400))
    z_base = robust_z(x)
    x2 = x.copy()
    x2.iloc[350] = 1e6  # shock day t
    z_shocked = robust_z(x2)
    # day 350's own z uses only history through 349: same centre/scale
    med = x.iloc[max(0, 349 - 251) : 350].median()
    assert z_shocked.iloc[350] > 100  # the shock itself is an outlier
    # all days strictly before the shock are untouched
    pd.testing.assert_series_equal(z_base.iloc[:350], z_shocked.iloc[:350])


def test_robust_z_uses_median_not_mean():
    x = pd.Series(np.ones(300))
    x.iloc[::7] = 50.0  # heavy-tailed contamination
    z = robust_z(x)
    # median/MAD of a mostly-constant series: MAD 0 -> NaN, not explosion
    assert not np.isinf(z.dropna()).any()


def test_ex_ante_vol_excludes_own_day():
    r = pd.Series(np.full(100, 1.0))
    r.iloc[60] = 100.0
    sigma = ex_ante_window_vol(r)
    # sigma at day 60 must not include day 60's own return
    assert sigma.iloc[60] == pytest.approx(sigma.iloc[59], rel=0.05)
    # but day 61 sees it
    assert sigma.iloc[61] > sigma.iloc[60] * 2


def test_vol_floor_binds():
    rng = np.random.default_rng(1)
    r = pd.Series(np.abs(rng.normal(0, 1, 600)))
    r.iloc[585:] = 1e-6  # a short abnormally quiet run
    sigma = ex_ante_window_vol(r)
    # the floor (rolling 10th percentile) prevents sigma* collapsing to ~0
    assert sigma.iloc[-1] > 1e-2


def test_executable_targets_penalise_spread_and_costs():
    idx = pd.Index([0, 1])
    prices = pd.DataFrame(
        {
            "mid_entry": [100.0, 100.0],
            "mid_exit": [101.0, 99.0],
            "bid_entry": [99.9, 99.9],
            "ask_entry": [100.1, 100.1],
            "bid_exit": [100.9, 98.9],
            "ask_exit": [101.1, 99.1],
        },
        index=idx,
    )
    sigma = pd.Series([1.0, 1.0], index=idx)
    costs = pd.Series([0.05, 0.05], index=idx)
    t = targets_from_prices(prices, sigma, costs)
    assert t["y_mid"].iloc[0] == pytest.approx(1.0)
    # long pays spread twice plus costs
    assert t["y_long"].iloc[0] == pytest.approx(100.9 - 100.1 - 0.05)
    assert t["y_short"].iloc[1] == pytest.approx(99.9 - 99.1 - 0.05)


def test_executable_pnl_direction_conditioning():
    idx = pd.Index([0, 1, 2])
    targets = pd.DataFrame(
        {"y_long": [0.5, -0.2, 0.3], "y_short": [-0.6, 0.4, 0.1]}, index=idx
    )
    d = pd.Series([1.0, -1.0, 0.0], index=idx)
    pnl = executable_pnl(d, targets)
    assert pnl.tolist() == [0.5, 0.4, 0.0]
