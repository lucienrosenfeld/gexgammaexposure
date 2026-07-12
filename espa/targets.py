"""Targets and the ex-ante volatility scale (Section 9).

The forecast model trains on the mid-price return; every *selection*
decision (dead zone, Sharpe, configuration choice, significance) runs on
direction-conditioned executable P&L, so a model that predicts mid
direction but cannot overcome the spread is rejected by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


def ex_ante_window_vol(
    window_returns: pd.Series,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.Series:
    """Floored ex-ante volatility scale sigma*_t.

    sigma_t = (1 - alpha) * sum_{j=1..20} alpha^(j-1) |r_{t-j}|

    using observations through t-1 only — the explicit lag structure
    exists so no implementation accidentally updates the estimate with
    the target day's own return. Floored at the rolling 10th percentile
    of sigma (itself strictly lagged) so a run of quiet windows cannot
    produce excessive leverage.
    """
    span = constants.vol_ewma_span
    alpha = 1.0 - 2.0 / (span + 1.0)  # 20-day-equivalent decay
    terms = constants.vol_ewma_terms
    weights = (1.0 - alpha) * alpha ** np.arange(terms)

    abs_r = window_returns.abs().to_numpy(dtype=float)
    n = abs_r.size
    sigma = np.full(n, np.nan)
    for t in range(1, n):
        j = min(terms, t)
        lags = abs_r[t - j : t][::-1]  # r_{t-1}, r_{t-2}, ...
        if np.isnan(lags).all():
            continue
        w = weights[:j]
        mask = ~np.isnan(lags)
        if mask.sum() == 0:
            continue
        sigma[t] = float(np.sum(w[mask] * lags[mask]))

    sigma_s = pd.Series(sigma, index=window_returns.index, name="sigma")
    floor = (
        sigma_s.shift(1)
        .rolling(constants.vol_floor_window, min_periods=constants.vol_ewma_terms)
        .quantile(constants.vol_floor_quantile)
    )
    return pd.concat([sigma_s, floor], axis=1).max(axis=1).rename("sigma_star")


@dataclass(frozen=True)
class WindowPrices:
    """Entry/exit prices for one day's window, roll-adjusted front ES."""

    mid_entry: float
    mid_exit: float
    bid_entry: float
    ask_entry: float
    bid_exit: float
    ask_exit: float


def targets_from_prices(
    prices: pd.DataFrame,
    sigma_star: pd.Series,
    costs: pd.Series,
) -> pd.DataFrame:
    """Build the modelling and selection targets.

    ``prices`` needs columns mid_entry, mid_exit, bid_entry, ask_entry,
    bid_exit, ask_exit; ``costs`` is c_t (fees plus modelled slippage),
    in price points.

    Returns columns:
      y_mid   — (Mid_exit - Mid_entry) / sigma*      (modelling target)
      y_long  — (Bid_exit - Ask_entry - c) / sigma*  (selection target)
      y_short — (Bid_entry - Ask_exit - c) / sigma*  (selection target)
    """
    s = sigma_star.reindex(prices.index)
    out = pd.DataFrame(index=prices.index)
    out["y_mid"] = (prices["mid_exit"] - prices["mid_entry"]) / s
    out["y_long"] = (prices["bid_exit"] - prices["ask_entry"] - costs) / s
    out["y_short"] = (prices["bid_entry"] - prices["ask_exit"] - costs) / s
    return out


def executable_pnl(direction: pd.Series, targets: pd.DataFrame) -> pd.Series:
    """Direction-conditioned executable P&L in sigma* units.

    ``direction`` in {-1, 0, +1}; flat days contribute exactly zero.
    """
    pnl = pd.Series(0.0, index=targets.index)
    long_days = direction > 0
    short_days = direction < 0
    pnl[long_days] = targets.loc[long_days, "y_long"]
    pnl[short_days] = targets.loc[short_days, "y_short"]
    return pnl
