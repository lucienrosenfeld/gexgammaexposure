"""Post-auction basis state B_live (Sections 7 and 10).

Two basis series exist. B_live is built at decision time from received
official constituent auction prints, last tradable prices for unresolved
constituents, and known index weights. B_final uses the eventual
official close and exists for ex-post diagnostics only. Only B_live may
enter the trading model; that restriction is enforced by construction —
the trading feature builder simply has no B_final argument.

    B_live_t = [ES_16:00:15 - SPXhat_close * exp((r - q) * tau)] / sigma_basis_252

Rich futures carry a reversion prior (w_B <= 0 in Stage 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_spx_close(
    weights: pd.Series,
    auction_prints: pd.Series,
    last_tradable: pd.Series,
    prior_close_index: float,
    prior_close_prices: pd.Series,
) -> float:
    """Decision-time estimate of the SPX official close.

    For constituents whose auction print has been received, use it; for
    the rest, use the last tradable price. The index level is scaled off
    the prior close so that only *relative* constituent moves matter and
    divisor bookkeeping cancels:

        SPXhat = prior_index * sum_i w_i * (P_i / P_i_prior)

    with w_i the known index weights (summing to 1).
    """
    prices = auction_prints.combine_first(last_tradable)
    common = weights.index.intersection(prices.index).intersection(prior_close_prices.index)
    w = weights.loc[common]
    rel = prices.loc[common] / prior_close_prices.loc[common]
    covered = w.sum()
    if covered <= 0:
        raise ValueError("no constituent coverage for SPX close estimate")
    return float(prior_close_index * (w * rel).sum() / covered)


def fair_value_factor(r: float, q: float, tau_years: float) -> float:
    """exp((r - q) * tau): carry from decision time to futures expiry.

    r from term SOFR to expiry, q from the index dividend futures strip.
    """
    return float(np.exp((r - q) * tau_years))


def live_basis(
    es_price: pd.Series,
    spx_close_estimate: pd.Series,
    r: pd.Series,
    q: pd.Series,
    tau_years: pd.Series,
    scale_window: int = 252,
) -> pd.Series:
    """B_live series across days, normalised by its own lagged dispersion.

    The denominator sigma_basis_252 is the strictly lagged rolling
    standard deviation of the raw basis, consistent with Section 8's
    no-same-day-information rule.
    """
    carry = np.exp((r - q) * tau_years)
    raw = es_price - spx_close_estimate * carry
    scale = raw.shift(1).rolling(scale_window, min_periods=60).std()
    return (raw / scale.replace(0.0, np.nan)).rename("B_live")
