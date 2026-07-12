"""Unsigned gamma density A_t and the strike-level building blocks (Section 11).

    Gtilde_i = Gamma_i * OI_i * Q_i * S^2 * 0.01
               * exp(-|K_i - S| / (lambda_S * S)) * exp(-T_i / lambda_T)

Deliberately no positioning sign eta_i: A_t answers the question public
data can answer — how much convexity sits near the money relative to the
liquidity available in the window, regardless of who holds it. Its
admitted roles are the Stage 2 interaction, the volatility conditioner,
and sizing; a standalone directional coefficient is a counted test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


@dataclass(frozen=True)
class OptionContract:
    """One SPX/SPXW or ES-option contract's strike-level state for a day."""

    strike: float
    expiry_years: float  # T_i, in years, from the observation day
    gamma: float  # per-contract Black-Scholes gamma
    open_interest: float
    multiplier: float
    same_day_volume: float = 0.0


def kernel_weight(
    strike: float,
    expiry_years: float,
    spot: float,
    lambda_s: float,
    lambda_t: float,
) -> float:
    """Proximity and expiry decay kernels."""
    return float(
        np.exp(-abs(strike - spot) / (lambda_s * spot)) * np.exp(-expiry_years / lambda_t)
    )


def contract_gamma_weight(
    c: OptionContract,
    spot: float,
    lambda_s: float,
    lambda_t: float,
    exposure: float | None = None,
) -> float:
    """Gtilde for one contract; ``exposure`` defaults to open interest.

    Pass same-day volume (or trade-level open/close-classified flow if
    sourced) as ``exposure`` to build the same-day component of the
    blended 0DTE surface.
    """
    size = c.open_interest if exposure is None else exposure
    return (
        c.gamma
        * size
        * c.multiplier
        * spot**2
        * 0.01
        * kernel_weight(c.strike, c.expiry_years, spot, lambda_s, lambda_t)
    )


def raw_gamma_density(
    contracts: list[OptionContract],
    spot: float,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> float:
    """Sum_i |Gtilde_i| at ``spot``, before the liquidity denominator."""
    return float(
        sum(abs(contract_gamma_weight(c, spot, lambda_s, lambda_t)) for c in contracts)
    )


def window_liquidity(
    window_dollar_volume: pd.Series,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.Series:
    """Median 20-day dollar volume of ES in the post-close window, lagged.

    Full-session ADV is rejected as the denominator because it is not
    the liquidity that would absorb a post-close unwind. Upgrade path:
    a top-of-book depth median for the same window if historical depth
    is sourced (Phase 4).
    """
    return (
        window_dollar_volume.shift(1)
        .rolling(constants.liquidity_median_window, min_periods=5)
        .median()
        .rename("window_liquidity")
    )


def gamma_density(
    daily_contracts: dict, spots: pd.Series, liquidity: pd.Series,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> pd.Series:
    """Raw (pre-z-score) A_t numerator over liquidity, per day.

    ``daily_contracts`` maps index label -> list[OptionContract].
    The caller applies :func:`espa.standardize.robust_z` to the result;
    A_t = z(sum|Gtilde| / Liquidity).
    """
    out = pd.Series(np.nan, index=spots.index, name="A_raw")
    for day in spots.index:
        contracts = daily_contracts.get(day)
        liq = liquidity.get(day, np.nan)
        if contracts is None or not np.isfinite(liq) or liq <= 0:
            continue
        out.loc[day] = raw_gamma_density(contracts, float(spots.loc[day]), lambda_s, lambda_t) / liq
    return out
