"""Path-gamma integral and its momentum residualisation (Section 11).

Raw statistic, accumulated over five-minute bars from 9:30 to 16:00,
re-marking the 0DTE surface as spot moves:

    PGI_t = -sum_s Gtilde_0DTE(S_s) * dS_s / S_s

Stale-OI handling: the surface blends previous-evening open interest
with same-day volume, Gtilde = Gtilde_priorOI + rho * Gtilde_volume,
rho in {0, 0.25, 0.5} as counted configurations.

The raw statistic is mechanically correlated with ordinary intraday
directional movement, so inside every training fold, on training data
only, PGI is regressed on [C_early, C_late, r_day] and the residual
PGI_perp is what enters Stage 2. The residualisation coefficients are
frozen per fold and applied forward — :class:`PGIResidualiser` is that
freeze, made unforgettable by the type system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS
from espa.features.options import OptionContract, contract_gamma_weight


def blended_0dte_gamma(
    contracts: list[OptionContract],
    spot: float,
    rho: float,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> float:
    """Gtilde_0DTE at ``spot``: prior-OI surface plus rho * same-day-volume surface."""
    total = 0.0
    for c in contracts:
        total += contract_gamma_weight(c, spot, lambda_s, lambda_t)
        if rho > 0.0 and c.same_day_volume > 0.0:
            total += rho * contract_gamma_weight(
                c, spot, lambda_s, lambda_t, exposure=c.same_day_volume
            )
    return total


def path_gamma_integral(
    intraday_spots: pd.Series,
    contracts_0dte: list[OptionContract],
    rho: float,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> float:
    """PGI for one day from the 9:30-16:00 five-minute spot path.

    The surface is re-marked at each bar's opening spot; the increment
    is weighted by the bar's return. Under a global short-gamma prior
    this approximates hedging inventory, but the statistic itself is
    positioning-agnostic — the name is 'path-weighted directional gamma
    statistic', and the Stage 2 coefficient on it is sign-unconstrained.
    """
    s = intraday_spots.to_numpy(dtype=float)
    if s.size < 2:
        return np.nan
    # Vectorised equivalent of summing blended_0dte_gamma(s[k-1]) * ret_k
    # over bars: the surface is re-marked at each bar's opening spot
    # through the S^2 term and the proximity kernel; per-contract gamma,
    # OI and volume are the day's fixed inputs.
    strikes = np.array([c.strike for c in contracts_0dte], dtype=float)
    gammas = np.array([c.gamma for c in contracts_0dte], dtype=float)
    mults = np.array([c.multiplier for c in contracts_0dte], dtype=float)
    tdecay = np.exp(
        -np.array([c.expiry_years for c in contracts_0dte], dtype=float) / lambda_t
    )
    size = np.array([c.open_interest for c in contracts_0dte], dtype=float)
    if rho > 0.0:
        size = size + rho * np.array(
            [max(c.same_day_volume, 0.0) for c in contracts_0dte], dtype=float
        )
    s0 = s[:-1]  # bar-opening spots, shape (n_bars,)
    kernel = np.exp(-np.abs(strikes[None, :] - s0[:, None]) / (lambda_s * s0[:, None]))
    gtilde = (gammas * size * mults * tdecay)[None, :] * (s0**2)[:, None] * 0.01 * kernel
    g_per_bar = gtilde.sum(axis=1)
    rets = (s[1:] - s0) / s0
    return float(-(g_per_bar * rets).sum())


@dataclass(frozen=True)
class PGIResidualiser:
    """Fold-frozen residualisation: PGI = a + b1*C_early + b2*C_late + b3*r_day + eps."""

    a: float
    b1: float
    b2: float
    b3: float

    def transform(
        self, pgi: pd.Series, c_early: pd.Series, c_late: pd.Series, r_day: pd.Series
    ) -> pd.Series:
        fitted = self.a + self.b1 * c_early + self.b2 * c_late + self.b3 * r_day
        return (pgi - fitted).rename("PGI_perp")


def residualise_pgi(
    pgi: pd.Series, c_early: pd.Series, c_late: pd.Series, r_day: pd.Series
) -> PGIResidualiser:
    """Fit the momentum regression on training data only; freeze coefficients.

    The caller is responsible for passing *training-fold* series here and
    applying :meth:`PGIResidualiser.transform` forward. The research
    question is thereby sharpened to whether the gamma-weighted path
    contains information beyond the path itself.
    """
    df = pd.concat(
        {"pgi": pgi, "c_early": c_early, "c_late": c_late, "r_day": r_day}, axis=1
    ).dropna()
    if len(df) < 10:
        return PGIResidualiser(0.0, 0.0, 0.0, 0.0)
    X = np.column_stack(
        [np.ones(len(df)), df["c_early"], df["c_late"], df["r_day"]]
    )
    beta, *_ = np.linalg.lstsq(X, df["pgi"].to_numpy(), rcond=None)
    return PGIResidualiser(*(float(b) for b in beta))
