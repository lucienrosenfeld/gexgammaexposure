"""Threshold selection (Section 16).

Trade frequency is a prior, never an optimisation target:

    theta* = argmax_theta [Sharpe_net_OOS(theta) - kappa*Turnover(theta)
                           - phi*Instability(theta)]

subject to a prespecified feasibility band of 10-50 percent of ordinary
days traded. The live trigger is expressed in forecast-uncertainty
units: trade iff |yhat| / sigma_yhat > z_theta.

Instability is measured as the dispersion of the objective's Sharpe
across chronological halves of the validation sample: a theta whose
performance lives entirely in one half is penalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants, ThresholdPenalty
from espa.targets import executable_pnl


def _sharpe(pnl: pd.Series) -> float:
    traded = pnl[pnl != 0.0]
    if len(traded) < 5 or traded.std(ddof=1) == 0:
        return np.nan
    # annualised on trade days is a comparison statistic here, not the
    # acceptance statistic (that is the deflated Sharpe in validation)
    return float(traded.mean() / traded.std(ddof=1) * np.sqrt(252))


@dataclass
class ThresholdResult:
    z_theta: float
    objective: float
    sharpe: float
    trade_fraction: float
    grid: pd.DataFrame


def select_threshold(
    forecast_z: pd.Series,
    direction: pd.Series,
    targets: pd.DataFrame,
    penalty: ThresholdPenalty = ThresholdPenalty(),
    constants: SpecConstants = DEFAULT_CONSTANTS,
    grid: np.ndarray | None = None,
) -> ThresholdResult:
    """Choose z_theta on validation executable P&L.

    ``forecast_z``: |yhat|/sigma_yhat per day. ``direction``: sign(yhat).
    ``targets``: executable target frame (y_long / y_short).
    """
    if grid is None:
        grid = np.arange(0.0, 3.01, 0.1)
    lo, hi = constants.trade_frequency_band
    n = int(forecast_z.notna().sum())
    rows = []
    for z in grid:
        traded = (forecast_z > z).fillna(False)
        frac = traded.sum() / n if n else 0.0
        d = direction.where(traded, 0.0)
        pnl = executable_pnl(d, targets)
        sharpe = _sharpe(pnl)
        turnover = float(frac)
        instability = _split_half_instability(pnl)
        feasible = lo <= frac <= hi
        obj = (
            sharpe - penalty.kappa * turnover - penalty.phi * instability
            if feasible and np.isfinite(sharpe)
            else -np.inf
        )
        rows.append((z, obj, sharpe, frac, turnover, instability, feasible))
    df = pd.DataFrame(
        rows,
        columns=["z_theta", "objective", "sharpe", "trade_fraction", "turnover", "instability", "feasible"],
    )
    if not np.isfinite(df["objective"]).any():
        # nothing feasible: report the band edge, flat objective — the
        # caller must treat this as 'no validated threshold', not trade
        return ThresholdResult(np.nan, -np.inf, np.nan, 0.0, df)
    best = df.loc[df["objective"].idxmax()]
    return ThresholdResult(
        z_theta=float(best["z_theta"]),
        objective=float(best["objective"]),
        sharpe=float(best["sharpe"]),
        trade_fraction=float(best["trade_fraction"]),
        grid=df,
    )


def _split_half_instability(pnl: pd.Series) -> float:
    traded = pnl[pnl != 0.0]
    if len(traded) < 10:
        return 1.0  # too few trades to establish stability: penalise
    half = len(traded) // 2
    s1, s2 = _sharpe(traded.iloc[:half]), _sharpe(traded.iloc[half:])
    if not (np.isfinite(s1) and np.isfinite(s2)):
        return 1.0
    return float(abs(s1 - s2) / 2.0)
