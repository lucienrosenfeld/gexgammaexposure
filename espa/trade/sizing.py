"""Position mapping, size, and the estimated volatility conditioner (Section 17).

    p_t = sign(yhat) * min(1, (|yhat| - theta)/theta)   beyond the threshold
    N_t = floor(RiskBudget * |p_t| / (PointValue * sigma* * h(A_t)))

subject to liquidity caps, notional caps, and catastrophe limits.

The volatility conditioner is estimated rather than assumed: on strictly
past data, log|r_window| = a + b*A + eps with b >= 0, and
h(A) = exp(bhat*A) capped to [0.75, 2.0], tested against the null h = 1.
The constraint permits the data to conclude that high unsigned density
coincides with pinning and *lower* post-close volatility, in which case
bhat goes to zero and the conditioner disappears — the correct outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


def position_fraction(y_hat: float, theta: float) -> float:
    """p_t in [-1, 1]; zero inside the dead zone.

    ``theta`` here is in the same units as ``y_hat`` (the raw-forecast
    equivalent of the uncertainty-unit trigger for the day).
    """
    if not np.isfinite(y_hat) or theta <= 0 or abs(y_hat) <= theta:
        return 0.0
    return float(np.sign(y_hat) * min(1.0, (abs(y_hat) - theta) / theta))


@dataclass
class VolatilityConditioner:
    """h(A_t) = exp(bhat * A_t), bhat >= 0, capped."""

    b_hat: float = 0.0
    constants: SpecConstants = DEFAULT_CONSTANTS

    @classmethod
    def fit(
        cls,
        window_returns: pd.Series,
        a_density: pd.Series,
        constants: SpecConstants = DEFAULT_CONSTANTS,
    ) -> "VolatilityConditioner":
        """OLS of log|r_window| on A over strictly past data, b clipped at 0."""
        df = pd.concat({"r": window_returns, "a": a_density}, axis=1).dropna()
        df = df[df["r"].abs() > 0]
        if len(df) < 30:
            return cls(b_hat=0.0, constants=constants)  # the h = 1 null
        y = np.log(df["r"].abs().to_numpy())
        X = np.column_stack([np.ones(len(df)), df["a"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return cls(b_hat=max(0.0, float(beta[1])), constants=constants)

    def h(self, a_t: float) -> float:
        if not np.isfinite(a_t):
            return 1.0
        raw = float(np.exp(self.b_hat * a_t))
        return float(np.clip(raw, self.constants.h_cap_low, self.constants.h_cap_high))


def contracts(
    risk_budget: float,
    p_t: float,
    point_value: float,
    sigma_star_points: float,
    h_at: float = 1.0,
    max_participation: float = 0.05,
    median_window_volume: float | None = None,
    notional_cap_contracts: int | None = None,
) -> int:
    """N_t with liquidity and notional caps.

    ``sigma_star_points`` is sigma* expressed in ES points so the risk
    budget is in currency.
    """
    if p_t == 0.0 or sigma_star_points <= 0:
        return 0
    n = int(np.floor(risk_budget * abs(p_t) / (point_value * sigma_star_points * h_at)))
    if median_window_volume is not None:
        n = min(n, int(max_participation * median_window_volume))
    if notional_cap_contracts is not None:
        n = min(n, notional_cap_contracts)
    return max(n, 0)
