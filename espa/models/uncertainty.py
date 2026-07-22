"""Forecast uncertainty from fold-level coefficient dispersion (Section 16).

The live trigger is |yhat| / sigma_yhat > z_theta, with sigma_yhat
propagated from the dispersion of coefficient estimates across refits
through today's feature vector. This prevents identical raw predictions
from being treated equally when model uncertainty differs across refits.

Do not conflate this cross-refit dispersion with the fold-level
bootstrap SEs of espa.models.stage2.bootstrap_coef_se: this class
measures across-refit model drift and serves the trade trigger; the
fold SEs measure within-fold sampling noise and serve the
D-STAGE2-ID-01 materiality floor. They are different quantities with
different consumers (amendment section 2.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForecastUncertainty:
    """Tracks coefficient vectors across folds/refits; propagates to x_t."""

    coef_history: list[np.ndarray] = field(default_factory=list)
    min_history: int = 3
    #: Floor on sigma_yhat as a fraction of the coefficient-implied scale,
    #: so an accidental run of identical refits cannot make every forecast
    #: look infinitely certain.
    rel_floor: float = 0.05

    def record(self, coef: np.ndarray) -> None:
        self.coef_history.append(np.asarray(coef, dtype=float))

    def sigma(self, x: np.ndarray) -> float:
        """Dispersion of x @ w over recorded coefficient vectors."""
        if len(self.coef_history) < self.min_history:
            return np.nan
        x = np.asarray(x, dtype=float)
        preds = np.array([x @ w for w in self.coef_history])
        spread = float(np.std(preds, ddof=1))
        mean_coef = np.mean(np.abs(np.vstack(self.coef_history)), axis=0)
        scale = float(np.abs(x) @ mean_coef)
        return max(spread, self.rel_floor * scale)

    def z_units(self, y_hat: float, x: np.ndarray) -> float:
        """|yhat| in forecast-uncertainty units; NaN until enough history."""
        s = self.sigma(x)
        if not np.isfinite(s) or s <= 0:
            return np.nan
        return abs(y_hat) / s
