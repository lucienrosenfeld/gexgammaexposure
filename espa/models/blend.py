"""Live blending of Stage 1 and the options block (Section 14).

    yhat = Qhat + lambda_t * Ohat

The blend weight compares exponentially weighted forecast losses rather
than an R^2 ratio, because out-of-sample R^2 is frequently negative and
a ratio of two near-zero quantities is unstable:

    L_Q = EWMA_60[(y - Qhat)^2],  L_full = EWMA_60[(y - Qhat - Ohat)^2]
    lambda_t = clip((L_Q - L_full) / (delta * L_Q), 0, 1),  delta = 0.05

A utility gate sits on top: if the options block's incremental
contribution to net P&L after costs over the trailing 60 trades is
negative, lambda_t is forced to zero regardless of the MSE comparison —
a model can reduce squared error inside the dead zone without improving
anything tradable. This is the formula that lets the options block
delete itself in production.

Everything here updates on *realised* outcomes, so today's lambda uses
losses through yesterday only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from espa.config import DEFAULT_CONSTANTS, SpecConstants


@dataclass
class LiveBlender:
    constants: SpecConstants = DEFAULT_CONSTANTS
    _loss_q: float | None = field(default=None, repr=False)
    _loss_full: float | None = field(default=None, repr=False)
    _incremental_pnl: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._incremental_pnl = deque(maxlen=self.constants.utility_gate_window)

    @property
    def _ewma_alpha(self) -> float:
        return 2.0 / (self.constants.blend_loss_span + 1.0)

    def current_lambda(self) -> float:
        """lambda_t from information through the last update only."""
        if self._loss_q is None or self._loss_full is None or self._loss_q <= 0:
            return 0.0  # options block earns its weight; it does not start with it
        if len(self._incremental_pnl) > 0 and sum(self._incremental_pnl) < 0:
            return 0.0  # utility gate
        raw = (self._loss_q - self._loss_full) / (self.constants.blend_delta * self._loss_q)
        return float(np.clip(raw, 0.0, 1.0))

    def blend(self, q_hat: float, o_hat: float) -> float:
        if not np.isfinite(o_hat):
            return q_hat
        return q_hat + self.current_lambda() * o_hat

    def update(
        self,
        y_realised: float,
        q_hat: float,
        o_hat: float,
        incremental_pnl: float | None = None,
    ) -> None:
        """Record one day's realised outcome after the window closes.

        ``incremental_pnl``: net executable P&L of the blended strategy
        minus the Stage-1-only strategy for this trade, if a trade
        occurred; None on flat days (flat days do not consume a slot in
        the trailing-60-trade gate).
        """
        if not (np.isfinite(y_realised) and np.isfinite(q_hat)):
            return
        a = self._ewma_alpha
        lq = (y_realised - q_hat) ** 2
        self._loss_q = lq if self._loss_q is None else (1 - a) * self._loss_q + a * lq
        if np.isfinite(o_hat):
            lf = (y_realised - (q_hat + o_hat)) ** 2
            self._loss_full = (
                lf if self._loss_full is None else (1 - a) * self._loss_full + a * lf
            )
        if incremental_pnl is not None and np.isfinite(incremental_pnl):
            self._incremental_pnl.append(incremental_pnl)
