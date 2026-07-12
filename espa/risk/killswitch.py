"""Kill switches (Section 23). All are halt-and-review, none auto-resume.

Performance: halt iff Sharpe over max(60 calendar days, 30 trades) < -1.
The trade-count minimum exists because a sparse strategy can put only
18-24 trades in 60 days and a Sharpe on that few observations is noise.

Forecast degradation: halt iff the cumulative loss gap versus a null
model over the recent n observations exceeds a threshold. Three nulls
are monitored: always-flat for the full strategy, Stage 1 for the
options block, imbalance-sign-only for Stage 1 itself.

Slippage: halt iff EWMA_20(realised/modelled) > 2, plus a single-trade
extreme limit. A consecutive-count rule is rejected because one normal
fill resetting a run of severe misses is exactly the wrong behaviour.

Feed integrity: any change in the imbalance feed's format or timing, the
SPX close feed, or the options vendor's methodology halts pending
review. The strategy's edge, if it exists, lives in those feeds.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field

import numpy as np

from espa.config import DEFAULT_CONSTANTS, SpecConstants


class HaltReason(enum.Enum):
    PERFORMANCE = "performance"
    FORECAST_DEGRADATION = "forecast_degradation"
    SLIPPAGE = "slippage"
    SLIPPAGE_EXTREME = "slippage_extreme"
    FEED_INTEGRITY = "feed_integrity"


@dataclass
class KillSwitchPanel:
    constants: SpecConstants = DEFAULT_CONSTANTS
    #: Threshold on the cumulative (model loss - null loss) gap.
    degradation_threshold: float = 5.0
    degradation_window: int = 60
    #: Single-trade slippage extreme, as realised/modelled ratio.
    slippage_extreme_ratio: float = 5.0

    _trades: list[tuple[dt.date, float]] = field(default_factory=list)
    _loss_gaps: dict[str, list[float]] = field(default_factory=dict)
    _slippage_ewma: float | None = None
    halted: bool = False
    halt_reasons: list[HaltReason] = field(default_factory=list)

    # --- performance ---
    def record_trade(self, date: dt.date, pnl: float) -> None:
        self._trades.append((date, pnl))
        self._check_performance(date)

    def _check_performance(self, today: dt.date) -> None:
        window_start = today - dt.timedelta(days=self.constants.kill_sharpe_days)
        recent = [p for d, p in self._trades if d >= window_start]
        # widen back in trade count until both minima are satisfied
        if len(recent) < self.constants.kill_sharpe_min_trades:
            recent = [p for _, p in self._trades[-self.constants.kill_sharpe_min_trades:]]
        if len(recent) < self.constants.kill_sharpe_min_trades:
            return  # not enough history for the statistic to mean anything
        arr = np.asarray(recent)
        sd = arr.std(ddof=1)
        if sd <= 0:
            return
        sharpe = arr.mean() / sd * np.sqrt(252)
        if sharpe < self.constants.kill_sharpe_level:
            self._halt(HaltReason.PERFORMANCE)

    # --- forecast degradation (per monitored null) ---
    def record_losses(self, null_name: str, model_loss: float, null_loss: float) -> None:
        gaps = self._loss_gaps.setdefault(null_name, [])
        gaps.append(model_loss - null_loss)
        recent = gaps[-self.degradation_window:]
        if len(recent) >= self.degradation_window and sum(recent) > self.degradation_threshold:
            self._halt(HaltReason.FORECAST_DEGRADATION)

    # --- slippage ---
    def record_slippage(self, realised: float, modelled: float) -> None:
        if modelled <= 0:
            return
        ratio = realised / modelled
        if ratio > self.slippage_extreme_ratio:
            self._halt(HaltReason.SLIPPAGE_EXTREME)
        a = 2.0 / (self.constants.slippage_ewma_span + 1.0)
        self._slippage_ewma = (
            ratio
            if self._slippage_ewma is None
            else (1 - a) * self._slippage_ewma + a * ratio
        )
        if self._slippage_ewma > self.constants.slippage_ratio_limit:
            self._halt(HaltReason.SLIPPAGE)

    # --- feed integrity ---
    def feed_change(self, description: str) -> None:
        self._halt(HaltReason.FEED_INTEGRITY)

    def _halt(self, reason: HaltReason) -> None:
        self.halted = True
        if reason not in self.halt_reasons:
            self.halt_reasons.append(reason)

    def clear_after_review(self) -> None:
        """Manual resume only; the panel never un-halts itself."""
        self.halted = False
        self.halt_reasons.clear()
