"""Execution protocol (Section 19), as a testable state machine.

Entry: limit one tick inside the touch at 16:00:15, 15-second timeout,
then marketable. Exit: sent marketable at t_exit with no passive attempt
— the deadline sits 30 seconds ahead of a trading halt, and a missed
exit means carrying the position through the halt into the 16:30
reopen. Confirmation check 15 seconds after the exit order, aggressive
re-submission on any miss.

No ordinary price stop: a tight technical stop inside a 14-minute
thin-book window converts microstructure noise into realised losses.
Risk controls are size, the dead zone, the time exit, and a catastrophe
rule whose purpose is survival rather than alpha:

    exit immediately iff |PnL| > L_cat = max(4 * sigma_window, L_dollar)

plus operational exits on stale quotes, locked/crossed abnormal markets,
spread beyond a threshold, feed loss, or quantity mismatch.

This module contains no broker connectivity; it emits order intents that
an adapter executes, which is what makes the protocol unit-testable.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field

import numpy as np

from espa.config import DEFAULT_CONSTANTS, SpecConstants
from espa.sessions import Session


class OrderType(enum.Enum):
    LIMIT_INSIDE_TOUCH = "limit_inside_touch"
    MARKETABLE = "marketable"


class ExitReason(enum.Enum):
    TIME = "time_exit"
    CATASTROPHE = "catastrophe"
    STALE_QUOTES = "stale_quotes"
    ABNORMAL_MARKET = "abnormal_market"
    WIDE_SPREAD = "wide_spread"
    FEED_LOSS = "feed_loss"
    QUANTITY_MISMATCH = "quantity_mismatch"


@dataclass(frozen=True)
class OrderIntent:
    side: int  # +1 buy, -1 sell
    quantity: int
    order_type: OrderType
    not_after: dt.datetime | None = None


@dataclass(frozen=True)
class CatastropheRule:
    sigma_window_dollars: float
    dollar_limit: float
    constants: SpecConstants = DEFAULT_CONSTANTS

    @property
    def l_cat(self) -> float:
        return max(
            self.constants.catastrophe_sigma_mult * self.sigma_window_dollars,
            self.dollar_limit,
        )

    def breached(self, pnl_dollars: float) -> bool:
        return abs(pnl_dollars) > self.l_cat


@dataclass(frozen=True)
class MarketState:
    time: dt.datetime
    bid: float
    ask: float
    quote_age_seconds: float
    feed_ok: bool = True

    @property
    def locked_or_crossed(self) -> bool:
        return self.bid >= self.ask

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class TradePlan:
    """One day's execution plan, derived from the session schedule."""

    date: dt.date
    side: int
    quantity: int
    session: Session
    constants: SpecConstants = DEFAULT_CONSTANTS

    @property
    def decision_time(self) -> dt.datetime:
        return dt.datetime.combine(self.date, self.constants.decision_time)

    @property
    def exit_deadline(self) -> dt.datetime:
        return dt.datetime.combine(self.date, self.session.exit_time(self.constants))

    def entry_intents(self) -> list[OrderIntent]:
        """Passive attempt with timeout, then marketable."""
        passive_until = self.decision_time + dt.timedelta(
            seconds=self.constants.entry_timeout_seconds
        )
        return [
            OrderIntent(self.side, self.quantity, OrderType.LIMIT_INSIDE_TOUCH, passive_until),
            OrderIntent(self.side, self.quantity, OrderType.MARKETABLE, self.exit_deadline),
        ]

    def exit_intent(self, quantity_held: int) -> OrderIntent:
        """Marketable, no passive attempt — the halt is behind the deadline."""
        return OrderIntent(-self.side, quantity_held, OrderType.MARKETABLE, None)


@dataclass
class ExecutionProtocol:
    """Evaluates operational-exit conditions during the holding window."""

    plan: TradePlan
    catastrophe: CatastropheRule
    max_spread: float
    max_quote_age_seconds: float = 5.0
    intended_quantity: int = 0
    executed_quantity: int = 0
    exit_reasons: list[ExitReason] = field(default_factory=list)

    def check(self, market: MarketState, pnl_dollars: float) -> ExitReason | None:
        """Return the exit reason if any condition fires; None to hold."""
        if market.time >= self.plan.exit_deadline:
            return self._flag(ExitReason.TIME)
        if self.catastrophe.breached(pnl_dollars):
            return self._flag(ExitReason.CATASTROPHE)
        if not market.feed_ok:
            return self._flag(ExitReason.FEED_LOSS)
        if market.quote_age_seconds > self.max_quote_age_seconds:
            return self._flag(ExitReason.STALE_QUOTES)
        if market.locked_or_crossed:
            return self._flag(ExitReason.ABNORMAL_MARKET)
        if market.spread > self.max_spread:
            return self._flag(ExitReason.WIDE_SPREAD)
        if self.executed_quantity != self.intended_quantity:
            return self._flag(ExitReason.QUANTITY_MISMATCH)
        return None

    def _flag(self, reason: ExitReason) -> ExitReason:
        self.exit_reasons.append(reason)
        return reason
