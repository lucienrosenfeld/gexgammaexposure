"""Receipt-time discipline (Section 7).

Rule for every feature: max information timestamp < decision time.

Three inputs are time-critical — the imbalance feed, the SPX close
estimate, and the options surface — and each carries its full timestamp
set so the rule can be *enforced* rather than assumed. Vendor
end-of-day surfaces reconstructed after the fact are look-ahead and are
rejected here, not filtered downstream.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from espa.config import DEFAULT_CONSTANTS, SpecConstants


class LookAheadError(ValueError):
    """A feature input carries information from at or after decision time."""


@dataclass(frozen=True)
class StampedMessage:
    """A feed message with the five-timestamp schema for imbalance data."""

    payload: object
    t_exchange: dt.datetime
    t_vendor: dt.datetime | None
    t_receive: dt.datetime
    t_decision: dt.datetime | None = None
    t_order: dt.datetime | None = None

    def usable_at(self, decision: dt.datetime) -> bool:
        """Demonstrably received before the decision time."""
        return self.t_receive < decision


@dataclass(frozen=True)
class SurfaceStamp:
    """Timestamp set for one options-surface input (quotes/Greeks/OI/vols)."""

    t_quote: dt.datetime | None
    t_trade: dt.datetime | None
    t_vendor_calc: dt.datetime | None
    t_receive: dt.datetime

    def max_information_time(self) -> dt.datetime:
        times = [t for t in (self.t_quote, self.t_trade, self.t_vendor_calc, self.t_receive) if t is not None]
        return max(times)


def assert_no_lookahead(
    information_time: dt.datetime, decision_time: dt.datetime, what: str = "feature"
) -> None:
    if information_time >= decision_time:
        raise LookAheadError(
            f"{what}: information time {information_time} is not strictly before "
            f"decision time {decision_time}"
        )


def filter_imbalance_messages(
    messages: list[StampedMessage],
    date: dt.date,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> list[StampedMessage]:
    """Messages usable by *primary* imbalance features.

    Primary features use only messages demonstrably received by the
    prespecified cutoff (15:59:55). Whether later messages add
    information is a separate, counted test (imbalance spec I2 with its
    receipt-stamped I_last), never an assumption.
    """
    cutoff = dt.datetime.combine(date, constants.imbalance_cutoff)
    return [m for m in messages if m.t_receive <= cutoff]


def last_received_message(
    messages: list[StampedMessage], decision: dt.datetime
) -> StampedMessage | None:
    """Receipt-stamped I_last for spec I2: last message received before decision."""
    usable = [m for m in messages if m.usable_at(decision)]
    if not usable:
        return None
    return max(usable, key=lambda m: m.t_receive)
