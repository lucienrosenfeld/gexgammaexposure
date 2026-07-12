"""Session schedule, exit-time rule, and the two research universes.

The 16:15 halt is never hard-coded (Section 2). The engine consumes a
session schedule — in production the exchange reference-data feed, in
research a :class:`SessionSchedule` built from that feed's history — and
computes

    t_exit = min(next_halt - exit_lead, 16:14:30)

If a session has no post-close halt, 16:14:30 remains the endpoint and
the trapped-through-halt failure mode is absent.

Universe membership (Section 20): D_ordinary excludes quarterly futures
and options expirations, FOMC days, month-end and quarter-end, index
rebalance dates, half days, and reconstitution windows. Dates that
require an external calendar (FOMC, MSCI/FTSE Russell) are supplied by
the caller; the computable exclusions are computed here. Event days are
never deleted: they are scored out-of-sample by the frozen model and
logged (see :mod:`espa.backtest.engine`).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from espa.config import DEFAULT_CONSTANTS, SpecConstants


@dataclass(frozen=True)
class Session:
    """One trading session as described by the reference-data feed."""

    date: dt.date
    #: Start of the daily equity-index futures halt, or None if the
    #: session has no post-close halt (e.g. an early close).
    halt_start: dt.time | None = dt.time(16, 15)
    #: True for exchange-shortened sessions.
    half_day: bool = False

    def exit_time(self, constants: SpecConstants = DEFAULT_CONSTANTS) -> dt.time:
        """t_exit = min(next_halt - exit_lead, default endpoint)."""
        default = constants.default_exit_time
        if self.halt_start is None:
            return default
        halt = dt.datetime.combine(self.date, self.halt_start)
        lead = halt - dt.timedelta(seconds=constants.exit_lead_seconds)
        return min(lead.time(), default)


@dataclass
class SessionSchedule:
    """Daily session table, consumed by the engine each morning."""

    sessions: dict[dt.date, Session] = field(default_factory=dict)

    def add(self, session: Session) -> None:
        self.sessions[session.date] = session

    def get(self, date: dt.date) -> Session:
        return self.sessions.get(date, Session(date=date))

    def exit_time(
        self, date: dt.date, constants: SpecConstants = DEFAULT_CONSTANTS
    ) -> dt.time:
        return self.get(date).exit_time(constants)


def _third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    return d


def quarterly_expiries(year: int) -> list[dt.date]:
    """Third Fridays of March, June, September, December."""
    return [_third_friday(year, m) for m in (3, 6, 9, 12)]


def is_month_end(date: dt.date, trading_days: list[dt.date]) -> bool:
    """Last trading day of its month within ``trading_days``."""
    later = [d for d in trading_days if d > date and d.month == date.month and d.year == date.year]
    return len(later) == 0


@dataclass
class UniverseCalendar:
    """Splits trading days into D_ordinary and D_event (Section 20)."""

    #: Externally sourced event dates: FOMC decision days, MSCI and FTSE
    #: Russell rebalance dates, reconstitution windows. The framework
    #: cannot compute these; they are inputs, and an empty list here is a
    #: data gap to be closed in Phase 0, not a statement that none exist.
    external_event_dates: set[dt.date] = field(default_factory=set)
    schedule: SessionSchedule = field(default_factory=SessionSchedule)

    def is_event_day(self, date: dt.date, trading_days: list[dt.date]) -> bool:
        if date in self.external_event_dates:
            return True
        if date in quarterly_expiries(date.year):
            return True
        if is_month_end(date, trading_days):  # covers quarter-end as a subset
            return True
        if self.schedule.get(date).half_day:
            return True
        return False

    def split(
        self, trading_days: list[dt.date]
    ) -> tuple[list[dt.date], list[dt.date]]:
        """Return (D_ordinary, D_event), disjoint and exhaustive."""
        ordinary, event = [], []
        for d in trading_days:
            (event if self.is_event_day(d, trading_days) else ordinary).append(d)
        return ordinary, event
