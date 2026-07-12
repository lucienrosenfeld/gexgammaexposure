"""Sessions, timestamps, execution protocol, kill switches, validation machinery."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from espa.config import DEFAULT_CONSTANTS, RunConfig
from espa.risk.killswitch import HaltReason, KillSwitchPanel
from espa.sessions import Session, SessionSchedule, UniverseCalendar, quarterly_expiries
from espa.timestamps import (
    LookAheadError,
    StampedMessage,
    assert_no_lookahead,
    filter_imbalance_messages,
    last_received_message,
)
from espa.trade.execution import (
    CatastropheRule,
    ExitReason,
    ExecutionProtocol,
    MarketState,
    OrderType,
    TradePlan,
)
from espa.trade.sizing import VolatilityConditioner, contracts, position_fraction
from espa.validation.metrics import deflated_sharpe_ratio, sign_stability
from espa.validation.registry import ConfigurationRegistry, PRESPECIFIED_CONFIGURATIONS
from espa.validation.walkforward import walk_forward_folds


def test_exit_time_never_hardcoded():
    d = dt.date(2026, 7, 10)
    normal = Session(date=d, halt_start=dt.time(16, 15))
    assert normal.exit_time() == dt.time(16, 14, 30)
    early_halt = Session(date=d, halt_start=dt.time(16, 10))
    assert early_halt.exit_time() == dt.time(16, 9, 30)  # halt - 30s wins
    no_halt = Session(date=d, halt_start=None)
    assert no_halt.exit_time() == dt.time(16, 14, 30)  # research endpoint


def test_universe_split_disjoint_exhaustive():
    days = list(pd.bdate_range("2026-03-01", "2026-04-10").date)
    cal = UniverseCalendar(external_event_dates={days[3]})
    ordinary, event = cal.split(days)
    assert set(ordinary) | set(event) == set(days)
    assert set(ordinary) & set(event) == set()
    assert days[3] in event
    # quarterly expiry (third Friday of March) is an event day
    exp = quarterly_expiries(2026)[0]
    if exp in days:
        assert exp in event


def test_receipt_time_discipline():
    d = dt.date(2026, 7, 10)
    decision = dt.datetime.combine(d, DEFAULT_CONSTANTS.decision_time)
    early = StampedMessage("a", t_exchange=dt.datetime.combine(d, dt.time(15, 55)),
                           t_vendor=None, t_receive=dt.datetime.combine(d, dt.time(15, 55, 1)))
    late = StampedMessage("b", t_exchange=dt.datetime.combine(d, dt.time(16, 0)),
                          t_vendor=None, t_receive=dt.datetime.combine(d, dt.time(16, 0, 20)))
    assert filter_imbalance_messages([early, late], d) == [early]
    assert last_received_message([early, late], decision) == early
    with pytest.raises(LookAheadError):
        assert_no_lookahead(decision, decision)


def test_execution_plan_and_catastrophe():
    d = dt.date(2026, 7, 10)
    plan = TradePlan(date=d, side=1, quantity=5, session=Session(date=d))
    intents = plan.entry_intents()
    assert intents[0].order_type == OrderType.LIMIT_INSIDE_TOUCH
    assert intents[1].order_type == OrderType.MARKETABLE
    assert plan.exit_intent(5).side == -1
    cat = CatastropheRule(sigma_window_dollars=1000.0, dollar_limit=2000.0)
    assert cat.l_cat == 4000.0  # max(4*sigma, L_dollar)
    proto = ExecutionProtocol(plan=plan, catastrophe=cat, max_spread=1.0,
                              intended_quantity=5, executed_quantity=5)
    t = dt.datetime.combine(d, dt.time(16, 5))
    ok = MarketState(time=t, bid=4000.0, ask=4000.25, quote_age_seconds=1.0)
    assert proto.check(ok, pnl_dollars=100.0) is None
    assert proto.check(ok, pnl_dollars=-5000.0) == ExitReason.CATASTROPHE
    after = MarketState(time=dt.datetime.combine(d, dt.time(16, 14, 31)),
                        bid=4000.0, ask=4000.25, quote_age_seconds=1.0)
    assert proto.check(after, 0.0) == ExitReason.TIME


def test_no_ordinary_price_stop():
    """A moderate adverse move inside the window must NOT trigger an exit."""
    d = dt.date(2026, 7, 10)
    plan = TradePlan(date=d, side=1, quantity=1, session=Session(date=d))
    cat = CatastropheRule(sigma_window_dollars=150.0, dollar_limit=500.0)
    proto = ExecutionProtocol(plan=plan, catastrophe=cat, max_spread=1.0,
                              intended_quantity=1, executed_quantity=1)
    t = dt.datetime.combine(d, dt.time(16, 8))
    m = MarketState(time=t, bid=3990.0, ask=3990.25, quote_age_seconds=1.0)
    assert proto.check(m, pnl_dollars=-500.0) is None  # inside L_cat: hold


def test_position_mapping():
    assert position_fraction(0.5, theta=1.0) == 0.0  # dead zone
    assert position_fraction(1.5, theta=1.0) == pytest.approx(0.5)
    assert position_fraction(-3.0, theta=1.0) == -1.0  # capped
    n = contracts(risk_budget=100_000, p_t=1.0, point_value=50.0,
                  sigma_star_points=4.0, h_at=1.0)
    assert n == 500
    n_capped = contracts(100_000, 1.0, 50.0, 4.0, 1.0,
                         max_participation=0.05, median_window_volume=1000)
    assert n_capped == 50


def test_vol_conditioner_null_and_positive():
    rng = np.random.default_rng(3)
    idx = pd.RangeIndex(300)
    a = pd.Series(rng.normal(0, 1, 300), index=idx)
    # null: |r| independent of A -> b ~ 0
    r0 = pd.Series(rng.lognormal(0, 0.3, 300), index=idx)
    v0 = VolatilityConditioner.fit(r0, a)
    assert v0.h(2.0) < 1.3
    # planted: |r| grows with A -> b > 0, capped
    r1 = pd.Series(np.exp(0.5 * a + rng.normal(0, 0.1, 300)), index=idx)
    v1 = VolatilityConditioner.fit(r1, a)
    assert v1.b_hat > 0.3
    assert v1.h(10.0) == DEFAULT_CONSTANTS.h_cap_high
    assert v1.h(-10.0) == DEFAULT_CONSTANTS.h_cap_low


def test_kill_switches():
    panel = KillSwitchPanel()
    today = dt.date(2026, 7, 10)
    for i in range(35):  # 35 losing trades: Sharpe well below -1
        panel.record_trade(today - dt.timedelta(days=35 - i), -1.0 + 0.01 * (i % 3))
    assert panel.halted and HaltReason.PERFORMANCE in panel.halt_reasons

    p2 = KillSwitchPanel()
    for _ in range(30):
        p2.record_slippage(realised=3.0, modelled=1.0)
    assert HaltReason.SLIPPAGE in p2.halt_reasons

    p3 = KillSwitchPanel()
    p3.record_slippage(realised=10.0, modelled=1.0)
    assert HaltReason.SLIPPAGE_EXTREME in p3.halt_reasons

    p4 = KillSwitchPanel()
    p4.feed_change("imbalance feed format change")
    assert p4.halted


def test_walk_forward_embargo():
    idx = pd.RangeIndex(600)
    folds = walk_forward_folds(idx, min_train=252)
    assert len(folds) >= 3
    for f in folds:
        gap = f.validation_index[0] - f.train_index[-1]
        assert gap > DEFAULT_CONSTANTS.embargo_days  # purge + embargo
        assert f.train_index[-1] < f.validation_index[0]  # expanding, no overlap


def test_registry_counts_honestly(tmp_path):
    reg = ConfigurationRegistry(path=tmp_path / "registry.jsonl")
    assert reg.trial_count() == len(PRESPECIFIED_CONFIGURATIONS)  # floor
    reg.log(RunConfig(name="a", rho=0.0))
    reg.log(RunConfig(name="b", rho=0.25))
    reg.log(RunConfig(name="a-again", rho=0.0))  # duplicate key
    assert reg.trial_count() == len(PRESPECIFIED_CONFIGURATIONS)
    # persistence is append-only
    reg2 = ConfigurationRegistry(path=tmp_path / "registry.jsonl")
    assert len(reg2.entries) == 3


def test_deflated_sharpe_punishes_trials():
    rng = np.random.default_rng(4)
    pnl = pd.Series(rng.normal(0.05, 1.0, 500))
    dsr_few, _ = deflated_sharpe_ratio(pnl, n_trials=2)
    dsr_many, _ = deflated_sharpe_ratio(pnl, n_trials=500)
    assert dsr_many < dsr_few


def test_sign_stability_flags_flippers():
    coefs = pd.DataFrame(
        {"stable": [0.5, 0.4, 0.6], "flipper": [0.5, -0.4, 0.6], "shrunk": [0.3, 0.0, 0.2]}
    )
    s = sign_stability(coefs)
    assert s["stable"] == 1.0
    assert s["flipper"] < 1.0
    assert s["shrunk"] == 1.0  # zero is neutral, not a flip
