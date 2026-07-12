"""Monte Carlo harness plumbing: determinism, aggregation, recovery fields."""

import numpy as np

from espa.backtest.montecarlo import (
    World,
    one_replication,
    run_monte_carlo,
    summarise_monte_carlo,
)


def test_replication_is_deterministic():
    a = one_replication(World("w", edge=0.5), rep=3, n_days=700)
    b = one_replication(World("w", edge=0.5), rep=3, n_days=700)
    assert a == b


def test_replications_differ_across_seeds():
    a = one_replication(World("w", edge=0.5), rep=0, n_days=700)
    b = one_replication(World("w", edge=0.5), rep=1, n_days=700)
    assert a["sharpe_full"] != b["sharpe_full"] or a["n_trades"] != b["n_trades"]


def test_run_and_summarise_small():
    worlds = [World("null", edge=0.0), World("edge", edge=0.5)]
    df = run_monte_carlo(worlds, n_reps=2, n_days=700, max_workers=2)
    assert len(df) == 4
    assert set(df["world"]) == {"null", "edge"}
    s = summarise_monte_carlo(df)
    assert len(s) == 2
    for col in (
        "accept_rate",
        "zero_trade_rate",
        "beta2_pos_rate",
        "recover_C_early",
        "unstable_noise_rate",
    ):
        assert col in s.columns
        assert s[col].between(0, 1).all() or s[col].isna().any()
    # acceptance is a rate, and both worlds carry their planted params
    assert (s.loc[s["world"] == "null", "edge"] == 0.0).all()
    assert (s.loc[s["world"] == "edge", "edge"] == 0.5).all()
