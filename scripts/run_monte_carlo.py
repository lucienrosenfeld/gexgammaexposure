#!/usr/bin/env python3
"""Monte Carlo calibration of the framework's Type I / Type II properties.

Runs seeded replications of the full protocol across synthetic worlds and
reports acceptance error rates, power by effect size, lambda behaviour,
and coefficient recovery. See espa/backtest/montecarlo.py.

Usage: python scripts/run_monte_carlo.py [--reps 200] [--days 1000] [--out results.csv]
"""

import argparse
import time

import pandas as pd

from espa.backtest.montecarlo import World, run_monte_carlo, summarise_monte_carlo

DEFAULT_WORLDS = [
    World("null", edge=0.0),
    World("stage1-weak", edge=0.25),
    World("stage1-med", edge=0.5),
    World("options-small", edge=0.5, options_edge=0.15),
    World("options-large", edge=0.5, options_edge=0.4),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="CSV path for per-rep rows")
    args = ap.parse_args()

    t0 = time.time()
    df = run_monte_carlo(
        DEFAULT_WORLDS, n_reps=args.reps, n_days=args.days,
        base_seed=args.seed, max_workers=args.workers,
    )
    elapsed = time.time() - t0
    if args.out:
        df.to_csv(args.out, index=False)

    summary = summarise_monte_carlo(df)
    pd.set_option("display.width", 200)
    print(f"\n{len(df)} replications in {elapsed/60:.1f} min "
          f"({elapsed/len(df):.1f} s/rep)\n")
    print("=== Error rates and trading behaviour ===")
    cols1 = ["world", "n_reps", "edge", "options_edge", "accept_rate",
             "zero_trade_rate", "median_trades", "mean_trade_fraction"]
    print(summary[cols1].round(3).to_string(index=False))
    print("\n=== Performance and blend behaviour (traded reps) ===")
    cols2 = ["world", "sharpe_mean", "sharpe_q10", "sharpe_q90", "tstat_mean",
             "lambda_median", "lambda_q90"]
    print(summary[cols2].round(3).to_string(index=False))
    print("\n=== Coefficient recovery ===")
    cols3 = ["world", "beta1_mean", "beta2_mean", "beta2_pos_rate",
             "recover_C_early", "recover_I_level", "recover_B_live",
             "unstable_planted_rate", "unstable_noise_rate"]
    print(summary[cols3].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
