#!/usr/bin/env python3
"""Run the full protocol end-to-end on synthetic data.

This is a framework demonstration, not research: the data is generated
with a known planted edge so every stage of the machinery (stacking,
blending, threshold warmup, acceptance metrics, event-day logging) has
something to exercise. Swap :func:`generate_synthetic_data` for real
prepared data (a StrategyData built from the feature modules) to run the
actual programme.

Usage: python scripts/run_synthetic_backtest.py [--days 750] [--edge 0.5]
"""

import argparse

from espa.backtest.engine import run_backtest
from espa.backtest.synthetic import generate_synthetic_data
from espa.config import RunConfig
from espa.validation.registry import ConfigurationRegistry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=750)
    ap.add_argument("--edge", type=float, default=0.5)
    ap.add_argument("--options-edge", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    config = RunConfig(name=f"synthetic-edge{args.edge}")
    data = generate_synthetic_data(
        n_days=args.days, edge=args.edge, options_edge=args.options_edge,
        seed=args.seed, config=config,
    )
    registry = ConfigurationRegistry()
    result = run_backtest(data, config, registry, min_threshold_history=60)

    s = result.summary
    print("=== ES post-auction strategy: synthetic run ===")
    print(f"days scored OOS:        {s['n_days']}")
    print(f"trades:                 {s['n_trades']}  ({s['trade_fraction']:.1%} of days)")
    print(f"Sharpe (full, per-trade): {s['sharpe_full']:.3f}   t = {s['tstat_full']:.2f}")
    print(f"Sharpe (Stage 1 only):    {s['sharpe_stage1']:.3f}   t = {s['tstat_stage1']:.2f}")
    print(f"deflated SR (full):     {s['dsr_full']:.3f}  vs benchmark SR0 = {s['dsr_benchmark_full']:.4f}")
    print(f"haircut Sharpe (full):  {s['haircut_sharpe_full']:.3f}  over {s['n_trials']} registered trials")
    print(f"bootstrap 95% CI:       {tuple(round(x, 4) for x in s['sharpe_ci_full'])}")
    print(f"mean blend lambda:      {s['mean_lambda']:.3f}")
    print(f"decision gate 1 pass:   {s['gate1_pass']}")
    if s["unstable_features"]:
        print(f"UNSTABLE FEATURES (remove and re-register): {s['unstable_features']}")
    print("\nStage 1 coefficients by fold:")
    print(result.coef_by_fold.round(4).to_string())
    print("\nStage 2 coefficients by fold:")
    print(result.stage2_coef_by_fold.round(4).to_string())
    print(f"\nevent days scored (excluded from selection): {len(result.event_day_log)}")


if __name__ == "__main__":
    main()
