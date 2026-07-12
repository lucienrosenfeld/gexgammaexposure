"""Backtest orchestration and synthetic data for framework verification."""

from espa.backtest.engine import StrategyData, BacktestResult, run_backtest
from espa.backtest.synthetic import generate_synthetic_data

__all__ = ["StrategyData", "BacktestResult", "run_backtest", "generate_synthetic_data"]
