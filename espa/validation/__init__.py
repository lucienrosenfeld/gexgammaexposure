"""Validation protocol: walk-forward, acceptance metrics, configuration registry."""

from espa.validation.walkforward import walk_forward_folds, WalkForwardFold
from espa.validation.metrics import (
    deflated_sharpe_ratio,
    sharpe_tstat,
    harvey_liu_haircut,
    block_bootstrap_sharpe_ci,
    sign_stability,
)
from espa.validation.registry import ConfigurationRegistry

__all__ = [
    "walk_forward_folds",
    "WalkForwardFold",
    "deflated_sharpe_ratio",
    "sharpe_tstat",
    "harvey_liu_haircut",
    "block_bootstrap_sharpe_ci",
    "sign_stability",
    "ConfigurationRegistry",
]
