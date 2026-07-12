"""End-to-end framework verification on synthetic data.

Two properties matter: the pipeline finds a planted edge, and it reports
a null when the edge is absent. Both run on the identical machinery.
"""

import numpy as np
import pytest

from espa.backtest.engine import run_backtest
from espa.backtest.synthetic import generate_synthetic_data
from espa.config import RunConfig
from espa.validation.registry import ConfigurationRegistry


@pytest.fixture(scope="module")
def edge_result():
    data = generate_synthetic_data(n_days=750, edge=0.5, seed=11)
    reg = ConfigurationRegistry()
    cfg = RunConfig(name="edge-test")
    return run_backtest(data, cfg, reg, min_train=252, min_threshold_history=60)


@pytest.fixture(scope="module")
def null_result():
    data = generate_synthetic_data(n_days=750, edge=0.0, seed=12)
    reg = ConfigurationRegistry()
    cfg = RunConfig(name="null-test")
    return run_backtest(data, cfg, reg, min_train=252, min_threshold_history=60)


def test_pipeline_finds_planted_edge(edge_result):
    s = edge_result.summary
    assert s["n_trades"] > 10
    assert s["sharpe_full"] > 0
    # constrained coefficients ended up with admissible signs in every fold
    coefs = edge_result.coef_by_fold
    assert (coefs["C_early"] >= -1e-10).all()
    assert (coefs["I_level"] >= -1e-10).all()
    assert (coefs["B_live"] <= 1e-10).all()


def test_planted_signs_recovered(edge_result):
    """The planted DGP loads + on C_early, + on I_level, - on B_live."""
    mean_coef = edge_result.coef_by_fold.mean()
    assert mean_coef["C_early"] > 0
    assert mean_coef["I_level"] > 0
    assert mean_coef["B_live"] < 0


def test_null_data_reports_null(null_result):
    s = null_result.summary
    # no acceptance-grade edge on pure noise
    assert not (np.isfinite(s["tstat_full"]) and s["tstat_full"] > 3.0)
    assert not s["gate1_pass"]


def test_event_days_scored_but_excluded(edge_result):
    log = edge_result.event_day_log
    assert len(log) > 0  # event days were scored by the frozen model
    # and none of them appear in the selection/trading frame
    assert len(set(log.index) & set(edge_result.daily.index)) == 0


def test_threshold_warmup_prevents_early_trading(edge_result):
    daily = edge_result.daily
    first_days = daily.iloc[:60]
    assert (first_days["direction"] == 0.0).all()  # no threshold history yet


def test_lambda_bounded(edge_result):
    lam = edge_result.daily["lambda"]
    assert (lam >= 0).all() and (lam <= 1).all()
