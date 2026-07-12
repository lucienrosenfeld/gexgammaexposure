"""Walk-forward backtest engine.

Orchestrates the full protocol: purged/embargoed expanding walk-forward
(Section 22), the stacking discipline per training window (Section 13),
sequential live blending with the utility gate (Section 14), the
uncertainty-scaled trigger with a threshold chosen only on *prior*
out-of-sample history (Section 16), executable-P&L accounting
(Section 9), and the ordinary/event universe split (Section 20 — the
model is fitted only on D_ordinary; event days are scored by the frozen
model and logged, never influencing selection).

Options-era discipline (Section 21): Stage 2 features are masked to NaN
before the sample split date, so pre-2022 data informs Stage 1 only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, RunConfig, SpecConstants
from espa.models.blend import LiveBlender
from espa.models.stage2 import fit_stage2_stacked
from espa.models.uncertainty import ForecastUncertainty
from espa.targets import executable_pnl
from espa.trade.threshold import select_threshold
from espa.validation.metrics import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    harvey_liu_haircut,
    sharpe_tstat,
    sign_stability,
)
from espa.validation.registry import ConfigurationRegistry
from espa.validation.walkforward import walk_forward_folds


@dataclass
class StrategyData:
    """Prepared daily inputs. Index of every member: trading day.

    Feature construction (and its receipt-time discipline) happens
    upstream in :mod:`espa.features`; the engine assumes everything here
    is decision-time safe and enforces only the protocol-level rules.
    """

    stage1_features: pd.DataFrame
    constraints: dict[str, int]
    y_mid: pd.Series
    targets: pd.DataFrame  # y_mid / y_long / y_short
    a_density: pd.Series
    pgi_raw: pd.Series
    c_early: pd.Series
    c_late: pd.Series
    r_day: pd.Series
    pinning: pd.Series
    ordinary: pd.Series  # bool: True -> D_ordinary
    options_era: pd.Series  # bool: True -> post sample-split date


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    coef_by_fold: pd.DataFrame
    stage2_coef_by_fold: pd.DataFrame
    summary: dict
    event_day_log: pd.DataFrame

    def __repr__(self) -> str:
        keys = ("n_trades", "sharpe_full", "tstat_full", "dsr_full", "haircut_sharpe_full")
        parts = ", ".join(f"{k}={self.summary.get(k)}" for k in keys)
        return f"BacktestResult({parts})"


def run_backtest(
    data: StrategyData,
    config: RunConfig,
    registry: ConfigurationRegistry,
    lam_stage1: float = 5.0,
    lam_stage2: float = 5.0,
    min_train: int = 252,
    min_threshold_history: int = 120,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> BacktestResult:
    idx = data.stage1_features.index
    ordinary_idx = idx[data.ordinary.reindex(idx).fillna(False)]

    # Section 21: options features exist only in the options era.
    era = data.options_era.reindex(idx).fillna(False)
    a_density = data.a_density.where(era)
    pgi_raw = data.pgi_raw.where(era)
    pinning = data.pinning.where(era)

    folds = walk_forward_folds(ordinary_idx, min_train=min_train, constants=constants)

    blender = LiveBlender(constants=constants)
    uncertainty = ForecastUncertainty()

    rows: list[dict] = []
    coef_rows: list[pd.Series] = []
    s2_coef_rows: list[pd.Series] = []
    event_rows: list[dict] = []

    for fold in folds:
        tr = fold.train_index
        fit = fit_stage2_stacked(
            stage1_features=data.stage1_features.loc[tr],
            constraints=data.constraints,
            y_mid=data.y_mid.loc[tr],
            a_density=a_density.loc[tr],
            pgi_raw=pgi_raw.loc[tr],
            c_early=data.c_early.loc[tr],
            c_late=data.c_late.loc[tr],
            r_day=data.r_day.loc[tr],
            pinning=pinning.loc[tr],
            lam_stage1=lam_stage1,
            lam_stage2=lam_stage2,
        )
        coef_rows.append(fit.stage1.coef.rename(fold.fold))
        s2_coef_rows.append(fit.stage2.coef.rename(fold.fold))
        uncertainty.record(fit.stage1.coef.to_numpy())

        # Event days falling inside this fold's date range: scored by the
        # frozen model, logged, never selected on.
        lo, hi = fold.validation_index[0], fold.validation_index[-1]
        event_days = idx[(idx >= lo) & (idx <= hi) & ~data.ordinary.reindex(idx).fillna(False)]
        for span, sink, is_event in ((fold.validation_index, rows, False), (event_days, event_rows, True)):
            fc = fit.forecast(
                stage1_features=data.stage1_features.loc[span],
                a_density=a_density.loc[span],
                pgi_raw=pgi_raw.loc[span],
                c_early=data.c_early.loc[span],
                c_late=data.c_late.loc[span],
                r_day=data.r_day.loc[span],
                pinning=pinning.loc[span],
            )
            for day in span:
                q = float(fc.loc[day, "Q_hat"])
                o = float(fc.loc[day, "O_hat"])
                lam_t = blender.current_lambda()
                y_hat = blender.blend(q, o) if np.isfinite(q) else np.nan
                x_day = data.stage1_features.loc[day].to_numpy(dtype=float)
                z = (
                    uncertainty.z_units(y_hat, x_day)
                    if np.isfinite(y_hat) and not np.isnan(x_day).any()
                    else np.nan
                )
                sink.append(
                    {
                        "day": day,
                        "fold": fold.fold,
                        "Q_hat": q,
                        "O_hat": o,
                        "lambda": lam_t,
                        "y_hat": y_hat,
                        "forecast_z": z,
                        "y_mid": float(data.y_mid.get(day, np.nan)),
                    }
                )
                # The blender learns from realised ordinary-day outcomes
                # only; event days never touch selection state.
                if not is_event and np.isfinite(q):
                    y_real = data.y_mid.get(day, np.nan)
                    if np.isfinite(y_real):
                        inc = _incremental_pnl(day, o, q, data.targets)
                        blender.update(float(y_real), q, o, incremental_pnl=inc)

    daily = pd.DataFrame(rows).set_index("day").sort_index()
    event_log = (
        pd.DataFrame(event_rows).set_index("day").sort_index()
        if event_rows
        else pd.DataFrame()
    )

    daily = _apply_threshold_sequentially(
        daily, data.targets, config, constants, min_threshold_history
    )

    summary = _summarise(daily, data.targets, registry, config)
    coef_by_fold = pd.DataFrame(coef_rows)
    stability = sign_stability(coef_by_fold)
    summary["sign_stability"] = stability.to_dict()
    # Section 22: a feature that flips sign across adjacent folds is
    # removed regardless of aggregate performance. The engine reports;
    # the researcher reruns without the feature (a new registry entry).
    summary["unstable_features"] = list(stability.index[stability < 1.0])
    registry.log(config, result={k: summary[k] for k in ("n_trades", "sharpe_full", "tstat_full")})

    return BacktestResult(
        daily=daily,
        coef_by_fold=coef_by_fold,
        stage2_coef_by_fold=pd.DataFrame(s2_coef_rows),
        summary=summary,
        event_day_log=event_log,
    )


def _incremental_pnl(day, o_hat: float, q_hat: float, targets: pd.DataFrame) -> float | None:
    """Options-block incremental executable P&L for the utility gate.

    Measured counterfactually at full options weight — sign(Q+O) versus
    sign(Q) — regardless of the live lambda, so the trailing window keeps
    rolling while the gate is shut and can reopen when the block starts
    helping again. Days where the two directions agree contribute 0.0.
    """
    if not (np.isfinite(o_hat) and np.isfinite(q_hat)):
        return None
    d_full = float(np.sign(q_hat + o_hat))
    d_q = float(np.sign(q_hat))
    if d_full == d_q:
        return 0.0
    row = targets.loc[[day]]
    pnl_full = float(executable_pnl(pd.Series([d_full], index=[day]), row).iloc[0])
    pnl_q = float(executable_pnl(pd.Series([d_q], index=[day]), row).iloc[0])
    return pnl_full - pnl_q


def _apply_threshold_sequentially(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
    config: RunConfig,
    constants: SpecConstants,
    min_history: int,
) -> pd.DataFrame:
    """Refit-cadence threshold selection on strictly prior OOS history.

    For each refit block, z_theta is chosen on all out-of-sample days
    before the block. Days before enough history exists are not traded —
    the warmup is the price of never selecting a threshold on data it
    will be applied to.
    """
    daily = daily.copy()
    daily["z_theta"] = np.nan
    daily["direction"] = 0.0
    days = daily.index
    step = constants.refit_frequency_days
    for start in range(0, len(days), step):
        block = days[start : start + step]
        hist = days[:start]
        if len(hist) < min_history:
            continue
        h = daily.loc[hist]
        res = select_threshold(
            forecast_z=h["forecast_z"],
            direction=np.sign(h["y_hat"]),
            targets=targets.loc[hist],
            penalty=config.threshold_penalty,
            constants=constants,
        )
        if np.isnan(res.z_theta):
            continue  # no valid selection possible (e.g. no feasible theta)
        # z_theta = +inf is a real selection: flat is admissible, and the
        # comparison below then trades nothing for the block.
        daily.loc[block, "z_theta"] = res.z_theta
        traded = daily.loc[block, "forecast_z"] > res.z_theta
        daily.loc[block, "direction"] = np.sign(daily.loc[block, "y_hat"]).where(traded, 0.0)
    daily["pnl"] = executable_pnl(daily["direction"], targets.reindex(daily.index))
    # Stage-1-only fallback, tracked in parallel: the bar Stage 2 must beat.
    daily["direction_q"] = np.sign(daily["Q_hat"]).where(daily["direction"] != 0.0, 0.0)
    daily["pnl_stage1"] = executable_pnl(daily["direction_q"], targets.reindex(daily.index))
    return daily


def _summarise(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
    registry: ConfigurationRegistry,
    config: RunConfig,
) -> dict:
    n_trials = registry.trial_count()
    out: dict = {"config": config.name, "n_trials": n_trials}
    for tag, col in (("full", "pnl"), ("stage1", "pnl_stage1")):
        pnl = daily[col]
        traded = pnl[pnl != 0.0]
        sr, t = sharpe_tstat(pnl)
        dsr, sr0 = deflated_sharpe_ratio(pnl, n_trials=n_trials)
        hc = harvey_liu_haircut(sr, len(traded), n_trials)
        lo, hi = block_bootstrap_sharpe_ci(pnl)
        out.update(
            {
                f"sharpe_{tag}": sr,
                f"tstat_{tag}": t,
                f"dsr_{tag}": dsr,
                f"dsr_benchmark_{tag}": sr0,
                f"haircut_sharpe_{tag}": hc,
                f"sharpe_ci_{tag}": (lo, hi),
            }
        )
    out["n_trades"] = int((daily["pnl"] != 0.0).sum())
    out["n_days"] = int(len(daily))
    out["trade_fraction"] = out["n_trades"] / out["n_days"] if out["n_days"] else np.nan
    out["mean_lambda"] = float(daily["lambda"].mean())
    # Decision gate 1 (Section 12): the acceptance test on Stage 1 alone.
    out["gate1_pass"] = bool(
        np.isfinite(out["tstat_stage1"]) and out["tstat_stage1"] > DEFAULT_CONSTANTS.acceptance_tstat
    )
    return out
