"""Monte Carlo harness over synthetic worlds.

One draw per world is one draw. This module runs each world as N seeded
replications and converts single-run anecdotes into measured error
rates:

- **False-positive rate** of the full acceptance criteria (deflated
  Sharpe positive at t > 3) in the null world — the number that must be
  near zero before the acceptance machinery is trusted.
- **Power** in edge worlds as a function of planted effect size — the
  number that determines how many days of history acceptance realistically
  requires at effect sizes actually believed in.
- **Lambda distribution** per world — does the blend weight rise above
  its noise floor only when an options edge exists?
- **Coefficient recovery** — sign and magnitude of the Stage 1
  constrained coefficients and the Stage 2 betas against the planted
  data-generating process, and how often the sign-stability filter fires
  on genuinely-noise features.

Every replication runs the identical full protocol as a single backtest;
nothing is shortcut. Registries are per-replication (floored at the
prespecified configuration count), because the multiple-testing
denominator of the *real* programme is per-programme, not per-draw.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.backtest.engine import run_backtest
from espa.backtest.synthetic import generate_synthetic_data
from espa.config import RunConfig
from espa.validation.registry import ConfigurationRegistry

#: Stage 1 features that the synthetic DGP loads (with their true signs)
#: and the ones that are pure noise. Recovery diagnostics compare against
#: this ground truth.
PLANTED_STAGE1_SIGNS = {"C_early": +1, "I_level": +1, "B_live": -1}
NOISE_STAGE1_FEATURES = ("C_late", "I_change", "X", "V")


@dataclass(frozen=True)
class World:
    """One synthetic data-generating process."""

    name: str
    edge: float
    options_edge: float = 0.0


def one_replication(
    world: World,
    rep: int,
    n_days: int = 1000,
    base_seed: int = 1000,
    min_threshold_history: int = 60,
) -> dict:
    """Run the full protocol on one seeded draw of one world."""
    seed = base_seed + rep
    cfg = RunConfig(name=f"{world.name}-rep{rep}")
    data = generate_synthetic_data(
        n_days=n_days, edge=world.edge, options_edge=world.options_edge,
        seed=seed, config=cfg,
    )
    registry = ConfigurationRegistry()
    res = run_backtest(
        data, cfg, registry, min_threshold_history=min_threshold_history
    )
    s = res.summary
    lam = res.daily["lambda"]
    coef1 = res.coef_by_fold.mean()
    coef2 = res.stage2_coef_by_fold.mean()

    row: dict = {
        "world": world.name,
        "rep": rep,
        "edge": world.edge,
        "options_edge": world.options_edge,
        "n_days_oos": s["n_days"],
        "n_trades": s["n_trades"],
        "trade_fraction": s["trade_fraction"],
        "zero_trades": s["n_trades"] == 0,
        "sharpe_full": s["sharpe_full"],
        "tstat_full": s["tstat_full"],
        "dsr_full": s["dsr_full"],
        "haircut_full": s["haircut_sharpe_full"],
        "sharpe_stage1": s["sharpe_stage1"],
        "tstat_stage1": s["tstat_stage1"],
        # the acceptance criterion of Section 22: deflated Sharpe
        # positive (SR beats the expected max of the trials) at t > 3
        "accept": bool(
            np.isfinite(s["tstat_full"])
            and s["tstat_full"] > 3.0
            and np.isfinite(s["dsr_full"])
            and s["dsr_full"] > 0.5
        ),
        "gate1_pass": bool(s["gate1_pass"]),
        "lambda_mean": float(lam.mean()),
        "lambda_final_qtr": float(
            np.nanmean(np.array_split(lam.to_numpy(), 4)[-1])
        ),
        "beta1_AxQ": float(coef2.get("A_x_Qhat", np.nan)),
        "beta2_PGI": float(coef2.get("PGI_perp", np.nan)),
        "beta3_P": float(coef2.get("P", np.nan)),
        "n_unstable": len(s["unstable_features"]),
    }
    for feat, sign in PLANTED_STAGE1_SIGNS.items():
        w = float(coef1.get(feat, np.nan))
        row[f"w_{feat}"] = w
        row[f"recovered_{feat}"] = bool(np.isfinite(w) and np.sign(w) == sign and w != 0)
    for feat in PLANTED_STAGE1_SIGNS | dict.fromkeys(NOISE_STAGE1_FEATURES):
        row[f"unstable_{feat}"] = feat in s["unstable_features"]
    return row


def _worker(args: tuple) -> dict:
    world, rep, n_days, base_seed = args
    return one_replication(world, rep, n_days=n_days, base_seed=base_seed)


def run_monte_carlo(
    worlds: list[World],
    n_reps: int | dict[str, int] = 100,
    n_days: int = 1000,
    base_seed: int = 1000,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """N seeded replications per world, in parallel. Returns one row per rep.

    Seeds are ``base_seed + world_index * 100_000 + rep`` so worlds do not
    share draws and the whole experiment replays exactly.
    """
    jobs = []
    for wi, world in enumerate(worlds):
        reps = n_reps[world.name] if isinstance(n_reps, dict) else n_reps
        for r in range(reps):
            jobs.append((world, r, n_days, base_seed + wi * 100_000))
    workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(_worker, jobs, chunksize=1):
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["world", "rep"]).reset_index(drop=True)


def summarise_monte_carlo(df: pd.DataFrame) -> pd.DataFrame:
    """Per-world error rates and recovery diagnostics.

    Sharpe statistics are reported over the replications that traded at
    all; the ``zero_trade_rate`` column carries the flat outcomes, so
    nothing is silently dropped.
    """
    out = []
    for world, g in df.groupby("world", sort=False):
        traded = g[~g["zero_trades"]]
        row = {
            "world": world,
            "n_reps": len(g),
            "edge": g["edge"].iloc[0],
            "options_edge": g["options_edge"].iloc[0],
            # error rates
            "accept_rate": g["accept"].mean(),
            "gate1_rate": g["gate1_pass"].mean(),
            "zero_trade_rate": g["zero_trades"].mean(),
            # trading behaviour
            "median_trades": g["n_trades"].median(),
            "mean_trade_fraction": g["trade_fraction"].mean(),
            # performance distribution (traded reps only)
            "sharpe_mean": traded["sharpe_full"].mean(),
            "sharpe_q10": traded["sharpe_full"].quantile(0.10),
            "sharpe_q90": traded["sharpe_full"].quantile(0.90),
            "tstat_mean": traded["tstat_full"].mean(),
            # blend behaviour
            "lambda_median": g["lambda_mean"].median(),
            "lambda_q90": g["lambda_mean"].quantile(0.90),
            # stage 2 coefficient recovery
            "beta1_mean": g["beta1_AxQ"].mean(),
            "beta2_mean": g["beta2_PGI"].mean(),
            "beta2_pos_rate": (g["beta2_PGI"] > 0).mean(),
            # stage 1 sign recovery and stability-filter behaviour
            **{
                f"recover_{f}": g[f"recovered_{f}"].mean()
                for f in PLANTED_STAGE1_SIGNS
            },
            "unstable_planted_rate": g[
                [f"unstable_{f}" for f in PLANTED_STAGE1_SIGNS]
            ].to_numpy().mean(),
            "unstable_noise_rate": g[
                [f"unstable_{f}" for f in NOISE_STAGE1_FEATURES]
            ].to_numpy().mean(),
        }
        out.append(row)
    return pd.DataFrame(out)
