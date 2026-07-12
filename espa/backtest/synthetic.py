"""Synthetic data generator.

Exists to verify the *framework* — leakage discipline, stacking,
blending, threshold warmup, acceptance metrics — on data whose true
data-generating process is known. It plants a controllable edge in the
Stage 1 observables and (optionally) in the options block, so tests can
check both that the pipeline finds a planted edge and that it reports a
null when the edge is absent. It proves nothing about markets, and no
number produced from it is a research result.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from espa.backtest.engine import StrategyData
from espa.config import DEFAULT_CONSTANTS, RunConfig, SpecConstants
from espa.features.core import Stage1Inputs, build_stage1_features
from espa.features.options import OptionContract, gamma_density, window_liquidity
from espa.features.pgi import path_gamma_integral
from espa.features.pinning import pinning_feature
from espa.targets import ex_ante_window_vol, targets_from_prices


def _bs_gamma(spot: float, strike: float, vol: float, t_years: float) -> float:
    t = max(t_years, 1e-4)
    d1 = (np.log(spot / strike) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
    return float(np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * vol * np.sqrt(t)))


def generate_synthetic_data(
    n_days: int = 900,
    edge: float = 0.25,
    options_edge: float = 0.0,
    seed: int = 11,
    config: RunConfig | None = None,
    constants: SpecConstants = DEFAULT_CONSTANTS,
    start: dt.date = dt.date(2022, 6, 1),
) -> StrategyData:
    """Build a complete :class:`StrategyData` with a planted edge.

    ``edge`` scales the dependence of the window return on the Stage 1
    signal; ``options_edge`` plants incremental structure on PGI_perp so
    the Stage 2 machinery has something real to find when asked.
    """
    rng = np.random.default_rng(seed)
    config = config or RunConfig(name="synthetic")
    days = pd.bdate_range(start=start, periods=n_days).date
    idx = pd.Index(days, name="day")

    # --- raw pre-close observables ---
    c_early = pd.Series(rng.normal(0, 0.0018, n_days), index=idx)
    c_late = pd.Series(rng.normal(0, 0.0009, n_days), index=idx)
    i_50 = pd.Series(rng.normal(0, 1.0, n_days), index=idx)
    i_55 = i_50 + rng.normal(0, 0.4, n_days)
    r_day = pd.Series(rng.normal(0, 0.008, n_days), index=idx)
    vwap_dev = pd.Series(rng.normal(0, 0.6, n_days), index=idx)
    b_live = pd.Series(rng.normal(0, 1.0, n_days), index=idx)
    x_comp = pd.Series(rng.normal(0, 1.0, n_days), index=idx)

    inputs = Stage1Inputs(
        c_early=c_early,
        c_late=c_late,
        i_55=i_55,
        i_50=i_50,
        x_composite=x_comp,
        b_live=b_live,
        vwap_deviation=vwap_dev,
        i_last=i_55 + rng.normal(0, 0.2, n_days),
        r_full_day=r_day,
    )
    features, constraints = build_stage1_features(inputs, config, constants)

    # --- ES level path and options surface ---
    spot = pd.Series(4000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))), index=idx)
    daily_contracts: dict = {}
    pgi_raw = pd.Series(np.nan, index=idx)
    pinning = pd.Series(np.nan, index=idx)
    atr = spot * 0.012
    for k, day in enumerate(idx):
        s = float(spot.iloc[k])
        strikes = np.round(s / 25.0) * 25.0 + 25.0 * np.arange(-4, 5)
        contracts = [
            OptionContract(
                strike=float(kk),
                expiry_years=1.0 / 365.0,
                gamma=_bs_gamma(s, float(kk), 0.15, 1.0 / 365.0),
                open_interest=float(rng.integers(500, 20000)),
                multiplier=100.0,
                same_day_volume=float(rng.integers(0, 5000)),
            )
            for kk in strikes
        ]
        daily_contracts[day] = contracts
        # intraday 5-minute path, 9:30 -> 16:00 (78 bars)
        path = s * np.exp(np.cumsum(np.concatenate([[0], rng.normal(0, 0.0009, 77)])))
        pgi_raw.loc[day] = path_gamma_integral(
            pd.Series(path), contracts, rho=config.rho, lambda_s=config.lambda_s,
            lambda_t=config.lambda_t,
        )
        pinning.loc[day] = pinning_feature(
            contracts, s, float(atr.iloc[k]), config.lambda_s, config.lambda_t
        )

    window_dollar_vol = pd.Series(rng.lognormal(18, 0.3, n_days), index=idx)
    liquidity = window_liquidity(window_dollar_vol, constants)
    a_raw = gamma_density(daily_contracts, spot, liquidity, config.lambda_s, config.lambda_t)
    from espa.standardize import robust_z

    a_density = robust_z(a_raw, constants)

    # --- window return with the planted edge ---
    signal = (
        features["C_early"].fillna(0)
        + features["I_level"].fillna(0)
        - features["B_live"].fillna(0)
    ) / 3.0
    pgi_z = robust_z(pgi_raw, constants).fillna(0)
    base_vol = 3.0  # ES points over the 14-minute window
    r_window_points = base_vol * (
        edge * signal + options_edge * pgi_z + rng.normal(0, 1.0, n_days)
    )

    sigma_star = ex_ante_window_vol(pd.Series(r_window_points, index=idx), constants)

    # --- executable prices around the mid ---
    tick = 0.25
    mid_entry = spot
    mid_exit = spot + r_window_points
    prices = pd.DataFrame(
        {
            "mid_entry": mid_entry,
            "mid_exit": mid_exit,
            "bid_entry": mid_entry - tick / 2,
            "ask_entry": mid_entry + tick / 2,
            "bid_exit": mid_exit - tick / 2,
            "ask_exit": mid_exit + tick / 2,
        },
        index=idx,
    )
    costs = pd.Series(0.05, index=idx)  # fees + modelled slippage, points
    targets = targets_from_prices(prices, sigma_star, costs)

    # --- universes ---
    month_end = pd.Series(
        [d.month != nd.month for d, nd in zip(days[:-1], days[1:])] + [True], index=idx
    )
    random_event = pd.Series(rng.random(n_days) < 0.06, index=idx)
    ordinary = ~(month_end | random_event)
    options_era = pd.Series(True, index=idx)

    return StrategyData(
        stage1_features=features,
        constraints=constraints,
        y_mid=targets["y_mid"],
        targets=targets,
        a_density=a_density,
        pgi_raw=pgi_raw,
        c_early=c_early,
        c_late=c_late,
        r_day=r_day,
        pinning=pinning,
        ordinary=ordinary,
        options_era=options_era,
    )
