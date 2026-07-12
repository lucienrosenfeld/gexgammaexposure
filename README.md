# ES Post-Auction Inventory-Resolution Strategy

Implementation of the master specification in [`docs/SPEC.md`](docs/SPEC.md):
a research framework for trading ES in the 16:00:15 → 16:14:30 (NY)
post-auction window, forecasting the volatility-scaled window return from an
observable core (pre-close pressure, closing-auction imbalance, live basis,
cross-asset confirmation, VWAP deviation) plus an incremental options block
(unsigned gamma density, momentum-residualised path-gamma integral,
concentration-weighted pinning) that can delete itself.

This repository implements the *framework* — features, models, validation
protocol, trade construction, risk controls — end to end. It does not ship
market data. Real research requires the Phase 0 data engineering described in
the spec (point-in-time imbalance history, strike-level options data with a
receipt-time audit, post-close ES intraday coverage); a synthetic generator
exists so the machinery itself is verifiable today.

## Layout: spec section → module

| Spec | Module |
|---|---|
| §2 exit rule (halt never hard-coded) | `espa/sessions.py` |
| §7 receipt-time discipline, five-timestamp schema | `espa/timestamps.py` |
| §7/§10 B_live estimator and basis feature | `espa/features/basis.py` |
| §8 robust strictly-lagged standardisation | `espa/standardize.py` |
| §9 σ*, mid target, executable selection targets | `espa/targets.py` |
| §10 Stage 1 core, imbalance specs I₁/I₂, frozen sign register | `espa/features/core.py`, `espa/features/cross_asset.py` |
| §11 A_t, kernels, window-liquidity denominator | `espa/features/options.py` |
| §11 PGI, ρ-blended 0DTE surface, fold-frozen residualisation | `espa/features/pgi.py` |
| §11 concentration-weighted pinning P_t | `espa/features/pinning.py` |
| §12 sign-constrained ridge (NNLS + L2), Stage 1 | `espa/models/constrained_ridge.py`, `espa/models/stage1.py` |
| §13 Stage 2 + mandatory stacking discipline | `espa/models/stage2.py` |
| §14 loss-based blend λ_t with utility gate | `espa/models/blend.py` |
| §16 uncertainty-scaled trigger, penalised threshold objective | `espa/models/uncertainty.py`, `espa/trade/threshold.py` |
| §17 position mapping, estimated h(A_t) conditioner | `espa/trade/sizing.py` |
| §19 execution protocol, catastrophe rule, operational exits | `espa/trade/execution.py` |
| §20 D_ordinary / D_event split | `espa/sessions.py` |
| §21–22 purged/embargoed walk-forward, deflated Sharpe, Harvey-Liu haircut, bootstrap CI, sign stability | `espa/validation/` |
| §22 append-only configuration registry (honest trial count) | `espa/validation/registry.py` |
| §23 kill switches | `espa/risk/killswitch.py` |
| Appendix B constants | `espa/config.py` |
| orchestration + event-day logging | `espa/backtest/engine.py` |
| synthetic DGP for framework verification | `espa/backtest/synthetic.py` |
| Monte Carlo error-rate calibration (Type I/II, λ, recovery) | `espa/backtest/montecarlo.py` |

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                  # framework verification suite
python scripts/run_synthetic_backtest.py --days 750 --edge 0.5
python scripts/run_monte_carlo.py --reps 200   # Type I/II calibration (~15 min on 4 cores)
```

The demo plants a known edge in synthetic data and runs the full protocol:
expanding walk-forward with quarterly refits and a 5-day embargo, per-window
stacking (Stage 2 fitted only against out-of-fold Stage 1 predictions),
sequential blending with the utility gate, threshold selection on strictly
prior out-of-sample history, executable-P&L accounting, and the acceptance
metrics with the registry's trial count as the multiple-testing denominator.
Event days are scored by the frozen model and logged separately, never
influencing selection.

## Running on real data

Build a `StrategyData` (see `espa/backtest/engine.py`) from your feeds:

1. **Features** — use `espa.features.*` builders. All standardisation is
   strictly lagged by construction; imbalance messages must pass through
   `espa.timestamps.filter_imbalance_messages` (primary features use only
   messages received by 15:59:55), and B_live must come from
   `espa.features.basis.estimate_spx_close` at decision time — the eventual
   official close is diagnostic-only and has no entry point into features.
2. **Targets** — `espa.targets.targets_from_prices` from roll-adjusted ES
   mids and top-of-book quotes at entry/exit, with your fee + slippage model
   as `c_t`.
3. **Universes** — `espa.sessions.UniverseCalendar` with externally sourced
   FOMC and rebalance dates (the framework computes expiries, month-ends and
   half-days; it cannot compute FOMC dates and does not pretend to).
4. **Run** — `espa.backtest.run_backtest(data, config, registry)`. Every
   configuration you try must be a distinct `RunConfig` logged in a
   persistent `ConfigurationRegistry`; the deflated-Sharpe and haircut
   denominators are computed from that registry, and the registry file is
   append-only.

Decision gate 1 (spec §12) is reported in the result summary as
`gate1_pass`: if Stage 1 alone shows no deflated-Sharpe-significant edge
after costs, the programme stops.

## What is deliberately absent

Per the spec's register of rejected designs (Appendix C): no dealer
positioning signs, no gamma-flip regime switches, no charm/vanna, no
hand-set weights, no full-session ADV denominator, no raw PGI coefficient,
no event-day dummies, no hard-coded 16:15 halt, no frequency-targeted
threshold, no R²-ratio blending, no consecutive-count slippage rule, and no
ordinary price stop. These are recorded decisions, not omissions.

## Status

Framework implementation (Phases 1–2 machinery, Phase 0 interfaces). Broker
and feed adapters, the overnight-horizon exit study, and the boosted-model
comparison are counted future work. Nothing here is investment advice, and
no synthetic-data number is a research result.
