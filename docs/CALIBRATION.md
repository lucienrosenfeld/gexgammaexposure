# Monte Carlo Calibration of the Validation Framework

Measured Type I / Type II properties of the full protocol on synthetic
worlds with known data-generating processes. 200 seeded replications per
world, 1,000 synthetic days each (~640–660 scored out-of-sample per
replication), every replication running the identical full protocol —
walk-forward, stacking, blending, sequential threshold selection with
flat admissible, acceptance metrics at the registry's 17-trial count.

Reproduce with `python scripts/run_monte_carlo.py --reps 200`
(deterministic; ~15 min on 4 cores). Extended-sample power runs used
`espa.backtest.montecarlo` directly at 1,500 and 2,000 days, 100
replications per cell.

**Scope.** These are properties of the *framework* on the *synthetic*
DGP (linear signal, Gaussian noise, one planted options channel). They
transfer to real data only to the extent the real world resembles that —
which is exactly why the acceptance criteria stay as conservative as
they are. No number here is evidence about markets.

## Worlds

| world | Stage-1 edge | options edge | interpretation |
|---|---|---|---|
| null | 0.00 | 0.00 | pure noise plus costs |
| stage1-weak | 0.25 | 0.00 | small observable edge |
| stage1-med | 0.50 | 0.00 | moderate observable edge (per-trade SR ≈ 0.27) |
| options-small | 0.50 | 0.15 | realistic-sized incremental options edge |
| options-large | 0.50 | 0.40 | generous options edge (upper bound / sanity) |

## Error rates and trading behaviour

| world | accept rate (t>3 + DSR) | zero-trade rate | median trades | mean trade fraction |
|---|---|---|---|---|
| null | **0.000** | 0.305 | 14.5 | 3.8% |
| stage1-weak | 0.015 | 0.060 | 67 | 11.1% |
| stage1-med | 0.315 | 0.000 | 83 | 13.9% |
| options-small | 0.425 | 0.000 | 82.5 | 13.8% |
| options-large | 0.920 | 0.000 | 71.5 | 12.1% |

- **False-positive rate of the acceptance criteria: 0/200** (95% upper
  bound ≈ 1.5% by the rule of three).
- The flat-is-admissible patch works: the null world now trades a mean
  3.8% of days (30% of replications trade zero days), versus 21.7% —
  more than the edge worlds — before the patch.
- Residual null trading still costs (mean per-trade Sharpe −0.12 among
  traded replications): threshold selection on a finite trailing window
  can be fooled locally. The deployment gate, not the θ optimiser, is
  the backstop, and it held in all 200 draws.

## Performance and blend behaviour (traded replications)

| world | Sharpe mean (q10, q90) | mean t-stat | λ median | λ q90 |
|---|---|---|---|---|
| null | −0.12 (−0.34, 0.10) | −0.63 | 0.061 | 0.142 |
| stage1-weak | 0.04 (−0.13, 0.22) | 0.41 | 0.062 | 0.142 |
| stage1-med | 0.27 (0.13, 0.41) | 2.48 | 0.059 | 0.136 |
| options-small | 0.32 (0.14, 0.49) | 2.76 | 0.264 | 0.391 |
| options-large | 0.58 (0.37, 0.82) | 4.87 | 0.730 | 0.862 |

- **λ noise floor ≈ 0.06 (median), 0.14 (q90)** — measured on the three
  worlds with no options edge. λ separates cleanly from the floor at
  options-small (median 0.26) and decisively at options-large (0.73).
  A live λ persistently above ~0.15 is evidence the options block is
  contributing; below that it is indistinguishable from noise.

## Coefficient recovery

| world | β₂ mean | β₂ sign-recovery | Stage-1 sign recovery (C_early / I_level / B_live) | stability filter fires: planted / noise |
|---|---|---|---|---|
| null | −0.001 | 0.515 | — (nothing planted) | 0.000 / 0.560 |
| stage1-weak | 0.002 | 0.535 | 1.00 / 1.00 / 1.00 | 0.000 / 0.549 |
| stage1-med | 0.008 | 0.540 | 1.00 / 1.00 / 1.00 | 0.000 / 0.589 |
| options-small | **0.214** | **1.000** | 1.00 / 1.00 / 1.00 | 0.000 / 0.551 |
| options-large | 0.503 | 1.000 | 1.00 / 1.00 / 1.00 | 0.000 / 0.548 |

- β₁ (the A×Q̂ interaction) is ≈ 0 in every world, as it should be —
  nothing was planted on it, and it does not free-ride on the planted
  channels.
- **β₂ sign recovery is 100% at options-small even though acceptance
  passes only 42.5% of the time.** The coefficient machinery detects an
  options edge long before the acceptance test can certify a tradable
  strategy — consistent with the spec's position that β₂ is the
  genuinely interesting number and accumulates evidence faster than
  P&L-level significance.
- The sign-stability filter never fired on a planted feature in 1,000
  replications, and fired on ~55% of pure-noise features per
  replication: it removes noise without collateral damage at these
  effect sizes.

## Power as a function of sample size

At the effect sizes actually believed in (per-trade Sharpe ≈ 0.2–0.3),
acceptance at t > 3 with ~650 OOS days has power ≈ 0.3–0.4. Extended
samples (100 replications per cell):

| world | days (≈OOS) | accept rate | mean t-stat |
|---|---|---|---|
| stage1-med | 1,000 (~650) | 0.315 | 2.48 |
| stage1-med | 1,500 (~1,100) | *see mc_power run* | |
| stage1-med | 2,000 (~1,600) | *see mc_power run* | |
| options-small | 1,000 (~650) | 0.425 | 2.76 |
| options-small | 1,500 (~1,100) | *see mc_power run* | |
| options-small | 2,000 (~1,600) | *see mc_power run* | |

## Bottom line

The framework's Type I error is measured at ~0 and its coefficient
diagnostics are far more sensitive than its acceptance test, which is
the intended asymmetry: research information accrues (β₂, λ, sign
stability) years before deployment certification does. The cost of the
t > 3 hurdle is real and now quantified: at believable effect sizes,
roughly two-thirds of true edges will not certify on ~650 OOS days.
That is the price of a near-zero false-positive rate under 17 registered
trials, and the remedy is accumulation, never looser criteria.
