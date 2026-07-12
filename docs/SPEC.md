# ES Post-Auction Inventory-Resolution Strategy

## Master Specification and Research Programme — Consolidated Document

This document consolidates the full strategy as it stands after three design rounds and a literature calibration pass. It is written to be sufficient on its own: a researcher with market data access and no knowledge of the prior drafts should be able to implement, validate, and operate the strategy from this document alone. Where a design choice was contested during development, the resolution and its reasoning are recorded, because the reasoning is part of the specification. Choices made for a reason should not be silently unwound later when the reason has been forgotten.

---

# Part I. Economic Foundations

## 1. The trade in one paragraph

The strategy trades the E-mini S&P 500 future (ES) in the brief post-auction window beginning at 16:00:15 New York time and ending 30 seconds before the equity-index futures trading halt, in practice 16:14:30. It forecasts the volatility-scaled ES return over that window using two blocks of information: an observable core built from pre-close price pressure, closing-auction imbalance, post-auction basis state, cross-asset confirmation, and VWAP deviation, and an incremental options block built from unsigned gamma density, a momentum-residualised path-gamma integral, and concentration-weighted strike pinning. Positions are taken only when the forecast clears an uncertainty-scaled threshold, sized by volatility targeting, and closed unconditionally before the halt. The options block can be deleted by the model itself, both in estimation through regularisation and in production through a loss-based blending weight, and the whole strategy is subject to a validation protocol designed for the defining constraint of the problem: one observation per trading day in a market regime that has existed only since mid-2022.

## 2. Market structure of the window

Three structural facts define the window and were each verified rather than assumed.

First, the CME equity-index daily fixing derives from the 30-second volume-weighted average price of Globex trades between 15:59:30 and 16:00:00 ET. There is no later settlement price toward which the 16:00 to 16:15 interval converges. Any framing of this trade as convergence between the cash close and a 16:15 futures settlement is wrong, and an earlier draft of this strategy made exactly that error.

Second, the NYSE closing auction resolves at 16:00, with D-Orders accepted until 15:59:50 and offsetting orders until 16:00. The auction state can change materially in the final seconds, which has direct consequences for what imbalance information is legitimately usable at a 16:00:15 decision time. Closing auctions have grown to a large share of daily volume, and the academic record documents systematic closing-price distortion from auction and market-on-close flows followed by reversal.

Third, CME equity-index futures halt daily from 16:15 to 16:30 ET, per CME's own product FAQ and multiple broker specifications, distinct from the 17:00 to 18:00 ET Globex maintenance period. The exit deadline therefore sits 30 seconds ahead of a hard trading halt. This halt is nonetheless never hard-coded. The engine reads the current-day session schedule from the exchange reference-data feed and computes:

t^exit_t = min( t^{next halt}_t − 30s, 16:14:30 )

If a session has no 16:15 halt, 16:14:30 remains the research-window endpoint and the trapped-through-halt failure mode is absent. Holiday and shortened sessions resolve automatically through the same feed.

## 3. The hypothesis

The 16:00:15 to 16:14:30 ES return reflects the resolution or continuation of inventory created by the cash closing auction, expiring-option hedges, basis trades, and participants who could not or would not complete their adjustment before the 16:00 fixing.

Several distinct mechanisms plausibly contribute, and the design treats them as separate testable channels rather than one aggregate story.

**Auction dislocation and reversion.** Concentrated closing flows push the cash close away from equilibrium, and the deviation reverts. The single-stock literature estimates that roughly 85 percent of the closing deviation reverses by the next morning. Two caveats govern how this number is used. It is measured overnight rather than within the first quarter hour, and it is measured on individual stocks, where auction distortions do not net. At the index level most constituent-level dislocation cancels except on rebalance days, so the ES-level magnitude is unknown and presumably far smaller. The number is a mechanism citation and never a calibration input.

**Post-fixing hedge adjustment.** Participants who hedge against the 16:00 fixing, including option market makers whose expiring exposure references the close, may carry residual futures inventory past 16:00 whose adjustment creates directional pressure. The 0DTE literature finds that net-gamma effects on the underlying dissipate within about an hour, which favours a short window for this channel.

**Basis correction.** ES rich or cheap to fair value against the official cash close tends to correct, though after 16:00 the cash index is no longer continuously tradable, so the basis is a state variable describing post-auction positioning pressure rather than a live arbitrage.

**Strike pinning and release.** Delta hedging of concentrated near-money open interest pulls or pushes price around dominant strikes, an effect documented in S&P futures on expiration days, with sign depending on positioning that is not publicly observable.

The two dominant channels may live on different clocks. Hedge adjustment argues for the 14-minute window, auction reversion for an overnight horizon. The research programme tests both horizons explicitly (Part V) so the data can say which mechanism carries whatever edge exists.

## 4. The central design principle: latent positioning does not steer the strategy

The original design this strategy grew out of made dealer gamma positioning the structural switch of the whole model: the same closing-flow signal was faded under estimated positive dealer gamma and followed under estimated negative dealer gamma, through per-strike positioning signs, tanh regime functions, a gamma-flip level, and charm and vanna exposures, all of which inherit their sign from a quantity nobody can observe.

That architecture is rejected, and the rejection is now empirically grounded rather than merely cautious. Cboe's own trade-level SPX data puts median net dealer gamma at 3:30 p.m. near +$173 million, roughly 0.04 to 0.17 percent of daily S&P futures liquidity. A regime model that flips trading direction on the sign of a quantity whose median is indistinguishable from zero at that scale is fitting noise. What the mechanism literature does establish is that gamma-conditioned momentum and reversal exist when imbalances are large and the underlying is illiquid. The design therefore extracts the testable content of the gamma story through three observables that require no positioning assumption to construct, and lets fitted, shrunk, sign-unconstrained coefficients answer the questions the original design answered by assumption. Charm and vanna are excluded entirely: both require the same unobservable positioning sign, both are second order in this window, and each adds hyperparameters that a sample of several hundred observations cannot fund. They are the first candidates for readmission if the core proves out and the sample grows.

## 5. The sample-size constraint

One trade per day, in a market whose defining feature (daily SPX expirations, hence 0DTE at scale) completed in May 2022, yields an effective stationary sample of roughly 700 to 1,000 observations at the time of writing. That supports a model with single-digit effective degrees of freedom under heavy regularisation. Every design decision in this document, from the fixed cross-asset composite to the discrete ρ grid to the refusal of event-day dummies, follows from this constraint. Where a richer specification was proposed during development, the default answer was to record it as a counted alternative rather than to enlarge the primary model.

---

# Part II. Data Architecture

## 6. Required data, in order of criticality

**Point-in-time NYSE closing-auction imbalance feed.** The binding constraint of the entire programme and the most expensive item. The edge, if it exists, most plausibly lives here. Vendor histories must be as-received, never as-corrected-later. If a clean point-in-time history cannot be sourced, this feature block is built forward through live capture before it is trusted, and the research timeline extends accordingly.

**Strike-level options data with daily history for SPX/SPXW and ES options.** Strike-level Greeks (CME strike-level analytics or OPRA-derived vendor data), open interest, same-day volume by strike, and implied volatilities. Trade-level open/close classification data upgrades the same-day gamma proxy if it can be sourced.

**Intraday ES data covering the post-close window.** Five-minute or finer bars including 16:00 to 16:15, with top-of-book quotes for executable-price targets. Many equity-index datasets truncate at the cash close, so coverage of this specific window is verified before any purchase.

**SPX constituent auction prints and index values with receipt timestamps**, for the live basis estimate.

**Rates and dividends.** Term SOFR and index dividend futures for fair value.

**Cross-asset intraday bars.** NQ, RTY, YM, 10-year note futures, DXY.

**Exchange reference data.** Session schedules and security definitions, consumed daily by the engine.

## 7. Timestamp and receipt-time discipline

Three inputs are time-critical, and each is subject to the same rule: the maximum information timestamp of any feature must precede the decision time.

t^{feature}_max < t^{decision}

**Imbalance.** Every message stores five timestamps: t^{exchange}, t^{vendor}, t^{receive}, t^{decision}, t^{order}. Primary features use only messages demonstrably received by 15:59:55. Whether the post-16:00 auction result message adds information at realistic latency is a separate, counted test, never an assumption.

**SPX official close.** The consolidated official close may not be received and processed by 16:00:15, since constituent auction prints arrive with varying latency. Two basis series are maintained. B^{live} is built at decision time from received official constituent prints, last tradable prices for unresolved constituents, and known index weights. B^{final} uses the eventual official close and exists for ex-post diagnostics only. Only B^{live} may enter the trading model.

**Options surface.** Every surface input stores t^{quote}, t^{trade}, t^{vendor calculation}, t^{receive}. Vendor end-of-day surfaces reconstructed after the fact, through late trade reports, corrections, or smoothing passes, are look-ahead and are excluded from live features. This applies to same-day volume, recomputed Greeks, implied vols, and open interest. An audit of the chosen vendor's reconstruction policy is a mandatory pre-purchase step.

## 8. Standardisation

All features are time-series standardised against their own strictly lagged history using robust statistics:

z_{j,t} = ( x_{j,t} − median_{t−251:t−1}(x_j) ) / ( 1.4826 · MAD_{t−251:t−1}(x_j) )

The window for day t ends at t−1 without exception. Median and MAD replace mean and standard deviation because closing variables are heavy-tailed and a few extreme observations should not move the estimated centre and scale. The rule applies to imbalance, basis, gamma density, short-horizon returns, and the slippage series.

---

# Part III. Targets and Features

## 9. Targets

**Volatility scale, strictly ex ante with a floor.**

σ_t = (1−α) Σ_{j=1}^{20} α^{j−1} |r^{window}_{t−j}|, using observations through t−1 only

σ*_t = max( σ_t, Q_{10%}^{rolling}(σ) )

The explicit lag structure exists so that no implementation accidentally updates the estimate with the target day's own return. The floor prevents a run of abnormally quiet windows from producing excessive leverage.

**Modelling target.** The forecast model is trained on the mid-price return:

y^{mid}_t = (Mid_{exit} − Mid_{entry}) / σ*_t

with P the roll-adjusted front-contract ES mid, rolls handled by volume-weighted adjustment with roll weeks flagged.

**Selection targets.** All decisions that select among configurations use direction-conditioned executable P&L:

y^{long}_t = (Bid_{exit} − Ask_{entry} − c_t) / σ*_t

y^{short}_t = (Bid_{entry} − Ask_{exit} − c_t) / σ*_t

with c_t covering fees plus modelled slippage. Dead-zone calibration, Sharpe estimation, configuration selection, and significance testing all run on the executable series. A model that predicts mid direction but cannot overcome the spread is rejected by construction, which matters because the post-close book is thin and the spread is a large fraction of the typical window move.

## 10. Stage 1 features: the observable core

x^Q_t = [ C^{early}, C^{late}, I_{55}, ΔI, B^{live}, X, V ]

**C^{early} = r^{ES}_{15:30–15:55} and C^{late} = r^{ES}_{15:55–16:00}.** Two non-overlapping pre-close horizons. They are separated because the final five minutes carry disproportionate MOC-related information, and they are non-overlapping so that coefficient interpretation is clean. The early horizon carries a continuation prior. The late horizon carries no prior, because the final five-minute move can represent durable information, temporary auction impact, or anticipatory positioning that unwinds after the print, and no one of those stories is strong enough to constrain a coefficient.

**I_{55} and ΔI = I_{55} − I_{50}.** SPX-weight-aggregated closing-auction imbalance at the 15:55 snapshot, and its change from 15:50, each normalised by 20-day median closing-auction volume. The level carries a directional prior. The change does not, because a rapidly worsening imbalance can induce pre-positioning before 16:00 that makes the post-close residual mean-reverting, and because its meaning is conditional on what price did while it changed. An imbalance being absorbed (ΔI > 0 with C^{late} < 0) is a different state from one being transmitted (ΔI > 0 with C^{late} > 0). The explicit absorption interaction −z(C^{late})·z(ΔI) is a recorded alternative that must displace, never augment, an existing feature.

The imbalance block has two alternative specifications, never combined, because [I_{55}, ΔI, I_{last}] together are severely collinear:

Spec I₁ (primary): [I_{55}, ΔI]

Spec I₂: [I_{last}, I_{last} − I_{55}], where I_{last} is the last message actually received before decision time, receipt-stamped

Spec I₂ answers whether the final seconds add information beyond the stable 15:55 state at realistic latency. Note that I_{50} appears in no specification alongside the others, since it is mechanically recoverable as I_{55} − ΔI.

**B^{live}.** Post-auction basis state:

B^{live}_t = [ ES_{16:00:15} − ŜPX^{close}_{16:00:15} · e^{(r−q)τ} ] / σ^{basis}_{252}

with r from term SOFR to expiry, q from the index dividend futures strip, and the estimated close constructed as in Section 7. Rich futures carry a reversion prior.

**X.** Fixed cross-asset composite over the 15:30 to 16:00 horizon:

s^{TY}_t = tanh( Corr_{60}(r^{ES}, r^{TY}) / c ), c = 0.20, fixed and prespecified

X_t = [ z(r^{NQ}) + z(r^{RTY}) + z(r^{YM}) − z(r^{DXY}) + s^{TY}_t · z(r^{TY}) ] / (4 + |s^{TY}_t|)

The Treasury weight is a bounded continuous transformation of a noisy lagged correlation, carrying no fitted parameters. It exists because the stock-bond relation changes sign across inflation and growth regimes, and a hard sign function would make binary jumps on correlation estimates near zero. A TY-free composite and an expanding-window PCA variant (orientation fixed by requiring positive correlation with the pre-close ES return) are counted alternatives. The fixed composite is primary because rolling PCA loadings on five series are unstable and sign-indeterminate across refits.

**V.** ES price at 16:00 relative to full-day VWAP, in ATR units. Captures whether the close occurred stretched or compressed against the day's accepted value. Unconstrained.

**Counted Stage 1 alternative from the literature.** The full-day 9:30 to 15:30 return, motivated by the finding that rest-of-day return predicts late-day return through hedging demand. It currently appears only inside the PGI residualisation. As a Stage 1 feature it must displace an existing feature under the parameter budget, and the caveat is recorded that the documented effect is pre-close, so its post-fixing persistence is precisely what the test would establish.

**Sign constraints, prespecified before any results:**

w_{C^{early}} ≥ 0, w_{I} ≥ 0, w_{B} ≤ 0

C^{late}, ΔI, X, and V are unconstrained. Sign constraints are the principal small-sample defence: a coefficient the data wants to flip against a strong economic prior is shrunk to zero instead of admitted with the wrong sign, at a small cost in fit and a large gain in out-of-sample survival. Constraints are imposed only where the prior is strong, and the register of which features are constrained was frozen before estimation.

## 11. Stage 2 features: the options block

**A_t, unsigned gamma density normalised by window liquidity.** For each relevant SPX/SPXW and ES-option contract i:

G̃_{i,t} = Γ_{i,t} · OI_{i,t} · Q_i · S_t² · 0.01 · exp(−|K_i − S_t|/(λ_S S_t)) · exp(−T_i/λ_T)

with λ_S and λ_T tuned in validation from initial values 0.0075 and 3/365 and counted as hyperparameters. Note the deliberate absence of any positioning sign η_i. Then:

A_t = z( Σ_i |G̃_{i,t}| / Liquidity_t )

Liquidity_t = Median_{20}[ DollarVolume^{ES}_{16:00–16:15} ]

upgraded to a top-of-book depth median for the same window if historical depth is sourced. Full-session ADV is rejected as the denominator because it is not the liquidity that would absorb a post-close unwind, and the gamma-fragility literature finds the price impact of hedging is conditional on underlying illiquidity, which is precisely what this ratio measures. A_t answers the question public data can answer: how much convexity sits near the money relative to the liquidity available in the window, regardless of who holds it. It carries no standalone directional coefficient in the primary model, because unsigned density has no unconditional direction and E[y | A] should be approximately zero. Its admitted roles are the interaction with the Stage 1 forecast, the volatility conditioner, and sizing. A standalone directional term is admitted only if it survives repeated out-of-sample testing, and that test is counted.

**PGI^⊥, the residualised path-gamma integral.** The raw statistic accumulates the 0DTE gamma surface against the day's realised path over five-minute bars from 9:30 to 16:00, re-marking the surface as spot moves:

PGI_t = −Σ_s G̃^{0DTE}_s · ΔS_s / S_s

Under a global prior that dealers are net short near-money 0DTE options, this approximates the futures inventory accumulated hedging that book during the day, and the hypothesis is that some of it releases after the options stop trading at 16:00. Three disciplines keep this honest.

First, the naming and the sign. This is a path-weighted directional gamma statistic and equals dealer inventory only if the global short-gamma prior is correct and stable, which the trade-level evidence says it often is not. Settlement extinguishes the option, never the hedge: market makers net across expiries, hedge at portfolio level, trade down exposure before the close, or execute against the fixing. The coefficient is therefore sign-unconstrained and shrunk like everything else.

Second, the momentum residualisation. The raw statistic is mechanically correlated with ordinary intraday directional movement, so a significant raw coefficient would demonstrate nothing about options. Inside every training fold, on training data only:

PGI_t = a + b₁C^{early}_t + b₂C^{late}_t + b₃r^{ES}_{09:30–15:30,t} + ε_t, and PGI^⊥_t = ε̂_t

The residualisation coefficients are frozen per fold and applied forward. The research question is thereby sharpened to whether the gamma-weighted path contains information beyond the path itself. A β₂ that survives this is the genuinely interesting number in the entire strategy.

Third, the stale-OI problem. Previous-evening open interest misses expiration-day flow, and same-day volume conflates opening and closing trades in both directions. The surface blends both:

G̃^{0DTE}_s = G̃^{prior OI}_s + ρ · G̃^{same-day volume}_s, ρ ∈ {0, 0.25, 0.5}

with the three discrete variants counted as configurations rather than ρ tuned continuously. Trade-level open/close classification replaces the volume proxy if sourced.

**P_t, concentration-weighted pinning.** With K* the strike maximising near-money unsigned gamma weight, D_t = (S_t − K*)/ATR, and the strike-concentration Herfindahl HHI^Γ_t = Σ_K (W_K/Σ_j W_j)²:

P_t = −HHI^Γ_t · D_t · e^{−|D_t|}

The exponential envelope bounds the feature so a distant strike cannot generate a large signal, and the concentration weight silences the putative magnet when gamma is diffuse across strikes, at the cost of no additional fitted coefficient. The pinning direction depends on positioning, so this coefficient is also unconstrained.

---

# Part IV. Model Architecture

## 12. Stage 1

Q̂_t = f_Q(x^Q_t)

Ridge regression on the seven Stage 1 features with the sign constraints of Section 10, implemented as non-negative least squares on sign-flipped constrained features with an L2 penalty, the penalty chosen by purged, embargoed walk-forward validation on executable P&L. Refit quarterly on an expanding window.

**Decision gate 1.** If Stage 1 shows no deflated-Sharpe-significant edge after costs out-of-sample, the programme stops. No options superstructure rescues a strategy whose observable core is noise. If it passes, Stage 1 is the production fallback that everything else must beat.

## 13. Stage 2

Ô_t = β₁ · A_t · Q̂^{OOF}_t + β₂ · PGI^⊥_t + β₃ · P_t

Three coefficients, ridge-shrunk, all sign-unconstrained, fitted on Stage 1 residual structure. Stage 2 never refits Stage 1 variables and never includes them as regressors alongside Q̂, because that would create near-exact redundancy and destroy the incrementality of the test.

**Stacking discipline, mandatory.** Within each training window: divide into internal chronological folds, generate out-of-fold predictions Q̂^{OOF}, fit Stage 2 against those, then refit Stage 1 on the complete training window to produce validation-period Q̂, and apply the Stage 2 coefficients without refitting. In-sample Stage 1 fitted values never touch Stage 2 estimation. Without this, the interaction term inherits Stage 1's in-sample optimism and the options block will look better than it is, which is the exact direction of error the architecture exists to prevent.

The interaction β₁A_tQ̂_t is the honest version of the original gamma regime switch. If dealers are systematically long the near-money complex, β₁ comes out negative (density dampens follow-through). If short, positive (density amplifies). If positioning is balanced or unstable, as the trade-level evidence suggests, β₁ shrinks to zero and nothing breaks.

## 14. Final forecast and live blending

ŷ_t = Q̂_t + λ_t · Ô_t

The blend weight compares exponentially weighted forecast losses rather than an R² ratio, because out-of-sample R² is frequently negative and a ratio of two near-zero quantities is unstable:

L^Q_t = EWMA_{60}[(y − Q̂)²], L^{full}_t = EWMA_{60}[(y − ŷ^{full})²]

λ_t = clip( (L^Q_t − L^{full}_t)/(δ·L^Q_t), 0, 1 ), δ = 0.05 prespecified

A utility gate sits on top: if the options block's incremental contribution to net P&L after costs over the trailing 60 trades is negative, λ_t is forced to zero regardless of the MSE comparison, because a model can reduce squared error inside the dead zone without improving anything tradable. This mechanism operationalises the requirement that the options model down-weight itself when its predictive accuracy deteriorates, as a formula rather than an intention.

## 15. Model complexity ceiling

The primary model is the constrained ridge described above and nothing else. If, and only if, it shows out-of-sample edge and residual diagnostics suggest interactions the linear form misses, a depth-2 gradient-boosted model on the same ten inputs may be tested against it, with the expectation, itself informative, that boosting matches or underperforms constrained ridge at this sample size. Any such test enters the configuration registry.

---

# Part V. Trade Construction

## 16. Threshold

Trade frequency is a prior, never an optimisation target. The threshold is chosen by maximising a penalised validation objective on executable P&L:

θ* = argmax_θ [ Sharpe^{net}_{OOS}(θ) − κ·Turnover(θ) − φ·Instability(θ) ]

subject to a prespecified feasibility band of 10 to 50 percent of ordinary days traded, with 30 to 40 percent recorded as the expectation. The live trigger is expressed in forecast-uncertainty units:

trade iff |ŷ_t| / σ̂_{ŷ,t} > z_θ

with σ̂_{ŷ,t} propagated from the fold-level dispersion of coefficient estimates through x_t. This prevents identical raw predictions from being treated equally when model uncertainty differs across refits.

## 17. Position mapping and size

p_t = sign(ŷ_t) · min(1, (|ŷ_t| − θ)/θ) beyond the threshold, zero inside it

N_t = floor( RiskBudget_t · |p_t| / (PointValue · σ*_t · h(A_t)) )

subject to liquidity caps (a maximum participation fraction of median window volume), notional caps, and the catastrophe limits below.

**Volatility conditioner, estimated rather than assumed.** On strictly past data:

log|r^{window}_t| = a + b·A_t + ε, with b ≥ 0, h(A_t) = exp(b̂·A_t) capped to [0.75, 2.0]

tested against the null h = 1. The constraint permits the data to conclude that high unsigned density coincides with pinning and lower post-close volatility, in which case b̂ goes to zero and the conditioner disappears, which is the correct outcome.

## 18. Horizon test

Because the auction-reversion mechanism is documented overnight while the hedging mechanism dissipates within an hour, one counted alternative exit is evaluated: same entry, exit at the next session's 9:30 cash open, on executable P&L with overnight margin and gap risk priced. If the overnight horizon dominates the 14-minute horizon after risk adjustment, the window was wrong and the framework says so. This test doubles as a decomposition of which mechanism carries any edge that appears.

## 19. Execution protocol

Entry: limit order one tick inside the touch at 16:00:15, 15-second timeout, then marketable. Exit: sent marketable at t^exit with no passive attempt, because the exit deadline sits 30 seconds ahead of a trading halt and a missed exit means carrying the position through the halt into the 16:30 reopen. Confirmation check 15 seconds after the exit order, aggressive re-submission on any miss.

No ordinary price stop. A tight technical stop inside a 14-minute thin-book window converts microstructure noise into realised losses, and the risk controls are size, the dead zone, and the time exit. In place of a price stop, a catastrophe rule whose purpose is survival rather than alpha and which should essentially never fire in backtest:

exit immediately if |PnL_t| > L_cat = max( 4σ^{window}_t, L_$ )

plus operational exits on stale quotes, locked or crossed abnormal markets, spread beyond a threshold, loss of the primary data feed, or executed quantity differing from intended quantity.

---

# Part VI. Research Universes and Calendar

## 20. Two disjoint universes

D_ordinary: all sessions excluding quarterly futures and options expirations, FOMC days, month-end and quarter-end, MSCI and FTSE Russell rebalance dates, half days, and reconstitution windows.

D_event: the excluded sessions.

The principal model is fitted and validated only on D_ordinary. Event days are never deleted from the data. They are scored out-of-sample by the frozen model and logged, without ever influencing model selection. Event sessions plausibly contain the largest forced closing flows, and the rebalancing literature documents predictable, revertible price patterns from calendar and threshold rebalancing, so they may ultimately support a separate strategy with its own model and risk budget. Mixing the two distributions through dummies in a sample of several hundred observations is refused. On expiration-heavy sessions in particular, the pinning literature documents materially different closing dynamics, which is precisely why they sit outside the ordinary model.

---

# Part VII. Validation Protocol

## 21. Sample partition

History splits at May 2022, when Tuesday and Thursday SPX expirations completed the daily cycle. Pre-2022 data may inform Stage 1 only. All options features are validated exclusively on post-May-2022 data.

## 22. Estimation and testing procedure

Walk-forward with expanding windows and quarterly refits. Purging removes training observations whose information overlaps validation observations, and a 5-day embargo around each validation fold kills leakage through the rolling normalisations. The internal-fold stacking discipline of Section 13 operates inside every training window.

**Acceptance criteria.** The deflated Sharpe ratio on executable P&L must be positive at a t-statistic above 3.0, per the multiple-testing hurdle now standard in the factor literature, and the reported Sharpe carries the Harvey-Liu haircut computed from the actual configuration count rather than a flat discount. A stationary block bootstrap of daily P&L produces a confidence interval on Sharpe rather than a point estimate. Coefficient sign instability across adjacent folds removes the offending feature regardless of aggregate performance.

**The configuration registry.** Every configuration ever run is logged, including abandoned ones, because the denominator of the multiple-testing correction is the honest count. The registry at specification time includes: the three ρ variants, the three cross-asset composite variants, imbalance specifications I₁ and I₂, the post-16:00 imbalance-message test, the absorption interaction, the full-day-return Stage 1 alternative, the standalone-A_t admission test, the h = 1 null, the overnight-horizon exit, any boosted-model comparison, and the two kernel hyperparameters. Additions to this list during research are appended, never substituted.

## 23. Kill switches

Performance: halt iff Sharpe over max(60 calendar days, 30 trades) < −1. The trade-count minimum exists because a sparse-trading strategy can put only 18 to 24 trades in a 60-day window, and a Sharpe on that few observations is noise.

Forecast degradation: halt iff Σ over the recent n observations of [Loss^{model} − Loss^{null}] exceeds a prespecified threshold, with three nulls monitored: always-flat for the full strategy, Stage 1 for the options block, and imbalance-sign-only for Stage 1 itself.

Slippage: halt iff EWMA_{20}(realised/modelled slippage) > 2, plus a single-trade extreme limit. A consecutive-count rule is rejected because one normal fill resetting a run of severe misses is exactly the wrong behaviour.

Feed integrity: any change in the imbalance feed's format or timing, the SPX close feed, or the options surface vendor's methodology halts trading pending review. The strategy's edge, if it exists, lives in those feeds.

---

# Part VIII. Literature Foundations

## 24. What the evidence establishes, and what it does not

The design leans on four clusters of evidence, used as priors rather than calibrations.

**Dealer hedging feedback.** Gamma-conditioned momentum and reversal are documented across equity options, index futures, and trade-level SPX data, conditional on imbalance size and underlying illiquidity (Barbon-Buraschi, Ni-Pearson-Poteshman and successors, Baltussen-Da-Lammers-Martens). This supports the existence of the mechanism the interaction term tests, and the illiquidity conditioning independently supports the window-liquidity denominator in A_t.

**0DTE specifically.** The evidence is genuinely conflicting on volatility amplification versus dampening, and the best-identified recent work finds net-gamma effects dissipating within about an hour and median net dealer gamma near zero at the relevant scale. This quantifies the prior behind the central design decision and sizes expectations for β₁ and β₂ downward.

**Closing-auction dislocation.** Closing prices are systematically distorted by auction and MOC flows and revert, with the caveat that the headline single-stock overnight reversal coefficient does not map to index level or to a 14-minute horizon, hence the horizon test.

**Validation methodology.** The deflated Sharpe ratio, the backtest-overfitting results, and the multiple-testing hurdles are built into the acceptance criteria directly, because a single closing-window signal tested many ways is exactly the setting those papers warn against.

---

# Part IX. Research Roadmap and Expected Outcome

## 25. Phased plan

**Phase 0, data engineering.** Verify ES intraday coverage of the post-close window. Price and audit the point-in-time imbalance history. Audit the options vendor's surface reconstruction policy. Build the receipt-timestamp schema and the live SPX close estimator. Stand up the exchange-schedule feed. Nothing else proceeds until this phase closes, because every false edge this strategy could manufacture now lives in this phase.

**Phase 1, Stage 1 research.** Build and validate the observable core on the full protocol. Decision gate 1 applies. Deliverable: a validated baseline or a documented null result.

**Phase 2, options block.** Build A_t, PGI^⊥, and P_t on post-May-2022 data with the stacking discipline. Test the incremental contribution on executable P&L. Run the horizon test in parallel.

**Phase 3, paper and live.** Paper-trade with full receipt-time capture to validate latency assumptions and the slippage model. Go live at minimum size with all kill switches armed. Accumulate the live-era sample that the confidence interval on β₂ realistically requires.

**Phase 4, extensions, strictly conditional on a validated core.** Event-day strategy as a separate programme. Vanna readmission via day's IV change interacted with A_t. Depth-based liquidity denominator. Boosted-model comparison.

## 26. Honest prior on the outcome

The most likely result: Stage 1 shows a small edge concentrated in the imbalance features, β₁ comes out near zero because net positioning is balanced at the relevant scale, β₂ on the residualised path-gamma integral is the genuinely interesting coefficient precisely because the momentum alibi has been removed from it, its confidence interval needs two or more years of live-era accumulation before it excludes zero, the horizon test reveals that part of the reversion payoff accrues overnight, and capacity is modest because the window is thin. The less likely result is that nothing survives executable costs. Either way the framework will have done its job, which is to return the answer, including the null, at research cost rather than drawdown cost.

---

# Appendix A. Notation

S_t: ES price. K_i, T_i, Γ_i, OI_i, Q_i: strike, expiry, gamma, open interest, multiplier of contract i. λ_S, λ_T: proximity and expiry kernel decays. σ*_t: floored ex-ante window volatility. Q̂_t: Stage 1 forecast. Q̂^{OOF}: out-of-fold Stage 1 forecast. Ô_t: Stage 2 incremental forecast. λ_t: live blend weight. A_t: unsigned gamma density over window liquidity. PGI^⊥: momentum-residualised path-gamma integral. P_t: concentration-weighted pinning. HHI^Γ: strike-concentration Herfindahl. B^{live}/B^{final}: decision-time and diagnostic basis. θ, z_θ: trade threshold in forecast and uncertainty units. h(A_t): volatility conditioner. L_cat: catastrophe loss limit. D_ordinary/D_event: research universes.

# Appendix B. Prespecified constants

α (EWMA vol decay): 20-day equivalent. Volatility floor: rolling 10th percentile. λ_S initial 0.0075, λ_T initial 3/365, both tuned and counted. c (TY tanh scale): 0.20. ρ grid: {0, 0.25, 0.5}. δ (blend improvement requirement): 0.05. h caps: [0.75, 2.0]. Catastrophe: 4σ^{window} or fixed dollar limit, whichever is larger. Trade-frequency feasibility band: 10 to 50 percent. Kill-switch Sharpe window: max(60 days, 30 trades). Slippage EWMA span: 20 trades, ratio limit 2. Embargo: 5 days. Normalisation window: 252 days ending t−1. Acceptance: deflated Sharpe t > 3.0 with Harvey-Liu haircut. Sample split: May 2022.

# Appendix C. Register of rejected designs

Recorded so the reasoning survives. Per-strike dealer positioning signs: unobservable, and median net dealer gamma is empirically near zero at the relevant scale. Tanh gamma regime switches and gamma-flip confidence machinery: hyperparameters spent smoothing a variable whose sign is already noise. Charm and vanna exposures: positioning-dependent, second-order in this window, readmissible under Phase 4. Fixed hand-set weights: replaced by fitted, constrained, validated coefficients. Full-session ADV as the gamma denominator: replaced by window liquidity. Raw PGI: replaced by the momentum residual. Frequency-targeted threshold: replaced by the penalised objective. R²-ratio blending: replaced by loss comparison with a utility gate. Consecutive-count slippage rule: replaced by EWMA plus extreme limit. Hard-coded 16:15 halt: replaced by the reference-data schedule rule. Event-day dummies: replaced by disjoint universes. Simultaneous imbalance regressors [I_{55}, ΔI, I_{last}]: replaced by alternative specifications. Sign constraints on C^{late} and ΔI: dropped on conditional-meaning grounds.
