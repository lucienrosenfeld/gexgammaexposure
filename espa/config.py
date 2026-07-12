"""Prespecified constants (Appendix B) and configuration containers.

Every constant that Appendix B freezes lives here as a frozen dataclass.
Values that the spec designates as *tuned and counted* hyperparameters
(lambda_S, lambda_T, the rho grid) carry their initial values here and
must be varied only through the configuration registry
(:mod:`espa.validation.registry`), never edited in place.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecConstants:
    """Appendix B, verbatim. Frozen: changing a value is a new configuration."""

    # --- volatility scale (Section 9) ---
    #: EWMA decay for the ex-ante window volatility; 20-day-equivalent span.
    vol_ewma_span: int = 20
    #: Number of lagged |returns| entering the truncated EWMA sum.
    vol_ewma_terms: int = 20
    #: Rolling quantile used as the volatility floor.
    vol_floor_quantile: float = 0.10
    #: Window (obs) for the rolling floor quantile.
    vol_floor_window: int = 252

    # --- robust standardisation (Section 8) ---
    #: Rolling normalisation window, ending t-1 without exception.
    zscore_window: int = 252
    #: Consistency constant making MAD comparable to a standard deviation.
    mad_scale: float = 1.4826

    # --- options kernels (Section 11); initial values, tuned and counted ---
    lambda_s: float = 0.0075
    lambda_t: float = 3.0 / 365.0
    #: Discrete grid for the same-day-volume blend; never tuned continuously.
    rho_grid: tuple[float, ...] = (0.0, 0.25, 0.5)
    #: Window (days) for the median window-liquidity denominator of A_t.
    liquidity_median_window: int = 20

    # --- cross-asset composite (Section 10) ---
    #: tanh scale for the Treasury correlation weight; fixed and prespecified.
    ty_tanh_scale: float = 0.20
    #: Rolling window (obs) for the lagged ES/TY correlation.
    ty_corr_window: int = 60

    # --- blending (Section 14) ---
    #: EWMA span (trades) for the forecast-loss comparison.
    blend_loss_span: int = 60
    #: delta: relative improvement required for full options weight.
    blend_delta: float = 0.05
    #: Trailing trade count for the utility gate on the options block.
    utility_gate_window: int = 60

    # --- volatility conditioner (Section 17) ---
    h_cap_low: float = 0.75
    h_cap_high: float = 2.0

    # --- trade construction (Sections 16-17, 19) ---
    #: Feasibility band on fraction of ordinary days traded.
    trade_frequency_band: tuple[float, float] = (0.10, 0.50)
    #: Catastrophe limit in window-sigma units (paired with a dollar floor).
    catastrophe_sigma_mult: float = 4.0
    #: Seconds before the next halt at which the exit order is sent.
    exit_lead_seconds: int = 30
    #: Research-window endpoint when a session has no post-close halt.
    default_exit_time: dt.time = dt.time(16, 14, 30)
    #: Decision / entry time.
    decision_time: dt.time = dt.time(16, 0, 15)
    #: Entry passive-order timeout before going marketable.
    entry_timeout_seconds: int = 15
    #: Exit confirmation check delay.
    exit_confirm_seconds: int = 15

    # --- kill switches (Section 23) ---
    kill_sharpe_days: int = 60
    kill_sharpe_min_trades: int = 30
    kill_sharpe_level: float = -1.0
    slippage_ewma_span: int = 20
    slippage_ratio_limit: float = 2.0

    # --- validation (Sections 21-22) ---
    embargo_days: int = 5
    #: Deflated-Sharpe acceptance t-statistic.
    acceptance_tstat: float = 3.0
    #: Sample split for the options universe: daily-expiration completion.
    sample_split_date: dt.date = dt.date(2022, 5, 16)
    #: Latest receipt time for primary imbalance features (Section 7).
    imbalance_cutoff: dt.time = dt.time(15, 59, 55)

    # --- Stage 1 refit cadence ---
    refit_frequency_days: int = 63  # quarterly


DEFAULT_CONSTANTS = SpecConstants()


@dataclass(frozen=True)
class ThresholdPenalty:
    """Penalty weights of the threshold objective (Section 16).

    kappa and phi are configuration-level choices, logged in the registry.
    """

    kappa: float = 0.01
    phi: float = 0.5


@dataclass
class RunConfig:
    """One registered configuration of the strategy.

    Everything that Section 22 counts as a configuration is a field here,
    so the multiple-testing denominator is the number of distinct
    ``RunConfig`` instances ever evaluated.
    """

    name: str
    rho: float = 0.0
    imbalance_spec: str = "I1"  # "I1" -> [I_55, dI]; "I2" -> [I_last, I_last - I_55]
    cross_asset_variant: str = "fixed"  # "fixed" | "no_ty" | "pca"
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t
    use_absorption_interaction: bool = False
    use_full_day_return_stage1: bool = False
    use_standalone_gamma_density: bool = False
    overnight_horizon: bool = False
    boosted_comparison: bool = False
    threshold_penalty: ThresholdPenalty = field(default_factory=ThresholdPenalty)
    notes: str = ""

    def key(self) -> tuple:
        """Identity of the configuration for registry counting."""
        return (
            self.rho,
            self.imbalance_spec,
            self.cross_asset_variant,
            self.lambda_s,
            self.lambda_t,
            self.use_absorption_interaction,
            self.use_full_day_return_stage1,
            self.use_standalone_gamma_density,
            self.overnight_horizon,
            self.boosted_comparison,
            self.threshold_penalty.kappa,
            self.threshold_penalty.phi,
        )
