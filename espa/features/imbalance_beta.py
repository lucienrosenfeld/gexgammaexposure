"""R-IMB-BETA-01: beta-weighted auction imbalance (frozen pipeline).

One counted *replacement* for the cap-weighted imbalance aggregation,
never an additional regressor:

    baseline  I_w = sum_i w_i q_i
    candidate I_beta = sum_i w_i beta~_i q_i

Frozen beta pipeline: rolling covariance beta of daily close-to-close
constituent returns against ES, 252-day window ending t-1, point-in-time
membership only, minimum 120 paired observations; shrinkage
beta_shrunk = 0.5*beta_hat + 0.5; clip to [0.25, 2.5]; fewer than 120
observations => beta~ = 1. No same-day return may enter.

NO PARAMETER OF THIS PIPELINE MAY BE VARIED except through a new,
separately counted registry entry (Appendix C item 7 of the amendment):
no alternative lookback, shrinkage, prior, clip, frequency, or fallback.

The pre-registration redundancy screen (|rho| between the standardised
candidate and baseline over all available point-in-time history) decides
ADMIT / REJECT / DEFER before any predictive evaluation; a rejection is
a successful result discovered at zero denominator cost, and a deferral
(history below the 250-day floor) is not a decision — the configuration
stays in m_unrun until the floor is met.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants
from espa.standardize import robust_z


def rolling_es_betas(
    constituent_returns: pd.DataFrame,
    es_returns: pd.Series,
    membership: pd.DataFrame,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.DataFrame:
    """Shrunk, clipped, strictly-lagged rolling betas per constituent.

    ``constituent_returns``: daily close-to-close returns, days x names.
    ``es_returns``: daily ES close-to-close returns.
    ``membership``: point-in-time membership mask (bool, days x names);
    non-members are NaN in the output and must be excluded upstream.

    The window ends at t-1 without exception — enforced here by the
    ``shift(1)`` on both legs, so no same-day return can enter no matter
    what the caller passes.
    """
    if not constituent_returns.index.equals(membership.index):
        raise ValueError("membership mask must share the return index")
    lag_r = constituent_returns.shift(1)
    lag_es = es_returns.shift(1)
    w, m = constants.beta_lookback_days, constants.beta_min_obs

    cov = lag_r.rolling(w, min_periods=m).cov(lag_es)
    var = lag_es.rolling(w, min_periods=m).var()
    beta_hat = cov.div(var, axis=0)

    # paired-observation floor: both legs non-NaN inside the window
    paired = (
        (lag_r.notna() & lag_es.notna().to_numpy()[:, None])
        .rolling(w, min_periods=1)
        .sum()
    )
    beta_hat = beta_hat.where(paired >= m)

    shrunk = constants.beta_shrinkage * beta_hat + (1 - constants.beta_shrinkage) * 1.0
    lo, hi = constants.beta_clip
    clipped = shrunk.clip(lower=lo, upper=hi)
    tilde = clipped.fillna(constants.beta_fallback)
    return tilde.where(membership, np.nan)


def beta_weighted_imbalance(
    q: pd.DataFrame,
    w: pd.DataFrame,
    betas: pd.DataFrame,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.Series:
    """I_beta,t = sum_i w_i beta~_i q_i for one snapshot panel.

    ``q``: constituent imbalance quantities (days x names), ``w``: index
    weights (days x names). Called once for the 15:50 panel and once for
    15:55. Where q and w are present but the beta is missing (e.g. a
    just-added member), the frozen fallback beta = 1 applies.
    """
    b = betas.reindex_like(q)
    b = b.where(~(q.notna() & w.notna() & b.isna()), constants.beta_fallback)
    return (w * b * q).sum(axis=1, min_count=1).rename("I_beta")


def cap_weighted_imbalance(q: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Baseline I_w = sum_i w_i q_i on the same panels."""
    return (w * q).sum(axis=1, min_count=1).rename("I_w")


class ScreenStatus(enum.Enum):
    ADMIT = "admit"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True)
class ScreenResult:
    status: ScreenStatus
    rho: float | None
    n_days_history: int
    n_usable: int
    threshold: float
    min_days: int

    def to_registry_record(self) -> dict:
        """Serialises into the registry's rejection/deferral/admission form."""
        return {
            "rule": "R-IMB-BETA-01 redundancy screen: reject iff "
            f"|Corr[z(I_beta), z(I_w)]| > {self.threshold}; defer iff "
            f"history < {self.min_days} point-in-time trading days",
            "rho": self.rho,
            "n_days_history": self.n_days_history,
            "n_usable_z_days": self.n_usable,
            "status": self.status.value,
        }


def redundancy_screen(
    i_beta: pd.Series,
    i_w: pd.Series,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> ScreenResult:
    """Pre-registration redundancy screen on the production standardisation.

    Computed over all available point-in-time history, before any
    targets, losses, Sharpes, or coefficients are inspected. The
    threshold was frozen before computation. Below the sample floor the
    screen is deferred, not decided. The usable-day count (days on which
    both z-scores exist under the strictly-lagged standardisation) is
    logged alongside the calendar count.
    """
    common = i_beta.dropna().index.intersection(i_w.dropna().index)
    n_hist = len(common)
    if n_hist < constants.beta_screen_min_days:
        return ScreenResult(
            ScreenStatus.DEFER, None, n_hist, 0,
            constants.beta_screen_threshold, constants.beta_screen_min_days,
        )
    zb = robust_z(i_beta.loc[common], constants)
    zw = robust_z(i_w.loc[common], constants)
    ok = zb.notna() & zw.notna()
    n_usable = int(ok.sum())
    if n_usable < 30:
        return ScreenResult(
            ScreenStatus.DEFER, None, n_hist, n_usable,
            constants.beta_screen_threshold, constants.beta_screen_min_days,
        )
    rho = float(np.corrcoef(zb[ok], zw[ok])[0, 1])
    status = (
        ScreenStatus.REJECT
        if abs(rho) > constants.beta_screen_threshold
        else ScreenStatus.ADMIT
    )
    return ScreenResult(
        status, rho, n_hist, n_usable,
        constants.beta_screen_threshold, constants.beta_screen_min_days,
    )
