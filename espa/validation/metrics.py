"""Acceptance metrics (Section 22).

- Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): probability that
  the observed Sharpe exceeds the expected maximum Sharpe of N unskilled
  trials, with non-normality corrections.
- Harvey-Liu haircut: the reported Sharpe is shrunk by the
  multiple-testing adjustment computed from the *actual* configuration
  count (the registry), never a flat discount.
- Stationary block bootstrap CI on Sharpe, because a point estimate on
  several hundred observations is an invitation to overread.
- Coefficient sign stability across adjacent folds: a feature whose
  coefficient flips sign fold-to-fold is removed regardless of aggregate
  performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from espa.config import DEFAULT_CONSTANTS, SpecConstants


def sharpe_tstat(pnl: pd.Series) -> tuple[float, float]:
    """(per-period Sharpe, t-statistic) on the executable P&L series."""
    x = pnl.dropna()
    x = x[x != 0.0]
    n = len(x)
    if n < 10 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    sr = float(x.mean() / x.std(ddof=1))
    return sr, sr * np.sqrt(n)


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] over n unskilled trials (false-strategy theorem)."""
    if n_trials <= 1:
        return 0.0
    emc = 0.5772156649015329  # Euler-Mascheroni
    sd = np.sqrt(var_sharpe)
    return float(
        sd
        * (
            (1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
            + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    )


@dataclass(frozen=True)
class NeffResult:
    """R-NEFF-01 output. ``fallback=True`` means the plain count controls.

    Recorded known weaknesses of the estimator (documented, not
    corrected): (a) zero-filling depresses correlations between
    strategies with differing trade dates, biasing N_eff toward the raw
    count — conservative; must never be 'corrected' via pairwise-overlap
    or trade-only correlations; (b) the 80%/80% stability windows share
    60% of observations, overstating stability — tolerable only because
    the check's failure mode defaults to the conservative estimator;
    (c) the participation ratio measures effective *dimensionality* of
    the return covariance, not the effective trial count of a
    max-statistic over correlated tests, and understates the latter at
    moderate correlation — a valid participation ratio is therefore a
    lower bound on effective trials, not the correct value, which is one
    reason the raw count is always computed and logged alongside it.
    """

    n_eff: float
    fallback: bool
    reason: str
    raw_count: int
    participation_ratio: float | None = None
    m_run: int = 0
    m_unrun: int = 0
    n_zero_variance: int = 0
    n_common: int = 0
    neff_first_window: float | None = None
    neff_last_window: float | None = None
    stability_divergence: float | None = None

    def effective(self) -> float:
        return float(self.raw_count) if self.fallback else self.n_eff


def _participation_ratio(returns: pd.DataFrame) -> float:
    corr = np.corrcoef(returns.to_numpy(), rowvar=False)
    corr = np.atleast_2d(corr)
    eig = np.linalg.eigvalsh(corr)
    return float(eig.sum() ** 2 / (eig**2).sum())


def effective_trials(
    returns_by_config: pd.DataFrame,
    m_unrun: int,
    raw_count: int,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> NeffResult:
    """R-NEFF-01: participation-ratio effective trial count with hard
    validity conditions, falling back to the raw (Harvey-Liu) count.

    ``returns_by_config``: daily OOS net returns, days x executed
    configurations, with explicit 0.0 on eligible-flat days and NaN only
    on genuinely ineligible days. The common intersection is the rows on
    which every executed configuration is eligible. Trade-only
    correlation matrices are prohibited by construction — the input
    contract is zero-filled eligible days, asserted below.

    Executed configurations with zero variance over the common
    intersection (all-flat, expected under flat-is-admissible) are
    removed from the matrix and each counted as one fully independent
    trial, mirroring the unrun treatment.
    """
    common = returns_by_config.dropna(axis=0, how="any")
    n_common = len(common)
    assert not common.isna().any().any()  # input contract

    variances = common.var(axis=0)
    zero_var = variances[variances <= 0].index
    n_zero = len(zero_var)
    live = common.drop(columns=zero_var)
    m_run = live.shape[1]

    def _fallback(reason: str) -> NeffResult:
        return NeffResult(
            n_eff=float(raw_count), fallback=True, reason=reason,
            raw_count=raw_count, m_run=m_run, m_unrun=m_unrun,
            n_zero_variance=n_zero, n_common=n_common,
        )

    if m_run < 2:
        return _fallback("fewer than two executed configurations with variance")
    if n_common < constants.neff_min_common_days:
        return _fallback(
            f"common intersection {n_common} < {constants.neff_min_common_days}"
        )
    if n_common < constants.neff_obs_per_config * m_run:
        return _fallback(
            f"common observations {n_common} < {constants.neff_obs_per_config} x "
            f"{m_run} configurations"
        )
    corr = np.corrcoef(live.to_numpy(), rowvar=False)
    eig = np.linalg.eigvalsh(np.atleast_2d(corr))
    if eig.min() < -1e-8:
        return _fallback("correlation matrix not PSD up to numerical error")

    pr_full = _participation_ratio(live)
    k = int(np.floor(constants.neff_stability_frac * n_common))
    pr_first = _participation_ratio(live.iloc[:k])
    pr_last = _participation_ratio(live.iloc[-k:])
    div = max(abs(pr_first - pr_full), abs(pr_last - pr_full)) / pr_full
    if div > constants.neff_stability_tol:
        return _fallback(
            f"stability check failed: divergence {div:.3f} > "
            f"{constants.neff_stability_tol}"
        )

    return NeffResult(
        n_eff=pr_full + m_unrun + n_zero,
        fallback=False,
        reason="valid",
        raw_count=raw_count,
        participation_ratio=pr_full,
        m_run=m_run,
        m_unrun=m_unrun,
        n_zero_variance=n_zero,
        n_common=n_common,
        neff_first_window=pr_first,
        neff_last_window=pr_last,
        stability_divergence=div,
    )


def deflated_sharpe_ratio(
    pnl: pd.Series,
    n_trials: int,
    var_trial_sharpe: float | None = None,
    neff: NeffResult | None = None,
) -> tuple[float, float]:
    """(DSR probability, benchmark SR0) for the executable P&L series.

    DSR = P[SR > SR0], where SR0 is the expected maximum Sharpe among
    ``n_trials`` configurations. ``var_trial_sharpe`` is the cross-trial
    variance of Sharpe estimates; defaults to the estimator variance of
    this series' Sharpe, a conservative stand-in when the registry does
    not yet hold per-trial Sharpes.

    ``neff``: optional R-NEFF-01 result. When supplied and not flagged
    fallback, its effective count replaces ``n_trials``. Callers must
    always compute and log the raw-count result alongside — the
    correction may change the headline, never hide the conservative
    number.
    """
    if neff is not None and not neff.fallback:
        n_trials = max(1, int(round(neff.effective())))
    x = pnl.dropna()
    x = x[x != 0.0]
    n = len(x)
    if n < 20 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    sr = float(x.mean() / x.std(ddof=1))
    g3 = float(stats.skew(x))
    g4 = float(stats.kurtosis(x, fisher=False))
    if var_trial_sharpe is None:
        var_trial_sharpe = (1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (n - 1)
    sr0 = expected_max_sharpe(n_trials, max(var_trial_sharpe, 1e-12))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z)), float(sr0)


def harvey_liu_haircut(
    sharpe: float, n_obs: int, n_trials: int, neff: NeffResult | None = None
) -> float:
    """Haircut Sharpe via Bonferroni-adjusted p-value (Harvey & Liu 2015).

    p_adj = min(1, n_trials * p); the haircut Sharpe is the one whose
    single-test p-value equals p_adj. Bonferroni is the most conservative
    of the three adjustments in the paper, which is the correct default
    when the true correlation structure across trials is unknown.

    ``neff`` follows the same contract as in
    :func:`deflated_sharpe_ratio`: a valid R-NEFF-01 result replaces
    ``n_trials``; the raw-count haircut must still be computed and
    logged by the caller.
    """
    if neff is not None and not neff.fallback:
        n_trials = max(1, int(round(neff.effective())))
    if not np.isfinite(sharpe) or n_obs < 10:
        return np.nan
    t = sharpe * np.sqrt(n_obs)
    p = 2 * (1 - stats.t.cdf(abs(t), df=n_obs - 1))
    p_adj = min(1.0, max(n_trials, 1) * p)
    if p_adj >= 1.0:
        return 0.0
    t_adj = stats.t.ppf(1 - p_adj / 2, df=n_obs - 1)
    return float(np.sign(sharpe) * t_adj / np.sqrt(n_obs))


def block_bootstrap_sharpe_ci(
    pnl: pd.Series,
    n_boot: int = 2000,
    mean_block: int = 10,
    ci: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Stationary block bootstrap (Politis-Romano) CI on per-period Sharpe."""
    x = pnl.dropna().to_numpy()
    x = x[x != 0.0]
    n = x.size
    if n < 30:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        s = x[idx]
        sd = s.std(ddof=1)
        sharpes[b] = s.mean() / sd if sd > 0 else np.nan
    lo, hi = np.nanquantile(sharpes, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def sign_stability(
    coef_by_fold: pd.DataFrame,
    tol: float = 1e-10,
    coef_se_by_fold: pd.DataFrame | None = None,
    materiality_mult: float = 1.0,
) -> pd.Series:
    """Fraction of adjacent-fold pairs on which each coefficient keeps its sign.

    ``coef_by_fold``: rows = folds (chronological), columns = features.
    Zero coefficients are neutral — shrinkage to zero is the designed
    response to a weak feature, not an instability. A feature scoring
    below 1.0 flipped sign between at least one adjacent pair and is
    removed regardless of aggregate performance (Section 22).

    Materiality floor (research-round amendment): when
    ``coef_se_by_fold`` is supplied, a fold pair contributes to the
    instability count only if |coef| > materiality_mult x SE on *both*
    sides of the flip. This is the D-STAGE2-ID-01 floor, and — per the
    amendment's resolved tension — it also governs the *removal* rule
    for Stage 2 features, whose honest prior is a coefficient near zero
    that flips sign as pure noise. Default behaviour (no SEs supplied)
    is unchanged, so the Stage 1 removal rule, whose constrained
    coefficients shrink to exact zeros, is not silently weakened.
    """
    out = {}
    for col in coef_by_fold.columns:
        v = coef_by_fold[col].to_numpy()
        se = (
            coef_se_by_fold[col].to_numpy()
            if coef_se_by_fold is not None and col in coef_se_by_fold.columns
            else None
        )
        pairs = stable = 0
        for i, (a, b) in enumerate(zip(v[:-1], v[1:])):
            if abs(a) <= tol or abs(b) <= tol:
                continue
            if se is not None:
                sa, sb = se[i], se[i + 1]
                material = (
                    np.isfinite(sa)
                    and np.isfinite(sb)
                    and abs(a) > materiality_mult * sa
                    and abs(b) > materiality_mult * sb
                )
                if not material:
                    continue  # immaterial flip: neither counted nor stable
            pairs += 1
            stable += int(np.sign(a) == np.sign(b))
        out[col] = stable / pairs if pairs else 1.0
    return pd.Series(out, name="sign_stability")
