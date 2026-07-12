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

import numpy as np
import pandas as pd
from scipy import stats


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


def deflated_sharpe_ratio(
    pnl: pd.Series,
    n_trials: int,
    var_trial_sharpe: float | None = None,
) -> tuple[float, float]:
    """(DSR probability, benchmark SR0) for the executable P&L series.

    DSR = P[SR > SR0], where SR0 is the expected maximum Sharpe among
    ``n_trials`` configurations. ``var_trial_sharpe`` is the cross-trial
    variance of Sharpe estimates; defaults to the estimator variance of
    this series' Sharpe, a conservative stand-in when the registry does
    not yet hold per-trial Sharpes.
    """
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


def harvey_liu_haircut(sharpe: float, n_obs: int, n_trials: int) -> float:
    """Haircut Sharpe via Bonferroni-adjusted p-value (Harvey & Liu 2015).

    p_adj = min(1, n_trials * p); the haircut Sharpe is the one whose
    single-test p-value equals p_adj. Bonferroni is the most conservative
    of the three adjustments in the paper, which is the correct default
    when the true correlation structure across trials is unknown.
    """
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


def sign_stability(coef_by_fold: pd.DataFrame, tol: float = 1e-10) -> pd.Series:
    """Fraction of adjacent-fold pairs on which each coefficient keeps its sign.

    ``coef_by_fold``: rows = folds (chronological), columns = features.
    Zero coefficients are neutral — shrinkage to zero is the designed
    response to a weak feature, not an instability. A feature scoring
    below 1.0 flipped sign between at least one adjacent pair and is
    removed regardless of aggregate performance (Section 22).
    """
    out = {}
    for col in coef_by_fold.columns:
        v = coef_by_fold[col].to_numpy()
        pairs = stable = 0
        for a, b in zip(v[:-1], v[1:]):
            if abs(a) <= tol or abs(b) <= tol:
                continue
            pairs += 1
            stable += int(np.sign(a) == np.sign(b))
        out[col] = stable / pairs if pairs else 1.0
    return pd.Series(out, name="sign_stability")
