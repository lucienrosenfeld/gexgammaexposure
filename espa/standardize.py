"""Robust, strictly-lagged time-series standardisation (Section 8).

z_{j,t} = (x_{j,t} - median_{t-251:t-1}(x_j)) / (1.4826 * MAD_{t-251:t-1}(x_j))

The window for day t ends at t-1 without exception; ``shift(1)`` below is
that exception-free rule, not an implementation convenience. Median and
MAD replace mean and standard deviation because closing variables are
heavy-tailed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


def robust_z(
    x: pd.Series,
    constants: SpecConstants = DEFAULT_CONSTANTS,
    min_periods: int = 60,
) -> pd.Series:
    """Strictly-lagged robust z-score of ``x``.

    The first ``min_periods`` observations are NaN rather than being
    normalised against a short, unstable window.
    """
    lagged = x.shift(1)
    med = lagged.rolling(constants.zscore_window, min_periods=min_periods).median()
    abs_dev = (lagged - med).abs()
    # MAD of the lagged window around the lagged median. Rolling apply of
    # the exact MAD is O(n*w); with the windows involved here that is fine
    # and exactness beats cleverness for a normalisation this load-bearing.
    mad = _rolling_mad(lagged, constants.zscore_window, min_periods)
    scale = constants.mad_scale * mad
    z = (x - med) / scale.replace(0.0, np.nan)
    return z


def _rolling_mad(lagged: pd.Series, window: int, min_periods: int) -> pd.Series:
    def mad(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.nan
        m = np.median(a)
        return float(np.median(np.abs(a - m)))

    return lagged.rolling(window, min_periods=min_periods).apply(mad, raw=True)


def robust_z_frame(
    df: pd.DataFrame,
    constants: SpecConstants = DEFAULT_CONSTANTS,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Column-wise :func:`robust_z`."""
    return pd.DataFrame(
        {c: robust_z(df[c], constants, min_periods) for c in df.columns},
        index=df.index,
    )
