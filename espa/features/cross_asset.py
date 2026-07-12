"""Cross-asset composite X over the 15:30 -> 16:00 horizon (Section 10).

Primary variant is the fixed composite:

    s_TY = tanh(Corr_60(r_ES, r_TY) / c),  c = 0.20 fixed
    X = [z(r_NQ) + z(r_RTY) + z(r_YM) - z(r_DXY) + s_TY * z(r_TY)] / (4 + |s_TY|)

The Treasury weight is a bounded continuous transform of a noisy lagged
correlation and carries no fitted parameters; a hard sign function would
make binary jumps on correlation estimates near zero. The TY-free
composite and an expanding-window PCA variant are counted alternatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants
from espa.standardize import robust_z


def treasury_weight(
    r_es: pd.Series,
    r_ty: pd.Series,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.Series:
    """s_TY from the strictly lagged rolling ES/TY return correlation."""
    corr = (
        r_es.shift(1)
        .rolling(constants.ty_corr_window, min_periods=constants.ty_corr_window // 2)
        .corr(r_ty.shift(1))
    )
    return np.tanh(corr / constants.ty_tanh_scale)


def cross_asset_composite(
    returns: pd.DataFrame,
    variant: str = "fixed",
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> pd.Series:
    """X_t from 15:30->16:00 returns of NQ, RTY, YM, DXY, TY (and ES for s_TY).

    ``returns`` needs columns: es, nq, rty, ym, dxy, ty.
    """
    z = {c: robust_z(returns[c], constants) for c in ("nq", "rty", "ym", "dxy", "ty")}
    if variant == "fixed":
        s_ty = treasury_weight(returns["es"], returns["ty"], constants)
        num = z["nq"] + z["rty"] + z["ym"] - z["dxy"] + s_ty * z["ty"]
        return (num / (4.0 + s_ty.abs())).rename("X")
    if variant == "no_ty":
        return ((z["nq"] + z["rty"] + z["ym"] - z["dxy"]) / 4.0).rename("X")
    if variant == "pca":
        return _expanding_pca_composite(returns, constants).rename("X")
    raise ValueError(f"unknown cross-asset variant: {variant}")


def _expanding_pca_composite(
    returns: pd.DataFrame, constants: SpecConstants, min_obs: int = 120
) -> pd.Series:
    """Expanding-window first principal component of the five z-scored series.

    Orientation is fixed each day by requiring positive correlation of
    the component loadings' output with the pre-close ES return, per the
    spec's sign-indeterminacy remedy. Counted alternative; rolling PCA
    loadings on five series are unstable, which is why this is not primary.
    """
    cols = ["nq", "rty", "ym", "dxy", "ty"]
    zdf = pd.DataFrame({c: robust_z(returns[c], constants) for c in cols})
    out = pd.Series(np.nan, index=returns.index)
    zv = zdf.to_numpy()
    es = returns["es"].to_numpy()
    for t in range(min_obs, len(returns)):
        hist = zv[:t]  # strictly lagged
        mask = ~np.isnan(hist).any(axis=1)
        if mask.sum() < min_obs // 2:
            continue
        h = hist[mask]
        cov = np.cov(h, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        pc = vecs[:, -1]
        # orient: component scores must correlate positively with ES return
        scores = h @ pc
        es_hist = es[:t][mask]
        ok = ~np.isnan(es_hist)
        if ok.sum() > 2 and np.corrcoef(scores[ok], es_hist[ok])[0, 1] < 0:
            pc = -pc
        row = zv[t]
        if not np.isnan(row).any():
            out.iloc[t] = float(row @ pc)
    return out
