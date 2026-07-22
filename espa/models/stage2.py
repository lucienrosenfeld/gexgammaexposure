"""Stage 2 and the mandatory stacking discipline (Section 13).

    Ohat = beta1 * A * Qhat_OOF + beta2 * PGI_perp + beta3 * P

Three coefficients, ridge-shrunk, all sign-unconstrained, fitted on
Stage 1 *residual* structure. Stage 2 never refits Stage 1 variables and
never includes them as regressors alongside Qhat.

Stacking discipline, mandatory: within each training window, internal
chronological folds generate Qhat_OOF; Stage 2 is fitted against those;
Stage 1 is then refit on the complete training window to produce
validation-period Qhat; Stage 2 coefficients are applied without
refitting. In-sample Stage 1 fitted values never touch Stage 2
estimation — otherwise the interaction term inherits Stage 1's in-sample
optimism, the exact direction of error this architecture exists to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants
from espa.features.pgi import PGIResidualiser, residualise_pgi
from espa.models.constrained_ridge import ConstrainedRidge
from espa.models.stage1 import Stage1Model, out_of_fold_predictions

STAGE2_FEATURES = ("A_x_Qhat", "PGI_perp", "P")


def bootstrap_coef_se(
    features: pd.DataFrame,
    residual: pd.Series,
    lam: float,
    constants: SpecConstants = DEFAULT_CONSTANTS,
    seed: int = 17,
) -> pd.Series:
    """Fold-level coefficient SEs via stationary block bootstrap (frozen).

    This is the SE method fixed before Phase 1 per amendment section 2.7:
    stationary block bootstrap (Politis-Romano, mean block
    ``id_bootstrap_mean_block``) over the training fold's (X, y) rows,
    ``id_bootstrap_resamples`` refits, identical across folds. These SEs
    serve the D-STAGE2-ID-01 materiality floor and the Stage 2 removal
    rule. They must NOT be conflated with ForecastUncertainty's
    cross-refit dispersion, which serves the trade trigger — the two
    measure different things (within-fold sampling noise versus
    across-refit model drift).
    """
    X = features.to_numpy(dtype=float)
    y = residual.reindex(features.index).to_numpy(dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    p_new = 1.0 / constants.id_bootstrap_mean_block
    coefs = []
    for _ in range(constants.id_bootstrap_resamples):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p_new else (idx[t - 1] + 1) % n
        try:
            m = ConstrainedRidge(lam=lam, fit_intercept=False).fit(
                X[idx], y[idx], [0] * X.shape[1]
            )
            coefs.append(m.coef_)
        except ValueError:
            continue
    if len(coefs) < 10:
        return pd.Series(np.nan, index=features.columns)
    return pd.Series(np.std(np.vstack(coefs), axis=0, ddof=1), index=features.columns)


@dataclass
class Stage2Model:
    lam: float = 1.0
    model: ConstrainedRidge = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # No intercept: Stage 2 models residual structure around Stage 1,
        # and a fitted level would leak Stage 1's bias back in.
        self.model = ConstrainedRidge(lam=self.lam, fit_intercept=False)

    def fit(self, features: pd.DataFrame, residual: pd.Series) -> "Stage2Model":
        self.model.fit(
            features.to_numpy(),
            residual.reindex(features.index).to_numpy(),
            [0, 0, 0],
            feature_names=list(features.columns),
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        pred = np.full(len(features), np.nan)
        clean = ~features.isna().any(axis=1)
        if clean.any():
            pred[clean.to_numpy()] = self.model.predict(features.loc[clean].to_numpy())
        return pd.Series(pred, index=features.index, name="O_hat")

    @property
    def coef(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self.model.feature_names_)


def stage2_features(
    a_density: pd.Series, q_hat: pd.Series, pgi_perp: pd.Series, pinning: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {"A_x_Qhat": a_density * q_hat, "PGI_perp": pgi_perp, "P": pinning}
    )


def robust_scales(features: pd.DataFrame) -> pd.Series:
    """Per-feature robust scale (1.4826 * MAD), computed on training data.

    Stage 2 features live on wildly different natural scales — the raw
    path-gamma statistic carries Gamma * OI * S^2 units while the
    interaction and pinning terms are O(1) — and a single ridge penalty
    across unscaled features silently deletes whichever feature has the
    largest units. Scales are fitted on the training window only and
    frozen into the :class:`StackedFit` (Section 8's lagged-history rule
    at the fold level).
    """
    scales = {}
    for c in features.columns:
        x = features[c].dropna()
        s = 1.4826 * float((x - x.median()).abs().median()) if len(x) else 0.0
        if s <= 0:
            s = float(x.std()) if len(x) > 1 else 0.0
        scales[c] = s if s > 0 else 1.0
    return pd.Series(scales)


@dataclass
class StackedFit:
    """One training window's frozen artefacts, applied forward unchanged."""

    stage1: Stage1Model
    stage2: Stage2Model
    pgi_residualiser: PGIResidualiser
    stage2_scales: pd.Series
    #: Fold-level Stage 2 coefficient SEs (frozen bootstrap, section 2.7)
    #: for the D-STAGE2-ID-01 materiality floor. NaN when Stage 2 was not
    #: estimable on this window.
    stage2_se: pd.Series = None
    #: The exact standardised training design used in Stage 2 estimation,
    #: exposed for fold_identification. None when Stage 2 was not
    #: estimable.
    stage2_design: pd.DataFrame = None

    def forecast(
        self,
        stage1_features: pd.DataFrame,
        a_density: pd.Series,
        pgi_raw: pd.Series,
        c_early: pd.Series,
        c_late: pd.Series,
        r_day: pd.Series,
        pinning: pd.Series,
    ) -> pd.DataFrame:
        """Validation-period Qhat and Ohat; blending happens downstream."""
        q_hat = self.stage1.predict(stage1_features)
        pgi_perp = self.pgi_residualiser.transform(pgi_raw, c_early, c_late, r_day)
        feats = stage2_features(a_density, q_hat, pgi_perp, pinning) / self.stage2_scales
        o_hat = self.stage2.predict(feats)
        return pd.DataFrame({"Q_hat": q_hat, "O_hat": o_hat})


def fit_stage2_stacked(
    stage1_features: pd.DataFrame,
    constraints: dict[str, int],
    y_mid: pd.Series,
    a_density: pd.Series,
    pgi_raw: pd.Series,
    c_early: pd.Series,
    c_late: pd.Series,
    r_day: pd.Series,
    pinning: pd.Series,
    lam_stage1: float,
    lam_stage2: float,
    n_folds: int = 5,
) -> StackedFit:
    """Run the full stacking discipline on one training window.

    All inputs are training-window series only; the caller (the
    walk-forward engine) guarantees purging and embargo.
    """
    # 1. Out-of-fold Stage 1 predictions inside the training window.
    q_oof = out_of_fold_predictions(stage1_features, y_mid, constraints, lam_stage1, n_folds)

    # 2. PGI residualisation fitted on training data only, then frozen.
    resid = residualise_pgi(pgi_raw, c_early, c_late, r_day)
    pgi_perp = resid.transform(pgi_raw, c_early, c_late, r_day)

    # 3. Stage 2 fitted against OOF predictions on Stage 1 residuals,
    #    on features standardised by training-window robust scales.
    feats2 = stage2_features(a_density, q_oof, pgi_perp, pinning)
    scales = robust_scales(feats2)
    feats2 = feats2 / scales
    residual = (y_mid - q_oof).rename("stage1_residual")
    ok = feats2.dropna().index.intersection(residual.dropna().index)
    stage2 = Stage2Model(lam=lam_stage2)
    stage2_se = pd.Series(np.nan, index=list(STAGE2_FEATURES))
    stage2_design = None
    if len(ok) >= 10:
        stage2.fit(feats2.loc[ok], residual.loc[ok])
        stage2_design = feats2.loc[ok]
        stage2_se = bootstrap_coef_se(feats2.loc[ok], residual.loc[ok], lam_stage2)
    else:  # not enough options-era data: options block contributes nothing
        stage2.model.coef_ = np.zeros(3)
        stage2.model.feature_names_ = list(STAGE2_FEATURES)

    # 4. Stage 1 refit on the complete training window for validation use.
    feat_clean = stage1_features.dropna()
    y_clean = y_mid.reindex(feat_clean.index).dropna()
    stage1 = Stage1Model(lam=lam_stage1).fit(
        feat_clean.loc[y_clean.index], y_clean, constraints
    )

    return StackedFit(
        stage1=stage1, stage2=stage2, pgi_residualiser=resid, stage2_scales=scales,
        stage2_se=stage2_se, stage2_design=stage2_design,
    )
