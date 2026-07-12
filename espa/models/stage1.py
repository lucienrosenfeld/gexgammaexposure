"""Stage 1: Qhat = f_Q(x_Q) (Section 12).

Ridge on the seven Stage 1 features with the frozen sign constraints,
refit quarterly on an expanding window. The ridge penalty is chosen by
the walk-forward machinery on *executable P&L* (espa.validation), not
here; this class is the estimator only.

Decision gate 1 lives in :mod:`espa.validation.metrics`: if Stage 1
shows no deflated-Sharpe-significant edge after costs out-of-sample, the
programme stops. If it passes, Stage 1 is the production fallback that
everything else must beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from espa.models.constrained_ridge import ConstrainedRidge


@dataclass
class Stage1Model:
    lam: float = 1.0
    model: ConstrainedRidge = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = ConstrainedRidge(lam=self.lam)

    def fit(
        self,
        features: pd.DataFrame,
        y_mid: pd.Series,
        constraints: dict[str, int],
    ) -> "Stage1Model":
        cons = [constraints[c] for c in features.columns]
        self.model.fit(
            features.to_numpy(),
            y_mid.reindex(features.index).to_numpy(),
            cons,
            feature_names=list(features.columns),
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        pred = np.full(len(features), np.nan)
        clean = ~features.isna().any(axis=1)
        if clean.any():
            pred[clean.to_numpy()] = self.model.predict(features.loc[clean].to_numpy())
        return pd.Series(pred, index=features.index, name="Q_hat")

    @property
    def coef(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self.model.feature_names_)


def out_of_fold_predictions(
    features: pd.DataFrame,
    y_mid: pd.Series,
    constraints: dict[str, int],
    lam: float,
    n_folds: int = 5,
) -> pd.Series:
    """Chronological out-of-fold Qhat inside a training window.

    Fold k's predictions come from a model fitted on all *other* folds.
    Chronological (not shuffled) so the fold structure respects the time
    series; used exclusively to feed Stage 2 estimation (Section 13's
    stacking discipline).
    """
    idx = features.index
    fold_edges = np.linspace(0, len(idx), n_folds + 1, dtype=int)
    oof = pd.Series(np.nan, index=idx, name="Q_hat_oof")
    for k in range(n_folds):
        val_idx = idx[fold_edges[k] : fold_edges[k + 1]]
        train_idx = idx.difference(val_idx)
        train_feat = features.loc[train_idx].dropna()
        train_y = y_mid.reindex(train_feat.index).dropna()
        train_feat = train_feat.loc[train_y.index]
        if len(train_feat) < len(features.columns) + 5:
            continue
        m = Stage1Model(lam=lam).fit(train_feat, train_y, constraints)
        oof.loc[val_idx] = m.predict(features.loc[val_idx]).to_numpy()
    return oof
