"""Sign-constrained ridge regression (Section 12).

Implemented, per the spec, as non-negative least squares on sign-flipped
constrained features with an L2 penalty:

- a feature constrained to w >= 0 enters as-is;
- a feature constrained to w <= 0 enters sign-flipped (its NNLS weight
  is then negated on the way out);
- an unconstrained feature enters twice, as (x, -x), the standard
  positive/negative split w = w+ - w-. The ridge penalty makes at most
  one of the pair nonzero at the optimum, so the split is exact.

Ridge is applied by row augmentation: sqrt(lam) * I appended to the
design, zeros appended to the target, which keeps the whole problem a
single NNLS call.

Sign constraints are the principal small-sample defence: a coefficient
the data wants to flip against a strong economic prior is shrunk to zero
instead of admitted with the wrong sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls


@dataclass
class ConstrainedRidge:
    """Ridge with per-feature sign constraints (+1, -1, or 0 = free)."""

    lam: float = 1.0
    fit_intercept: bool = True
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0
    feature_names_: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        constraints: list[int] | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "ConstrainedRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        constraints = np.asarray(constraints, dtype=int)
        if X.ndim != 2 or X.shape[1] != constraints.size:
            raise ValueError("X columns must match the constraint vector")
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        if X.shape[0] < X.shape[1] + 2:
            raise ValueError("not enough clean observations to fit")

        y_off = float(np.mean(y)) if self.fit_intercept else 0.0
        yc = y - y_off

        # Build the NNLS design: one column per constrained feature
        # (flipped if the constraint is <= 0), two per free feature.
        cols, backmap = [], []  # backmap: (orig_index, sign_multiplier)
        for j, c in enumerate(constraints):
            if c > 0:
                cols.append(X[:, j])
                backmap.append((j, +1.0))
            elif c < 0:
                cols.append(-X[:, j])
                backmap.append((j, -1.0))
            else:
                cols.append(X[:, j])
                backmap.append((j, +1.0))
                cols.append(-X[:, j])
                backmap.append((j, -1.0))
        D = np.column_stack(cols)

        if self.lam > 0:
            aug = np.sqrt(self.lam) * np.eye(D.shape[1])
            D = np.vstack([D, aug])
            yc = np.concatenate([yc, np.zeros(aug.shape[0])])

        w_nn, _ = nnls(D, yc)

        coef = np.zeros(constraints.size)
        for w, (j, sgn) in zip(w_nn, backmap):
            coef[j] += sgn * w
        self.coef_ = coef
        self.intercept_ = y_off
        self.feature_names_ = list(feature_names) if feature_names else None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
