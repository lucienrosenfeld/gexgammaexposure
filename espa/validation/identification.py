"""D-STAGE2-ID-01: Stage 2 identification diagnostics.

Measurement infrastructure, not a trial. Per Stage 2 training fold, on
the exact standardised design used in estimation (A*Qhat_OOF, PGI_perp,
P): the Gram matrix, its condition number kappa, pairwise correlations,
VIFs, the design-variance share of the smallest eigenmode, and the
coefficients with their fold-level standard errors. Training-fold
information only.

Alarm definition (frozen): an identification alarm requires
kappa >= id_kappa_threshold (10, recalibrated for a three-column design:
kappa = 30 needs pairwise rho ~ 0.94 while ridge sign instability begins
near rho ~ 0.8, kappa ~ 9 — the conventional threshold could never fire
in time) AND a material adjacent-fold sign change. A sign change is
material only if |coef| exceeds its fold-level SE on both sides —
without the floor, Section 26's honest prior (beta1 ~ 0) would flip sign
as pure noise and manufacture alarms, illegitimately authorising the
composite-replacement hypothesis.

Persistent non-identification: alarms on >= 30% of comparable
adjacent-fold transitions, minimum three alarms. Consequence: nothing
automatic — it permits one later, separately registered replacement
hypothesis, specified before viewing its predictive results and counted
as a new trial.

This module emits log records only. No code path here may mutate
features, constraints, or model structure — enforced by construction:
it imports nothing from espa.models or espa.features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


@dataclass(frozen=True)
class FoldIDRecord:
    """One training fold's identification measurements."""

    fold: int
    n_obs: int
    gram: pd.DataFrame
    kappa: float
    pairwise_corr: pd.DataFrame
    vif: pd.Series
    smallest_eigenmode_share: float
    coefs: pd.Series
    coef_ses: pd.Series

    def to_log_record(self) -> dict:
        return {
            "fold": self.fold,
            "n_obs": self.n_obs,
            "kappa": self.kappa,
            "gram": self.gram.to_dict(),
            "pairwise_corr": self.pairwise_corr.to_dict(),
            "vif": self.vif.to_dict(),
            "smallest_eigenmode_share": self.smallest_eigenmode_share,
            "coefs": self.coefs.to_dict(),
            "coef_ses": self.coef_ses.to_dict(),
        }


def fold_identification(
    fold: int,
    X_standardised: pd.DataFrame,
    coefs: pd.Series,
    coef_ses: pd.Series,
) -> FoldIDRecord:
    """Measure one fold on the exact standardised design used in estimation."""
    X = X_standardised.dropna()
    n = len(X)
    cols = list(X.columns)
    M = X.to_numpy(dtype=float)
    gram = pd.DataFrame(M.T @ M / max(n, 1), index=cols, columns=cols)
    corr = pd.DataFrame(np.corrcoef(M, rowvar=False), index=cols, columns=cols)
    eig = np.linalg.eigvalsh(corr.to_numpy())
    eig = np.clip(eig, 0.0, None)
    kappa = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    try:
        vif = pd.Series(np.diag(np.linalg.inv(corr.to_numpy())), index=cols)
    except np.linalg.LinAlgError:
        vif = pd.Series(np.inf, index=cols)
    share = float(eig.min() / eig.sum()) if eig.sum() > 0 else 0.0
    return FoldIDRecord(
        fold=fold,
        n_obs=n,
        gram=gram,
        kappa=kappa,
        pairwise_corr=corr,
        vif=vif,
        smallest_eigenmode_share=share,
        coefs=coefs.copy(),
        coef_ses=coef_ses.copy(),
    )


@dataclass
class IdentificationTracker:
    """Accumulates FoldIDRecords chronologically; applies the frozen rules."""

    constants: SpecConstants = DEFAULT_CONSTANTS
    records: list[FoldIDRecord] = field(default_factory=list)

    def add(self, record: FoldIDRecord) -> None:
        self.records.append(record)

    def _material_sign_change(self, prev: FoldIDRecord, cur: FoldIDRecord) -> bool:
        mult = self.constants.id_materiality_se_mult
        for c in cur.coefs.index:
            if c not in prev.coefs.index:
                continue
            a, b = prev.coefs[c], cur.coefs[c]
            sa, sb = prev.coef_ses.get(c, np.nan), cur.coef_ses.get(c, np.nan)
            if not (np.isfinite(a) and np.isfinite(b) and a != 0 and b != 0):
                continue
            if np.sign(a) == np.sign(b):
                continue
            if (
                np.isfinite(sa)
                and np.isfinite(sb)
                and abs(a) > mult * sa
                and abs(b) > mult * sb
            ):
                return True
        return False

    def alarms(self) -> list[int]:
        """Fold indices at which the frozen alarm definition fires."""
        fired = []
        for prev, cur in zip(self.records[:-1], self.records[1:]):
            if cur.kappa >= self.constants.id_kappa_threshold and self._material_sign_change(prev, cur):
                fired.append(cur.fold)
        return fired

    def n_transitions(self) -> int:
        return max(0, len(self.records) - 1)

    def persistent(self) -> bool:
        """>= 30% of adjacent-fold transitions alarmed, minimum three."""
        n = self.n_transitions()
        if n == 0:
            return False
        a = len(self.alarms())
        return (
            a >= self.constants.id_min_alarms
            and a / n >= self.constants.id_persistence_rate
        )

    def to_log_records(self) -> list[dict]:
        return [r.to_log_record() for r in self.records]
