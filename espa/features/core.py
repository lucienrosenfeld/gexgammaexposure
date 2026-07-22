"""Stage 1 observable core (Section 10).

    x_Q = [C_early, C_late, I_55, dI, B_live, X, V]

with the frozen sign-constraint register

    w_{C_early} >= 0, w_{I} >= 0, w_{B} <= 0

and C_late, dI, X, V unconstrained. The register was frozen before
estimation; it is data, not code to be edited when a coefficient
misbehaves — a coefficient the data wants to flip against a strong prior
is shrunk to zero instead of admitted with the wrong sign.

Imbalance has two alternative specifications, never combined, because
[I_55, dI, I_last] together are severely collinear:

    I1 (primary): [I_55, dI]           with dI = I_55 - I_50
    I2:           [I_last, I_last - I_55], receipt-stamped

Counted Stage 1 add-ons (absorption interaction, full-day return) must
*displace* an existing feature under the parameter budget; the builder
enforces displacement by requiring the caller to name the feature they
replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from espa.config import DEFAULT_CONSTANTS, RunConfig, SpecConstants
from espa.standardize import robust_z

#: Frozen sign-constraint register: +1 -> coefficient >= 0, -1 -> <= 0,
#: 0 -> unconstrained. Keyed by canonical feature name.
STAGE1_SIGN_CONSTRAINTS: dict[str, int] = {
    "C_early": +1,
    "C_late": 0,
    "I_level": +1,  # I_55 under I1, I_last under I2 — the level carries the prior
    "I_change": 0,
    "B_live": -1,
    "X": 0,
    "V": 0,
    # counted alternatives (each must displace a feature):
    "absorption": 0,
    "r_full_day": 0,
}

STAGE1_FEATURES: tuple[str, ...] = (
    "C_early",
    "C_late",
    "I_level",
    "I_change",
    "B_live",
    "X",
    "V",
)


@dataclass
class Stage1Inputs:
    """Raw daily series the Stage 1 builder consumes. All decision-time safe.

    Index: trading day. Every series is the *raw* observable; z-scoring
    happens inside the builder against strictly lagged history.
    """

    #: r_ES 15:30 -> 15:55.
    c_early: pd.Series
    #: r_ES 15:55 -> 16:00.
    c_late: pd.Series
    #: SPX-weight-aggregated imbalance at the 15:55 snapshot, normalised
    #: by 20-day median closing-auction volume.
    i_55: pd.Series
    #: Same at the 15:50 snapshot.
    i_50: pd.Series
    #: Cross-asset composite X (already z-scored internally).
    x_composite: pd.Series
    #: B_live, already dispersion-normalised (basis.live_basis).
    b_live: pd.Series
    #: ES 16:00 price minus full-day VWAP, in ATR units.
    vwap_deviation: pd.Series
    #: Spec I2 only: last receipt-stamped imbalance before decision time.
    i_last: pd.Series | None = None
    #: Counted alternative: full-day 9:30 -> 15:30 return.
    r_full_day: pd.Series | None = None
    #: Spec I1beta only (R-IMB-BETA-01): beta-weighted aggregations of the
    #: same snapshot panels, produced by espa.features.imbalance_beta.
    i_beta_55: pd.Series | None = None
    i_beta_50: pd.Series | None = None


def build_stage1_features(
    inputs: Stage1Inputs,
    config: RunConfig,
    constants: SpecConstants = DEFAULT_CONSTANTS,
    displace: dict[str, str] = None,
    i1beta_admitted: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assemble the seven Stage 1 features and their sign constraints.

    Returns ``(features, constraints)`` where constraints maps each
    output column to its frozen sign (+1 / -1 / 0).

    ``displace`` maps an added counted feature to the existing feature it
    replaces (e.g. ``{"absorption": "V"}``); adding without displacing
    violates the parameter budget and raises.

    ``i1beta_admitted``: attestation that the registry holds an ADMIT
    record for R-IMB-BETA-01 (redundancy screen passed). Running spec
    I1beta without it is impossible, not merely discouraged — pass
    ``registry.is_admitted("imbalance=I1beta")`` here.
    """
    displace = displace or {}
    z = lambda s: robust_z(s, constants)  # noqa: E731

    cols: dict[str, pd.Series] = {}
    cols["C_early"] = z(inputs.c_early)
    cols["C_late"] = z(inputs.c_late)

    if config.imbalance_spec == "I1":
        cols["I_level"] = z(inputs.i_55)
        cols["I_change"] = z(inputs.i_55 - inputs.i_50)
    elif config.imbalance_spec == "I2":
        if inputs.i_last is None:
            raise ValueError("imbalance spec I2 requires the receipt-stamped i_last series")
        cols["I_level"] = z(inputs.i_last)
        cols["I_change"] = z(inputs.i_last - inputs.i_55)
    elif config.imbalance_spec == "I1beta":
        if not i1beta_admitted:
            raise ValueError(
                "imbalance spec I1beta requires a recorded ADMIT from the "
                "R-IMB-BETA-01 redundancy screen (pass i1beta_admitted="
                "registry.is_admitted('imbalance=I1beta'))"
            )
        if inputs.i_beta_55 is None or inputs.i_beta_50 is None:
            raise ValueError("imbalance spec I1beta requires i_beta_55 and i_beta_50")
        cols["I_level"] = z(inputs.i_beta_55)
        cols["I_change"] = z(inputs.i_beta_55 - inputs.i_beta_50)
    else:
        raise ValueError(f"unknown imbalance spec: {config.imbalance_spec}")

    cols["B_live"] = inputs.b_live  # already normalised by its own lagged dispersion
    cols["X"] = inputs.x_composite
    cols["V"] = z(inputs.vwap_deviation)

    if config.use_absorption_interaction:
        _apply_displacement(cols, displace, "absorption")
        cols["absorption"] = -(cols["C_late"] * z(inputs.i_55 - inputs.i_50))
    if config.use_full_day_return_stage1:
        if inputs.r_full_day is None:
            raise ValueError("full-day-return alternative requires r_full_day series")
        _apply_displacement(cols, displace, "r_full_day")
        cols["r_full_day"] = z(inputs.r_full_day)

    features = pd.DataFrame(cols)
    constraints = {c: STAGE1_SIGN_CONSTRAINTS[c] for c in features.columns}
    return features, constraints


def _apply_displacement(
    cols: dict[str, pd.Series], displace: dict[str, str], new_feature: str
) -> None:
    victim = displace.get(new_feature)
    if victim is None:
        raise ValueError(
            f"counted feature '{new_feature}' must displace an existing feature "
            "(parameter budget, Section 10); pass displace={'"
            + new_feature
            + "': '<feature>'}"
        )
    if victim not in cols:
        raise ValueError(f"cannot displace unknown feature '{victim}'")
    del cols[victim]
