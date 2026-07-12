"""Concentration-weighted pinning P_t (Section 11).

With K* the strike maximising near-money unsigned gamma weight,
D = (S - K*)/ATR, and HHI the strike-concentration Herfindahl over
gamma weights:

    P = -HHI * D * exp(-|D|)

The exponential envelope bounds the feature so a distant strike cannot
generate a large signal; the concentration weight silences the putative
magnet when gamma is diffuse across strikes, at the cost of no
additional fitted coefficient. Pinning direction depends on positioning,
so the Stage 2 coefficient on P is unconstrained.
"""

from __future__ import annotations

import numpy as np

from espa.config import DEFAULT_CONSTANTS
from espa.features.options import OptionContract, contract_gamma_weight


def strike_gamma_weights(
    contracts: list[OptionContract],
    spot: float,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> dict[float, float]:
    """Unsigned gamma weight aggregated per strike."""
    weights: dict[float, float] = {}
    for c in contracts:
        w = abs(contract_gamma_weight(c, spot, lambda_s, lambda_t))
        weights[c.strike] = weights.get(c.strike, 0.0) + w
    return weights


def pinning_feature(
    contracts: list[OptionContract],
    spot: float,
    atr: float,
    lambda_s: float = DEFAULT_CONSTANTS.lambda_s,
    lambda_t: float = DEFAULT_CONSTANTS.lambda_t,
) -> float:
    """P_t for one day; NaN when there is no usable strike mass."""
    weights = strike_gamma_weights(contracts, spot, lambda_s, lambda_t)
    total = sum(weights.values())
    if total <= 0 or not np.isfinite(atr) or atr <= 0:
        return np.nan
    k_star = max(weights, key=weights.get)
    hhi = sum((w / total) ** 2 for w in weights.values())
    d = (spot - k_star) / atr
    return float(-hhi * d * np.exp(-abs(d)))
