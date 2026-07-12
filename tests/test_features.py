"""Feature blocks: Stage 1 assembly rules, options block, PGI residualisation."""

import numpy as np
import pandas as pd
import pytest

from espa.config import RunConfig
from espa.features.core import STAGE1_FEATURES, Stage1Inputs, build_stage1_features
from espa.features.options import OptionContract, contract_gamma_weight, raw_gamma_density
from espa.features.pgi import path_gamma_integral, residualise_pgi
from espa.features.pinning import pinning_feature
from espa.features.cross_asset import cross_asset_composite


def _inputs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    s = lambda scale=1.0: pd.Series(rng.normal(0, scale, n), index=idx)  # noqa: E731
    return Stage1Inputs(
        c_early=s(0.002), c_late=s(0.001), i_55=s(), i_50=s(),
        x_composite=s(), b_live=s(), vwap_deviation=s(),
        i_last=s(), r_full_day=s(0.01),
    )


def test_stage1_has_seven_features_and_frozen_constraints():
    feats, cons = build_stage1_features(_inputs(), RunConfig(name="t"))
    assert tuple(feats.columns) == STAGE1_FEATURES
    assert cons["C_early"] == +1
    assert cons["I_level"] == +1
    assert cons["B_live"] == -1
    assert cons["C_late"] == 0 and cons["I_change"] == 0
    assert cons["X"] == 0 and cons["V"] == 0


def test_i2_spec_swaps_never_augments():
    feats, _ = build_stage1_features(_inputs(), RunConfig(name="t", imbalance_spec="I2"))
    assert len(feats.columns) == 7  # never [I_55, dI, I_last] together


def test_counted_feature_must_displace():
    cfg = RunConfig(name="t", use_absorption_interaction=True)
    with pytest.raises(ValueError, match="displace"):
        build_stage1_features(_inputs(), cfg)
    feats, _ = build_stage1_features(_inputs(), cfg, displace={"absorption": "V"})
    assert "absorption" in feats.columns and "V" not in feats.columns
    assert len(feats.columns) == 7  # budget preserved


def _contracts(spot):
    return [
        OptionContract(strike=k, expiry_years=1 / 365, gamma=0.002,
                       open_interest=1000, multiplier=100)
        for k in (spot - 50, spot, spot + 50)
    ]


def test_gamma_kernel_decays_with_distance_and_expiry():
    spot = 4000.0
    near = OptionContract(4000, 1 / 365, 0.002, 1000, 100)
    far = OptionContract(4200, 1 / 365, 0.002, 1000, 100)
    late = OptionContract(4000, 30 / 365, 0.002, 1000, 100)
    w = lambda c: abs(contract_gamma_weight(c, spot, 0.0075, 3 / 365))  # noqa: E731
    assert w(near) > w(far)
    assert w(near) > w(late)


def test_gamma_density_is_unsigned():
    spot = 4000.0
    pos = _contracts(spot)
    neg = [OptionContract(c.strike, c.expiry_years, -c.gamma, c.open_interest, c.multiplier)
           for c in pos]
    assert raw_gamma_density(pos, spot) == pytest.approx(raw_gamma_density(neg, spot))


def test_pinning_sign_and_envelope():
    spot, atr = 4000.0, 30.0
    c = _contracts(spot)
    # spot above the dominant strike -> D > 0 -> P < 0 (pull back toward K*)
    p_above = pinning_feature(c, spot + 20, atr)
    p_below = pinning_feature(c, spot - 20, atr)
    assert np.sign(p_above) != np.sign(p_below)
    # envelope: a very distant strike cannot generate a large signal
    p_far = pinning_feature(c, spot + 500, atr)
    assert abs(p_far) < abs(p_above)


def test_pgi_zero_on_flat_path():
    c = _contracts(4000.0)
    flat = pd.Series(np.full(78, 4000.0))
    assert path_gamma_integral(flat, c, rho=0.0) == pytest.approx(0.0)


def test_pgi_residualisation_removes_momentum():
    rng = np.random.default_rng(1)
    n = 300
    idx = pd.RangeIndex(n)
    ce = pd.Series(rng.normal(0, 1, n), index=idx)
    cl = pd.Series(rng.normal(0, 1, n), index=idx)
    rd = pd.Series(rng.normal(0, 1, n), index=idx)
    pgi = 2.0 * ce - 1.5 * rd + pd.Series(rng.normal(0, 0.1, n), index=idx)
    res = residualise_pgi(pgi, ce, cl, rd)
    perp = res.transform(pgi, ce, cl, rd)
    assert abs(np.corrcoef(perp, ce)[0, 1]) < 0.05
    assert abs(np.corrcoef(perp, rd)[0, 1]) < 0.05


def test_cross_asset_composite_bounded_weights():
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.RangeIndex(n)
    rets = pd.DataFrame(
        {c: rng.normal(0, 0.001, n) for c in ("es", "nq", "rty", "ym", "dxy", "ty")},
        index=idx,
    )
    x = cross_asset_composite(rets, "fixed")
    assert x.dropna().abs().max() < 20  # sane scale
    x2 = cross_asset_composite(rets, "no_ty")
    assert x2.notna().sum() > 0
    with pytest.raises(ValueError):
        cross_asset_composite(rets, "nonsense")
