import pytest
from tradingagents.macro import config

pytestmark = pytest.mark.unit


def test_factor_order_is_canonical():
    assert config.FACTORS == ["d_10y", "d_dxy", "d_hy_spread", "oil_ret", "mkt", "growth_value"]


def test_every_indicator_has_a_known_pillar():
    for spec in config.INDICATORS:
        assert spec.pillar in config.PILLARS, f"{spec.name} has unknown pillar {spec.pillar}"


def test_pillar_weights_cover_all_pillars():
    assert set(config.PILLAR_WEIGHTS) == set(config.PILLARS)


def test_factor_regime_map_uses_known_factors_and_pillars():
    for factor, weights in config.FACTOR_REGIME_MAP.items():
        assert factor in config.FACTORS
        for pillar in weights:
            assert pillar in config.PILLARS


def test_gate_thresholds_present():
    assert config.GATE_RED_BREADTH == 4
    assert 0.0 < config.EV_TILT_CAP <= 1.0
