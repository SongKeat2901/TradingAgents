import pytest
pytestmark = pytest.mark.unit


def test_log_returns_and_simulation_is_deterministic():
    from tradingagents.agents.utils.forward_distribution import (
        daily_log_returns, simulate_paths,
    )
    closes = [100.0 * (1.001 ** i) for i in range(400)]
    rets = daily_log_returns(closes)
    assert len(rets) == 399
    paths_a = simulate_paths(spot=closes[-1], returns=rets, horizon=252,
                             n_paths=500, block=10, seed=42)
    paths_b = simulate_paths(spot=closes[-1], returns=rets, horizon=252,
                             n_paths=500, block=10, seed=42)
    assert paths_a == paths_b
    assert len(paths_a) == 500
    assert all(len(p) == 252 for p in paths_a)


def test_first_barrier_touch_partitions_to_100pct():
    from tradingagents.agents.utils.forward_distribution import first_barrier_probabilities
    up = [[100 + i * (20/252) for i in range(1, 253)] for _ in range(5)]
    down = [[100 - i * (20/252) for i in range(1, 253)] for _ in range(5)]
    probs = first_barrier_probabilities(up + down, bull=115.0, bear=85.0)
    assert abs(probs["bull"] + probs["base"] + probs["bear"] - 1.0) < 1e-9
    assert probs["bull"] == pytest.approx(0.5, abs=0.01)
    assert probs["bear"] == pytest.approx(0.5, abs=0.01)
    assert probs["base"] == pytest.approx(0.0, abs=0.01)


def test_compute_forward_probabilities_picks_levels_and_sums_100():
    from tradingagents.agents.utils.forward_distribution import (
        compute_forward_probabilities, format_forward_block,
    )
    closes = [100.0 + (i % 7) for i in range(800)]
    vp = {
        "structural_36mo": {"poc": 50.0, "hvn": [120.0, 80.0], "vah": 110.0, "val": 70.0},
        "tactical_6mo":    {"poc": 103.0, "hvn": [108.0, 99.0], "vah": 106.0, "val": 101.0},
    }
    out = compute_forward_probabilities("XYZ", "2026-05-28", spot=103.0,
                                        closes=closes, volume_profile=vp,
                                        n_paths=500)
    s = out["scenarios"]
    # First-barrier partition sums to 1.0
    assert abs(s["bull"]["probability"] + s["base"]["probability"]
               + s["bear"]["probability"] - 1.0) < 1e-9
    # Targets must straddle spot
    assert s["bull"]["target"] > 103.0 > s["bear"]["target"]
    # REFINED: Base must be near spot, NOT the structural POC (50.0)
    assert s["base"]["target"] == pytest.approx(103.0, abs=2.0)
    # Bull target should be the nearest HVN above spot (108 from tactical, not 120 from structural)
    assert s["bull"]["target"] == pytest.approx(108.0, abs=0.5)
    block = format_forward_block(out)
    assert "## 12-month scenario probabilities" in block
    assert "Use these targets and probabilities verbatim" in block
