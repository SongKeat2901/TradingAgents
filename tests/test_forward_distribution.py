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
    # spot=100. Tactical HVNs at 110 (10% up) and 94 (6% down) qualify.
    # Structural POC 50 is far below — must NOT be selected as Base.
    # Tactical POC 102 sits between Bear and Bull → selected as Base.
    vp = {
        "structural_36mo": {"poc": 50.0, "hvn": [130.0, 70.0], "vah": 115.0, "val": 60.0},
        "tactical_6mo":    {"poc": 102.0, "hvn": [110.0, 94.0], "vah": 105.0, "val": 99.0},
    }
    out = compute_forward_probabilities("XYZ", "2026-05-28", spot=100.0,
                                        closes=closes, volume_profile=vp,
                                        n_paths=500)
    s = out["scenarios"]
    # First-barrier partition sums to 1.0
    assert abs(s["bull"]["probability"] + s["base"]["probability"]
               + s["bear"]["probability"] - 1.0) < 1e-9
    # Bull target = nearest HVN ≥5% above spot → 110
    assert s["bull"]["target"] == pytest.approx(110.0, abs=0.5)
    # Bear target = nearest HVN ≤5% below spot → 94
    assert s["bear"]["target"] == pytest.approx(94.0, abs=0.5)
    # Base target = tactical POC (102), since 94 < 102 < 110
    assert s["base"]["target"] == pytest.approx(102.0, abs=0.5)
    # Strict ordering — Base must lie between Bear and Bull
    assert s["bear"]["target"] < s["base"]["target"] < s["bull"]["target"]
    block = format_forward_block(out)
    assert "## 12-month scenario probabilities" in block
    assert "Use these targets and probabilities verbatim" in block
    # Terminal-zone semantics: with flat-ish synthetic closes, many paths end
    # near spot → Base should have substantial mass (not ~0% as it was under
    # first-barrier-touch). Assert > 20% to be robust to 500-path variance.
    assert s["base"]["probability"] > 0.20


def test_pick_targets_falls_back_to_spot_when_tactical_poc_outside_interval():
    """GOOGL-style case: tactical POC sits below Bear (e.g. tactical POC $317
    when spot is $383 and Bear is $336). Base must fall back to spot, never
    sit outside the (Bear, Bull) interval."""
    from tradingagents.agents.utils.forward_distribution import _pick_targets
    vp = {
        "structural_36mo": {"poc": 164.0, "hvn": [200.0, 130.0], "vah": 197.0, "val": 114.0},
        "tactical_6mo":    {"poc": 317.0, "hvn": [336.0, 440.0], "vah": 350.0, "val": 300.0},
    }
    bull, base, bear = _pick_targets(spot=383.0, vp=vp)
    # bear is a HVN ≥5% below spot 383 → 336
    # bull is a HVN ≥5% above spot 383 → 440
    # tactical POC 317 < bear 336 → Base must fall back to spot 383
    assert bear < base < bull
    assert base == pytest.approx(383.0, abs=0.5)


def test_terminal_zone_probabilities_classifies_by_end_price():
    from tradingagents.agents.utils.forward_distribution import terminal_zone_probabilities
    # Two paths end above 110, one in the middle, two below 90 → 40/20/40
    paths = [
        [100, 105, 112],
        [100, 108, 115],
        [100, 102, 100],
        [100, 95, 85],
        [100, 90, 80],
    ]
    out = terminal_zone_probabilities(paths, bull=110.0, bear=90.0)
    assert out["bull"] == pytest.approx(0.4)
    assert out["base"] == pytest.approx(0.2)
    assert out["bear"] == pytest.approx(0.4)
    assert abs(sum(out.values()) - 1.0) < 1e-9
