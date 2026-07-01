import pytest
from tradingagents.graph.propagation import Propagator

pytestmark = pytest.mark.unit


def test_create_initial_state_default_reuse_false():
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x")
    assert st["reuse_raw"] is False


def test_create_initial_state_reuse_true():
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x", reuse_raw=True)
    assert st["reuse_raw"] is True


def test_build_config_sets_reuse_raw():
    import argparse
    from cli.research import _build_config
    ns = argparse.Namespace()
    # minimal namespace mirroring the parser's fields used by _build_config
    for k, v in {
        "deep": "claude-opus-4-8", "quick": "claude-sonnet-4-6", "debate_rounds": 1,
        "risk_rounds": 1, "token_source": "auto", "openclaw_profile_path": None,
        "openclaw_profile_name": None, "pacing_seconds": 0, "max_tokens": 8192,
        "deep_cooldown_seconds": 0, "output_dir": "/tmp/x", "reuse_raw": True,
    }.items():
        setattr(ns, k, v)
    cfg = _build_config(ns)
    assert cfg["reuse_raw"] is True
