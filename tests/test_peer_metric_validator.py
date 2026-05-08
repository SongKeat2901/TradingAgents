"""Tests for the Phase-7.3 peer-metric validator."""
import json

import pytest

pytestmark = pytest.mark.unit


def _write_peer_data(tmp_path):
    """Real MARA peer ratios + peers from the 2026-05-06 cadence."""
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-06",
        "_unavailable": [],
        "RIOT": {
            "latest_quarter_capex_to_revenue": 78.7,
            "latest_quarter_op_margin": -72.6,
            "ttm_pe": None,
            "forward_pe": None,
            "net_debt": 636_497_000,
            "ttm_ebitda": -326_712_000,
            "nd_ebitda": None,
        },
        "CIFR": {
            "latest_quarter_capex_to_revenue": 385.4,
            "latest_quarter_op_margin": -383.42,
            "ttm_pe": 17.0,
            "forward_pe": 40.95,
            "net_debt": -150_000_000,
            "ttm_ebitda": 197_000_000,
            "nd_ebitda": -0.76,
        },
        "CLSK": {
            "latest_quarter_capex_to_revenue": 20.4,
            "latest_quarter_op_margin": -37.83,
            "ttm_pe": None,
            "forward_pe": 28.5,
            "net_debt": 200_000_000,
            "ttm_ebitda": -229_000_000,
            "nd_ebitda": None,
        },
    }
    peers = {
        "RIOT": {"ticker": "RIOT"},
        "CIFR": {"ticker": "CIFR"},
        "CLSK": {"ticker": "CLSK"},
    }
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")
    return raw


def test_passes_when_metrics_match_peer_ratios(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = (
        "RIOT capex/revenue 78.7%; CIFR forward P/E 40.95x; CLSK op margin "
        "−37.8% (per raw/peer_ratios.json)."
    )
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_catches_clsk_op_margin_sign_flip_fabrication(tmp_path):
    """The exact MARA failure: claimed `CLSK op margin ~5%` when actual
    is −37.83%. Sign + magnitude both wrong."""
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = "CLSK op margin ~5% per raw/peers.json"
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert len(v) == 1
    assert v[0].severity == "MATERIAL"
    assert v[0].type == "wrong_peer_metric"
    assert v[0].ticker == "CLSK"
    assert v[0].claimed_value.strip() == "~5%" or "5" in v[0].claimed_value
    assert "-37.83" in str(v[0].actual_value)


def test_catches_riot_ev_ebitda_fabricated_attribution(tmp_path):
    """`RIOT EV/EBITDA ~12×` is NOT a peer_ratios.json column; if attributed
    to peer_ratios.json it's a fabricated source."""
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = (
        "Per raw/peers.json: RIOT EV/EBITDA ~12×, ND/EBITDA <1×; CIFR "
        "EV/EBITDA ~9×, P/S ~6×, ND/EBITDA ~1.5×."
    )
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    # Should flag at least the RIOT EV/EBITDA fabrication (EV/EBITDA isn't in peer_ratios)
    fabrications = [x for x in v if x.type == "fabricated_metric_attribution"]
    assert any(x.ticker == "RIOT" and "ev/ebitda" in x.metric.lower() for x in fabrications)


def test_skips_when_peers_json_missing(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    text = "RIOT op margin 5%"
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_skips_unavailable_peers(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-06",
        "_unavailable": ["RIOT"],
        "RIOT": {"unavailable": True, "reason": "missing rows"},
    }
    peers = {"RIOT": {"ticker": "RIOT"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")
    text = "RIOT op margin 5%"
    # Should skip — peer data marked unavailable, can't verify
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_render_violations_text_pass():
    from tradingagents.validators.peer_metric_validator import render_peer_violations_text
    out = render_peer_violations_text([])
    assert "PEER VALIDATION PASS" in out


def test_value_parser_handles_common_formats():
    from tradingagents.validators.peer_metric_validator import _parse_value
    assert _parse_value("~12×")[0] == 12.0
    assert _parse_value("~12×")[1] == "ratio"
    assert _parse_value("38.5%")[0] == 38.5
    assert _parse_value("38.5%")[1] == "pct"
    assert _parse_value("$1.5B")[0] == 1.5
    assert _parse_value("$1.5B")[1] == "billions"
    assert _parse_value("-37.83%")[0] == -37.83
    # Unparseable
    assert _parse_value("low single-digit")[1] == "raw"
