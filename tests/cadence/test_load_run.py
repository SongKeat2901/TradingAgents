import json
import pytest
from tradingagents.cadence.batch import load_run

pytestmark = pytest.mark.unit


def _build(tmp_path):
    rd = tmp_path / "2026-06-05-AAPL"
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text(
        "intro\n- **Reference price:** $177.00 (yfinance close of 2026-06-05)\n")
    (rd / "validation_report.json").write_text(json.dumps(
        {"total_violations": 1, "blocking_violations": 1,
         "phase_7_5_net_debt": {"violations": [{"severity": "MATERIAL"}]}}))
    (rd / "raw" / "intrinsic_value.json").write_text(
        json.dumps({"ticker": "AAPL", "inputs": {"net_debt": 39139000000.0}}))
    (rd / "raw" / "peer_ratios.json").write_text(json.dumps({"COHR": {"ttm_pe": 178.67}}))
    (rd / "raw" / "financials.json").write_text(json.dumps({"ticker": "AAPL"}))
    return rd


def test_load_run_populates_fields(tmp_path):
    rd = _build(tmp_path)
    run = load_run(rd)
    assert run.ticker == "AAPL"
    assert run.trade_date == "2026-06-05"
    assert run.reference_price == 177.00
    assert run.validation["blocking_violations"] == 1
    assert run.intrinsic_value["inputs"]["net_debt"] == 39139000000.0
    assert "COHR" in run.peer_ratios


def test_load_run_missing_raw_is_empty_dict(tmp_path):
    rd = tmp_path / "2026-06-05-XYZ"
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text("no ref price here\n")
    (rd / "validation_report.json").write_text("{}")
    run = load_run(rd)
    assert run.peer_ratios == {}
    assert run.reference_price is None
