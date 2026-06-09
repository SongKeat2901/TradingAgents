import json
import pytest
from pathlib import Path
from cli import cadence_followup as cf

pytestmark = pytest.mark.unit


def _mk_run(pre, name, validation):
    rd = pre / name
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text("- **Reference price:** $10.00 (yfinance close)\n")
    (rd / "validation_report.json").write_text(json.dumps(validation))
    for f in ("intrinsic_value", "peer_ratios", "financials"):
        (rd / "raw" / f"{f}.json").write_text("{}")
    return rd


def test_no_write_emits_contract(tmp_path, capsys):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", {"blocking_violations": 1,
        "phase_7_5_net_debt": {"violations": [{"severity": "MATERIAL",
            "type": "definitional_drift", "claimed_dollars": 163000000000.0,
            "match_text": "$163B authorized buyback"}]}})
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trade_date"] == "2026-06-05"
    assert out["tickers"][0]["ticker"] == "AAA"
    assert out["tickers"][0]["grade"] == "A"
    assert out["tickers"][0]["published"] is False


def test_empty_base_reports_no_batch(tmp_path, capsys):
    pre = tmp_path / "preaudit"; pre.mkdir()
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trade_date"] is None
    assert out["tickers"] == []
