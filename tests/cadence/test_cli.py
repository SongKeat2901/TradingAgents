import json
import pytest
from pathlib import Path
from cli import cadence_followup as cf

pytestmark = pytest.mark.unit

BUYBACK_FP = {"blocking_violations": 1, "phase_7_5_net_debt": {"violations": [
    {"severity": "MATERIAL", "type": "definitional_drift",
     "claimed_dollars": 163000000000.0, "match_text": "$163B authorized buyback"}]}}
UNKNOWN_FLAG = {"blocking_violations": 1, "phase_7_5_net_debt": {"violations": [
    {"severity": "MATERIAL", "type": "definitional_drift",
     "claimed_dollars": 50000000000.0, "match_text": "Net debt is $50B"}]}}


def _mk_run(pre, name, validation, with_pdf=False):
    rd = pre / name
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text("- **Reference price:** $10.00 (yfinance close)\n")
    (rd / "validation_report.json").write_text(json.dumps(validation))
    for f in ("intrinsic_value", "peer_ratios", "financials"):
        (rd / "raw" / f"{f}.json").write_text("{}")
    if with_pdf:
        parts = name.split("-")
        date = "-".join(parts[:3]); ticker = parts[3]
        (rd / ("research-%s-%s.pdf" % (date, ticker))).write_text("%PDF")
    return rd


def test_no_write_emits_contract(tmp_path, capsys):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP)
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trade_date"] == "2026-06-05"
    assert out["token_valid"] is None
    assert out["tickers"][0]["ticker"] == "AAA"
    assert out["tickers"][0]["grade"] == "A"
    assert out["tickers"][0]["published"] is False


def test_empty_base_reports_no_batch(tmp_path, capsys):
    pre = tmp_path / "preaudit"; pre.mkdir()
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["trade_date"] is None and out["tickers"] == []


def test_writes_held_on_invalid_token(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: False)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["token_valid"] is False
    assert out["writes_held"] is True
    assert out["reauth_url"]
    assert out["tickers"][0]["grade"] == "A"
    assert out["tickers"][0]["published"] is False


def test_week_required_when_missing(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "empty_final")])
    out = json.loads(capsys.readouterr().out)
    assert out["week"] is None
    assert out["week_required"] is True
    assert out["writes_held"] is True
    assert out["tickers"][0]["published"] is False


def test_hold_ticker_not_published(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-BAD", UNKNOWN_FLAG, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    published = []
    monkeypatch.setattr(cf.pub, "publish_pdf", lambda *a, **k: published.append(1) or "ID")
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert out["tickers"][0]["grade"] == "HOLD"
    assert out["tickers"][0]["needs_adjudication"]
    assert out["tickers"][0]["published"] is False
    assert published == []


def test_pdf_missing_degrades(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=False)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["tickers"][0]["published"] is False
    assert out["tickers"][0]["error"] == "PDF missing"


def test_publish_success_promotes_and_refreshes(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    monkeypatch.setattr(cf.pub, "publish_pdf", lambda *a, **k: "NEW_ID")
    def fake_promote(run_dir, final_base, week):
        return Path(final_base) / week / Path(run_dir).name
    monkeypatch.setattr(cf.pub, "promote", fake_promote)
    refreshed = []
    monkeypatch.setattr(cf.pub, "refresh_summary_sheet",
                        lambda **k: refreshed.append(1) or True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert out["tickers"][0]["published"] is True
    assert "wk 24 2026" in out["tickers"][0]["promoted_to"]
    assert refreshed == [1]


def test_week_inferred_from_highest_existing(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP)
    final = tmp_path / "final"
    (final / "wk 22 2026").mkdir(parents=True)
    (final / "wk 23 2026").mkdir(parents=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(final), "--no-write"])
    out = json.loads(capsys.readouterr().out)
    assert out["week"] == "wk 23 2026"
