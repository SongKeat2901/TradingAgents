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


def test_week_defaults_to_iso_week_when_omitted(tmp_path, capsys):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "empty_final"),
                  "--no-write"])
    out = json.loads(capsys.readouterr().out)
    # No --week and no existing folders -> the ISO calendar week we publish in.
    assert out["week"] == cf._iso_week_label()
    assert out["week_required"] is False


def test_iso_week_label_known_dates():
    import datetime
    assert cf._iso_week_label(datetime.date(2026, 6, 29)) == "wk 27 2026"  # Mon, ISO wk27
    assert cf._iso_week_label(datetime.date(2026, 6, 26)) == "wk 26 2026"  # Fri, still wk26
    assert cf._iso_week_label(datetime.date(2026, 6, 24)) == "wk 26 2026"


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


def test_publish_success_promotes_and_refreshes_summary(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    monkeypatch.setattr(cf.pub, "publish_pdf", lambda *a, **k: "NEW_ID")
    def fake_promote(run_dir, final_base, week):
        return Path(final_base) / week / Path(run_dir).name
    monkeypatch.setattr(cf.pub, "promote", fake_promote)
    refresh = {}
    monkeypatch.setattr(cf.pub, "refresh_summary_sheet",
                        lambda **k: refresh.update(k) or True)
    plan = {}
    monkeypatch.setattr(cf.pub, "refresh_trading_plan",
                        lambda **k: plan.update(k) or True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert out["tickers"][0]["published"] is True
    assert "wk 24 2026" in out["tickers"][0]["promoted_to"]
    # both sheets rendered -> current, nothing left pending
    assert out["summary_updated"] is True
    assert out["trading_plan_updated"] is True
    assert out["summary_update_pending"] is False
    assert refresh["script"].endswith("update_summary.py")
    assert plan["script"].endswith("refresh_trading_plan.sh")


def test_revalidate_when_decision_newer(tmp_path, monkeypatch):
    import time
    from cli import cadence_followup as cf
    pre = tmp_path / "preaudit"
    rd = _mk_run(pre, "2026-06-05-AAA", {"blocking_violations": 0})
    # make decision.md newer than validation_report.json
    vr = rd / "validation_report.json"  # noqa: F841 (referenced via mtime only)
    time.sleep(0.01)
    (rd / "decision.md").write_text((rd / "decision.md").read_text() + "\nedited\n")
    called = {}
    monkeypatch.setattr("cli.research_validation.run_phase_7_validators",
                        lambda d: called.setdefault("ran", True) or {"total_violations": 0, "blocking_violations": 0})
    monkeypatch.setattr("cli.research_validation.write_validation_report",
                        lambda d, r: None)
    cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert called.get("ran") is True


def test_no_revalidate_flag_skips(tmp_path, monkeypatch):
    from cli import cadence_followup as cf
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", {"blocking_violations": 0})
    called = {}
    monkeypatch.setattr("cli.research_validation.run_phase_7_validators",
                        lambda d: called.setdefault("ran", True) or {})
    cf.main(["--preaudit-base", str(pre), "--no-write", "--no-revalidate"])
    assert "ran" not in called


def test_summary_refresh_failure_keeps_pending(tmp_path, capsys, monkeypatch):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP, with_pdf=True)
    monkeypatch.setattr(cf.pub, "gog_token_valid", lambda *a, **k: True)
    monkeypatch.setattr(cf.pub, "publish_pdf", lambda *a, **k: "ID")
    monkeypatch.setattr(cf.pub, "promote",
                        lambda run_dir, fb, wk: Path(fb) / wk / Path(run_dir).name)
    monkeypatch.setattr(cf.pub, "refresh_summary_sheet", lambda **k: False)
    monkeypatch.setattr(cf.pub, "refresh_trading_plan", lambda **k: True)
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(tmp_path / "final"),
                  "--week", "wk 24 2026"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["tickers"][0]["published"] is True
    # one sheet's render failed -> stays flagged so the update isn't silently lost
    assert out["summary_updated"] is False
    assert out["trading_plan_updated"] is True
    assert out["summary_update_pending"] is True


def test_week_defaults_to_iso_ignoring_existing_folders(tmp_path, capsys):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", BUYBACK_FP)
    final = tmp_path / "final"
    (final / "wk 22 2026").mkdir(parents=True)
    (final / "wk 99 2099").mkdir(parents=True)   # a higher sequential — must be ignored
    rc = cf.main(["--preaudit-base", str(pre), "--final-base", str(final), "--no-write"])
    out = json.loads(capsys.readouterr().out)
    # ISO week of today, NOT the highest existing folder (sequential is gone).
    assert out["week"] == cf._iso_week_label()
    assert out["week"] != "wk 99 2099"
