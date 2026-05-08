"""Tests for Phase 7.4: cli/research_validation.py runner."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _setup_clean_run(tmp_path):
    """Build a minimal valid run dir with prices.json + a clean decision."""
    rd = tmp_path / "run"
    raw = rd / "raw"
    raw.mkdir(parents=True)
    (raw / "prices.json").write_text(json.dumps({
        "ohlcv": (
            "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
            "2026-05-06,195.78,198.5,193.25,197.96,7764900,0.0,0.0\n"
            "2026-05-07,196.24,198.15,190.32,192.96,8641932,0.0,0.0\n"
        )
    }), encoding="utf-8")
    (raw / "peer_ratios.json").write_text(json.dumps({
        "trade_date": "2026-05-08",
        "_unavailable": [],
    }), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps({}), encoding="utf-8")
    (rd / "decision.md").write_text(
        "# COIN — 2026-05-08\n\n"
        "Spot reference $192.96 (yfinance close on 2026-05-07).\n",
        encoding="utf-8",
    )
    return rd


def _setup_run_with_fabrication(tmp_path):
    """Run dir with the COIN $206.50 fabrication baked in."""
    rd = _setup_clean_run(tmp_path)
    (rd / "decision.md").write_text(
        "# COIN — 2026-05-08\n\n"
        "The May 8 session subsequently closed at $206.50 per prices.json\n"
        "On 2026-05-08 close $206.50 confirmed the post-print bounce.\n",
        encoding="utf-8",
    )
    return rd


def test_run_phase_7_validators_clean_run_passes(tmp_path):
    from cli.research_validation import run_phase_7_validators

    rd = _setup_clean_run(tmp_path)
    results = run_phase_7_validators(rd)

    assert results["total_violations"] == 0
    assert results["phase_7_1_price_date"]["violations"] == []
    assert results["phase_7_2_quote_attribution"]["violations"] == []
    assert results["phase_7_3_peer_metric"]["violations"] == []
    assert "decision.md" in results["files_scanned"]


def test_run_phase_7_validators_catches_fabrication(tmp_path):
    """End-to-end: the runner catches the COIN-style $206.50 forward-
    projection through Phase 7.1."""
    from cli.research_validation import run_phase_7_validators

    rd = _setup_run_with_fabrication(tmp_path)
    results = run_phase_7_validators(rd)

    assert results["total_violations"] > 0
    pd_violations = results["phase_7_1_price_date"]["violations"]
    assert any(
        v.get("type") == "fabricated_future_close"
        and v.get("claimed_date") == "2026-05-08"
        and v.get("claimed_price") == 206.50
        for v in pd_violations
    ), f"expected COIN fabrication violation; got {pd_violations}"


def test_write_validation_report_persists_json(tmp_path):
    from cli.research_validation import (
        run_phase_7_validators,
        write_validation_report,
    )

    rd = _setup_run_with_fabrication(tmp_path)
    results = run_phase_7_validators(rd)
    report_path = write_validation_report(rd, results)

    assert report_path.exists()
    assert report_path.name == "validation_report.json"
    loaded = json.loads(report_path.read_text())
    assert loaded["total_violations"] == results["total_violations"]
    assert "phase_7_1_price_date" in loaded


def test_format_validation_summary_pass(tmp_path):
    from cli.research_validation import (
        run_phase_7_validators,
        format_validation_summary,
    )

    rd = _setup_clean_run(tmp_path)
    results = run_phase_7_validators(rd)
    summary = format_validation_summary(results)
    assert "PASS" in summary
    assert "0 violation" in summary


def test_format_validation_summary_fail_includes_phase_breakdown(tmp_path):
    from cli.research_validation import (
        run_phase_7_validators,
        format_validation_summary,
    )

    rd = _setup_run_with_fabrication(tmp_path)
    results = run_phase_7_validators(rd)
    summary = format_validation_summary(results)
    assert "FAIL" in summary
    assert "price/date" in summary
    assert "quote" in summary
    assert "peer" in summary


def test_runner_is_fail_soft_on_missing_run_dir(tmp_path):
    """A non-existent run dir shouldn't crash — returns 0 violations
    because there's nothing to scan."""
    from cli.research_validation import run_phase_7_validators

    nonexistent = tmp_path / "doesnt_exist"
    # Don't create it
    results = run_phase_7_validators(nonexistent)
    assert results["total_violations"] == 0
    assert results["files_scanned"] == []


def test_runner_handles_run_dir_with_no_prices_json(tmp_path):
    """If raw/prices.json is missing, Phase 7.1 emits an informational
    no_prices_data violation when there are claims to validate. Otherwise
    it returns clean."""
    from cli.research_validation import run_phase_7_validators

    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "decision.md").write_text(
        "On 2026-05-08 closed at $206.50",
        encoding="utf-8",
    )
    results = run_phase_7_validators(rd)
    # Should produce the no_prices_data violation
    pd_violations = results["phase_7_1_price_date"]["violations"]
    assert any(v.get("type") == "no_prices_data" for v in pd_violations)
