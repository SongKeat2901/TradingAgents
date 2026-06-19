import json
import pytest

from tradingagents.macro import reports

pytestmark = pytest.mark.unit

_DECISION = """\
# Decision

Reference price: **$100.00**

**Rating: BUY**

## 12-Month Scenario Analysis

| Scenario | Prob | Target |
|---|---|---|
| Bull | 30% | $140.00 |
| Base | 50% | $120.00 |
| Bear | 20% | $80.00 |

EV = **$116.00**
"""


def _run_dir(tmp_path, ticker="TEST", date="2026-06-01", body=_DECISION):
    d = tmp_path / f"{date}-{ticker}"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": ticker, "trade_date": date}))
    (d / "decision.md").write_text(body)
    return d


def test_load_base_ev_reads_ev_and_pct(tmp_path):
    be = reports.load_base_ev(_run_dir(tmp_path))
    assert be.ticker == "TEST"
    assert be.reference_price == 100.0
    assert be.ev == 116.0
    assert round(reports.ev_pct(be), 4) == 0.16     # +16%


def test_ev_pct_derives_from_scenarios_when_ev_absent(tmp_path):
    body = _DECISION.replace("EV = **$116.00**", "")
    be = reports.load_base_ev(_run_dir(tmp_path, body=body))
    assert be.ev is None
    # derived = Σ prob·target = .3*140 + .5*120 + .2*80 = 118 → +18%
    assert round(reports.ev_pct(be), 4) == 0.18


def test_returns_none_for_incomplete_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert reports.load_base_ev(d) is None


def test_latest_run_per_ticker_picks_newest_date(tmp_path):
    _run_dir(tmp_path, "AAA", "2026-05-01")
    _run_dir(tmp_path, "AAA", "2026-06-01")
    _run_dir(tmp_path, "BBB", "2026-05-15")
    latest = reports.latest_runs(tmp_path)
    assert latest["AAA"].research_date == "2026-06-01"
    assert set(latest) == {"AAA", "BBB"}


def test_latest_runs_descends_into_week_subdirs(tmp_path):
    wk = tmp_path / "wk 23 2026"
    wk.mkdir()
    _run_dir(wk, "AAPL", "2026-06-01")        # nested under a week bucket
    _run_dir(tmp_path, "MSFT", "2026-05-20")  # directly under base
    latest = reports.latest_runs(tmp_path)
    assert set(latest) == {"AAPL", "MSFT"}
    assert latest["AAPL"].research_date == "2026-06-01"


def test_scenario_ladder_extracts_bull_base_bear(tmp_path):
    be = reports.load_base_ev(_run_dir(tmp_path))
    ladder = reports.scenario_ladder(be)
    assert ladder == {"bull": 140.0, "base": 120.0, "bear": 80.0}


def test_load_base_ev_survives_unreadable_decision(tmp_path):
    import json
    d = tmp_path / "2026-06-01-ERR"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "ERR", "trade_date": "2026-06-01"}))
    (d / "decision.md").mkdir()   # a dir where a file is expected → OSError on read_text
    assert reports.load_base_ev(d) is None


def test_latest_runs_mtime_tiebreak_on_same_date(tmp_path):
    import os
    b1 = tmp_path / "staging"
    b1.mkdir()
    b2 = tmp_path / "rerun"
    b2.mkdir()
    old = _run_dir(b1, "ASX", "2026-06-02",
                   body="Reference price: **$40.00**\n**Rating: UNDERWEIGHT**\nEV = **$42.00**\n")
    new = _run_dir(b2, "ASX", "2026-06-02",
                   body="Reference price: **$40.00**\n**Rating: HOLD**\nEV = **$42.00**\n")
    os.utime(old / "state.json", (1_000_000, 1_000_000))   # older write
    os.utime(new / "state.json", (2_000_000, 2_000_000))   # newer write
    latest = reports.latest_runs([b1, b2])
    assert latest["ASX"].rating == "HOLD"                  # newest-written wins on same date


def test_load_intrinsic_reads_base_fair_value(tmp_path):
    import json as _json
    d = _run_dir(tmp_path)
    raw = d / "raw"
    raw.mkdir()
    (raw / "intrinsic_value.json").write_text(_json.dumps(
        {"profile": "STANDARD", "fair_value": {"bear": 100, "base": 150, "bull": 200},
         "margin_of_safety_pct": 0.2}))
    out = reports.load_intrinsic(d)
    assert out == {"fair_value": 150, "margin_of_safety_pct": 0.2, "profile": "STANDARD"}


def test_load_intrinsic_missing_file_returns_none(tmp_path):
    assert reports.load_intrinsic(_run_dir(tmp_path)) is None   # no raw/intrinsic_value.json


def test_load_intrinsic_null_fair_value(tmp_path):
    import json as _json
    d = _run_dir(tmp_path)
    raw = d / "raw"
    raw.mkdir()
    (raw / "intrinsic_value.json").write_text(_json.dumps(
        {"profile": "UNPROFITABLE", "fair_value": {"bear": None, "base": None, "bull": None},
         "margin_of_safety_pct": None}))
    out = reports.load_intrinsic(d)
    assert out["fair_value"] is None and out["profile"] == "UNPROFITABLE"


def test_load_company_name_reads_name_line(tmp_path):
    import json as _json
    d = _run_dir(tmp_path)
    raw = d / "raw"
    raw.mkdir()
    (raw / "financials.json").write_text(_json.dumps(
        {"fundamentals": "Name: Test Corp\nSector: Technology\nPE: 22.5\n"}))
    assert reports.load_company_name(d) == "Test Corp"


def test_load_company_name_missing_file_returns_empty(tmp_path):
    d = _run_dir(tmp_path)
    # no raw/financials.json present
    assert reports.load_company_name(d) == ""


def test_load_company_name_no_name_line_returns_empty(tmp_path):
    import json as _json
    d = _run_dir(tmp_path)
    raw = d / "raw"
    raw.mkdir()
    (raw / "financials.json").write_text(_json.dumps(
        {"fundamentals": "Sector: Technology\nPE: 22.5\n"}))
    assert reports.load_company_name(d) == ""


def test_load_company_name_empty_run_dir_returns_empty():
    assert reports.load_company_name("") == ""
    assert reports.load_company_name(None) == ""
