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


def test_load_base_ev_survives_unreadable_decision(tmp_path):
    import json
    d = tmp_path / "2026-06-01-ERR"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "ERR", "trade_date": "2026-06-01"}))
    (d / "decision.md").mkdir()   # a dir where a file is expected → OSError on read_text
    assert reports.load_base_ev(d) is None
