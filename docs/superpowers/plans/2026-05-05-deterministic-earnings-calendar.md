# Deterministic Earnings Calendar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the LLM from writing "data to follow" / "upcoming" for earnings that already happened. Add a pure-Python earnings calendar that the Researcher writes to `raw/calendar.json` and that PM Pre-flight Python-appends as a deterministic "Reporting status" Markdown table to `pm_brief.md` after the LLM call. Bundles a small `_SYSTEM` prompt anchor in PM Pre-flight as the LLM-side reinforcement.

**Architecture:** Researcher fetches `yfinance.Ticker(symbol).earnings_dates` per ticker, splits past (Reported EPS not NaN, date < trade_date) vs future (date > trade_date), writes `raw/calendar.json`. PM Pre-flight reads that JSON and appends a Markdown table to `pm_brief.md` after its own LLM-written content — guaranteeing the dates aren't paraphrased by the LLM. Downstream agents read `pm_brief.md` via `format_for_prompt` and see the calendar as ground truth.

**Tech Stack:** Python 3.13, yfinance (existing dep), pandas (yfinance return type), pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-05-deterministic-earnings-calendar-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tradingagents/agents/utils/calendar.py` | `compute_calendar(trade_date, tickers)` — pulls earnings dates per ticker, returns dict with last_reported + next_expected per ticker, gracefully handles yfinance unavailability per ticker |
| `tests/test_calendar.py` | Unit tests: past/future split, fiscal-period derivation (calendar-year + non-calendar like MSFT), unavailable fallback per ticker, top-level trade_date field |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/researcher.py` | Call `compute_calendar(date, [ticker] + peers)` after the existing classifier call; write `raw/calendar.json` |
| `tradingagents/agents/managers/pm_preflight.py` | Add `_format_calendar_block(raw_dir)` helper; append its result to `pm_brief.md` after the LLM-written content; add a "Temporal anchor" section to `_SYSTEM` (Option A) |
| `tests/test_researcher.py` | Assert `calendar.json` written + has documented schema |
| `tests/test_pm_preflight.py` | Assert pm_brief.md ends with the appended Reporting status block when calendar.json exists; assert graceful fallback when missing |

---

## Task 1: Calendar module + unit tests

**Files:**
- Create: `tradingagents/agents/utils/calendar.py`
- Create: `tests/test_calendar.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calendar.py`:

```python
"""Tests for the deterministic earnings calendar (Phase-6.2 temporal-anchor)."""
from datetime import date as _date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _earnings_df(rows):
    """Build a yfinance.Ticker.earnings_dates-style DataFrame.

    rows: list of (date_str, reported_eps_or_None) tuples. None → NaN
    (i.e., not yet reported).
    """
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"Reported EPS": [r[1] for r in rows]},
        index=idx,
    )


def test_compute_calendar_splits_past_and_future():
    """Past = Reported EPS not NaN AND date < trade_date.
    Future = date > trade_date.
    """
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2025-07-30", 2.95),     # past
        ("2025-10-29", 3.10),     # past
        ("2026-01-29", 3.22),     # past
        ("2026-04-29", 3.45),     # past — most recent
        ("2026-07-25", None),     # future — first expected
        ("2026-10-28", None),     # future
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["trade_date"] == "2026-05-01"
    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["MSFT"]["next_expected"] == "2026-07-25"
    assert out["MSFT"]["source"] == "yfinance"
    assert "_unavailable" in out
    assert out["_unavailable"] == []


def test_compute_calendar_handles_empty_yfinance_return():
    """If yfinance returns an empty DataFrame, the ticker is marked unavailable."""
    from tradingagents.agents.utils.calendar import compute_calendar

    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = pd.DataFrame()  # empty

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["TICKERX"])

    assert out["TICKERX"]["unavailable"] is True
    assert "yfinance" in out["TICKERX"]["reason"].lower()
    assert "TICKERX" in out["_unavailable"]


def test_compute_calendar_handles_yfinance_exception():
    """If yfinance raises, the ticker is marked unavailable; other tickers still get computed."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings_msft = _earnings_df([
        ("2026-04-29", 3.45),
        ("2026-07-25", None),
    ])

    def _ticker_factory(symbol):
        if symbol == "BAD":
            raise RuntimeError("yfinance internal error")
        m = MagicMock()
        m.earnings_dates = earnings_msft
        return m

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", side_effect=_ticker_factory):
        out = compute_calendar("2026-05-01", ["MSFT", "BAD"])

    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["BAD"]["unavailable"] is True
    assert "BAD" in out["_unavailable"]
    assert "MSFT" not in out["_unavailable"]


def test_fiscal_period_derivation_msft_non_calendar_fy():
    """MSFT FY runs Jul-Jun. 2026-04-29 reports the quarter ending 2026-03-31,
    which is FY26 Q3 (Jan-Mar 2026 = 9 months into FY26 that started Jul-2025)."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-29", 3.45),
        ("2026-07-25", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["MSFT"]["fiscal_period"] == "FY26 Q3"


def test_fiscal_period_derivation_calendar_year_company():
    """For tickers without a non-calendar-FY entry in the lookup table,
    fall back to calendar-year fiscal periods (Q1 = Jan-Mar, etc.)."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-22", 1.85),
        ("2026-07-23", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["GOOGL"])

    # GOOGL not in the non-calendar-FY table → calendar-year fiscal period
    # 2026-04-22 reports the quarter ending 2026-03-31 → "Q1 2026"
    assert out["GOOGL"]["fiscal_period"] == "Q1 2026"


def test_no_past_earnings_returns_unavailable():
    """If yfinance returns only future earnings (none reported yet), the ticker
    is marked unavailable — last_reported is the field that anchors temporal reasoning."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-07-25", None),     # future only
        ("2026-10-28", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["NEWCO"])

    assert out["NEWCO"]["unavailable"] is True
    assert "NEWCO" in out["_unavailable"]


def test_no_future_earnings_still_returns_last_reported():
    """If yfinance has past earnings but no future row, last_reported is set
    and next_expected is None (acceptable; calendar block will say 'unknown')."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-29", 3.45),     # past only
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["MSFT"]["next_expected"] is None
    assert "MSFT" not in out["_unavailable"]
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_calendar.py -v
```

Expected: ImportError on `tradingagents.agents.utils.calendar`.

- [ ] **Step 3: Implement the calendar module**

Create `tradingagents/agents/utils/calendar.py`:

```python
"""Deterministic earnings calendar (Phase-6.2 temporal-anchor mitigation).

Run #6 of the deterministic-classifier validation (2026-05-04, trade date
2026-05-01) showed the LLM writing "GOOGL/AMZN earnings (late April 2026,
data to follow)" for events that had already happened ~7 days earlier.
The cause: no explicit "today is 2026-05-01" anchor in the prompt.

This module pulls per-ticker earnings dates from yfinance and produces a
deterministic dict that the Researcher writes to raw/calendar.json. PM
Pre-flight Python-appends a formatted Markdown block to pm_brief.md
after its LLM call so the dates can never be paraphrased.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import yfinance as yf


# Companies with non-calendar fiscal years. The fy_start_month is the
# month in which the company's fiscal year begins. For tickers not in
# this table, calendar-year fiscal periods are used (Q1 = Jan-Mar, etc.).
_NON_CALENDAR_FISCAL_YEARS: dict[str, dict[str, int]] = {
    "MSFT": {"fy_start_month": 7},   # FY runs Jul-Jun
    "ORCL": {"fy_start_month": 6},   # FY runs Jun-May
    "ADBE": {"fy_start_month": 12},  # FY runs Dec-Nov
    "CRM":  {"fy_start_month": 2},   # FY runs Feb-Jan
}


def _quarter_end_for_report_date(report_date: datetime) -> datetime:
    """Earnings are typically reported within ~30 days of quarter-end.
    Map a report date to the quarter-end it most likely covers."""
    # Walk back: the most recent quarter end (Mar/Jun/Sep/Dec) before report_date.
    year = report_date.year
    candidates = [
        datetime(year, 3, 31),
        datetime(year, 6, 30),
        datetime(year, 9, 30),
        datetime(year, 12, 31),
        datetime(year - 1, 12, 31),
    ]
    past = [c for c in candidates if c <= report_date]
    return max(past) if past else candidates[-1]


def _fiscal_period(ticker: str, report_date_str: str) -> str:
    """Return a "FY{YY} Q{n}" or "Q{n} {YYYY}" string for the given report date."""
    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    quarter_end = _quarter_end_for_report_date(report_date)

    if ticker in _NON_CALENDAR_FISCAL_YEARS:
        fy_start_month = _NON_CALENDAR_FISCAL_YEARS[ticker]["fy_start_month"]
        # Quarter index within fiscal year. fy_start_month=7 (MSFT) means
        # Jul-Sep is FY Q1, Oct-Dec is Q2, Jan-Mar is Q3, Apr-Jun is Q4.
        offset = (quarter_end.month - fy_start_month) % 12
        q_index = offset // 3 + 1  # 1-4
        # Fiscal year: the year in which the FY ends. For MSFT FY26 ends Jun-2026,
        # so a quarter ending 2026-03-31 is in FY26 (since fy_start was Jul-2025).
        fy_year = quarter_end.year if quarter_end.month >= fy_start_month else quarter_end.year
        # Edge case: if fy_start_month > the quarter month, the fiscal year already ticked over.
        # Example: MSFT report on 2026-04-29 covers quarter ending 2026-03-31 (Jan-Mar). The FY
        # started Jul-2025 (FY26), so this quarter is FY26 Q3. quarter_end.month=3 < fy_start=7,
        # but the FY year = 2026 because FY26 = Jul-2025 → Jun-2026.
        if quarter_end.month < fy_start_month:
            fy_year = quarter_end.year
        else:
            fy_year = quarter_end.year + 1
        return f"FY{fy_year % 100:02d} Q{q_index}"
    else:
        # Calendar-year fiscal period
        cal_q = (quarter_end.month - 1) // 3 + 1
        return f"Q{cal_q} {quarter_end.year}"


def _compute_one_ticker(symbol: str, trade_date: str) -> dict[str, Any]:
    """Pull earnings_dates for one ticker, split past/future, derive fiscal period."""
    try:
        ticker = yf.Ticker(symbol)
        earnings = ticker.earnings_dates
    except Exception as exc:  # noqa: BLE001 — yfinance can raise many error types
        return {"unavailable": True, "reason": f"yfinance error: {exc}"}

    if earnings is None or earnings.empty:
        return {"unavailable": True, "reason": "yfinance returned no earnings dates"}

    trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")

    # earnings_dates index is a DatetimeIndex; "Reported EPS" col is NaN for future
    # rows and a number for past rows.
    rows = earnings.reset_index()
    rows.columns = [c if c != "index" else "earnings_date" for c in rows.columns]
    # The actual yfinance column name for the index is sometimes "Earnings Date";
    # normalize.
    if "Earnings Date" in rows.columns:
        rows = rows.rename(columns={"Earnings Date": "earnings_date"})

    past_rows = []
    future_rows = []
    for _, row in rows.iterrows():
        d = row["earnings_date"]
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        d_naive = d.replace(tzinfo=None) if d.tzinfo else d
        date_str = d_naive.strftime("%Y-%m-%d")
        reported = row.get("Reported EPS")
        if reported is not None and not pd.isna(reported) and d_naive < trade_dt:
            past_rows.append((date_str, d_naive))
        elif d_naive > trade_dt:
            future_rows.append((date_str, d_naive))

    if not past_rows:
        return {"unavailable": True, "reason": "no past earnings before trade_date"}

    last_reported, _ = max(past_rows, key=lambda r: r[1])
    next_expected = None
    if future_rows:
        next_expected, _ = min(future_rows, key=lambda r: r[1])

    return {
        "last_reported": last_reported,
        "fiscal_period": _fiscal_period(symbol, last_reported),
        "next_expected": next_expected,
        "source": "yfinance",
    }


def compute_calendar(trade_date: str, tickers: list[str]) -> dict[str, Any]:
    """Build the per-ticker earnings calendar.

    trade_date: "YYYY-MM-DD" string. The "today" anchor for past/future split.
    tickers: list of ticker symbols (typically [main_ticker] + peers).

    Returns:
        {
          "trade_date": str,
          "<TICKER>": {
            "last_reported": "YYYY-MM-DD",
            "fiscal_period": "FY26 Q3" or "Q1 2026",
            "next_expected": "YYYY-MM-DD" or None,
            "source": "yfinance",
          } OR {"unavailable": True, "reason": "..."},
          ...
          "_unavailable": list of ticker symbols that returned unavailable
        }
    """
    out: dict[str, Any] = {"trade_date": trade_date, "_unavailable": []}
    for symbol in tickers:
        ticker_data = _compute_one_ticker(symbol, trade_date)
        out[symbol] = ticker_data
        if ticker_data.get("unavailable"):
            out["_unavailable"].append(symbol)
    return out


# pandas import is deferred to runtime — yfinance pulls it transitively but
# we don't import it at the top of the module to keep the import graph cheap.
import pandas as pd  # noqa: E402
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_calendar.py -v
```

Expected: 7 passed.

If `test_fiscal_period_derivation_msft_non_calendar_fy` fails: walk through the `_fiscal_period` math by hand for `2026-04-29` and verify the FY year and quarter index. The fy_start_month logic has an edge case around year-boundary that's annotated.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/calendar.py tests/test_calendar.py
git commit -m "feat(calendar): pure-Python deterministic earnings calendar (Phase-6.2)"
```

---

## Task 2: Researcher writes calendar.json

**Files:**
- Modify: `tradingagents/agents/researcher.py`
- Modify: `tests/test_researcher.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_researcher.py` and add this test at the bottom:

```python
def test_researcher_writes_calendar_json(tmp_path, monkeypatch):
    """Researcher must write raw/calendar.json with documented schema."""
    from tradingagents.agents import researcher

    # Stub the data fetchers so the test doesn't network out
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0),
        "close_200_sma": _INDICATOR_STUB(460.0),
        "rsi": _INDICATOR_STUB(58.0),
        "macd": _INDICATOR_STUB(1.2),
        "boll_ub": _INDICATOR_STUB(430.0),
        "boll_lb": _INDICATOR_STUB(390.0),
        "atr": _INDICATOR_STUB(8.0),
    })

    # Stub compute_calendar so the test doesn't network out either
    def fake_compute_calendar(trade_date, tickers):
        return {
            "trade_date": trade_date,
            "_unavailable": [],
            **{t: {
                "last_reported": "2026-04-29",
                "fiscal_period": "Q1 2026",
                "next_expected": "2026-07-29",
                "source": "yfinance",
            } for t in tickers},
        }
    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        fake_compute_calendar,
    )

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    cal_path = Path(state["raw_dir"]) / "calendar.json"
    assert cal_path.exists()
    cal = json.loads(cal_path.read_text())
    assert cal["trade_date"] == "2026-05-01"
    # Main ticker + 2 peers in _stub_state default
    assert "MSFT" in cal
    assert cal["MSFT"]["last_reported"] == "2026-04-29"
    assert cal["MSFT"]["next_expected"] == "2026-07-29"
    assert "_unavailable" in cal
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_researcher.py::test_researcher_writes_calendar_json -v
```

Expected: FAIL — `calendar.json` not yet written.

- [ ] **Step 3: Modify researcher.py**

Open `tradingagents/agents/researcher.py`. Find the existing block that calls `compute_classification` and writes `classification.json`:

```python
    from tradingagents.agents.utils.classifier import compute_classification
    classification = compute_classification(reference, prices.get("ohlcv", ""))
    (raw / "classification.json").write_text(
        json.dumps(classification, indent=2, default=str), encoding="utf-8"
    )
```

Add immediately after:

```python
    # Phase-6.2 temporal-anchor: deterministic earnings calendar.
    # See tradingagents/agents/utils/calendar.py + the design spec at
    # docs/superpowers/specs/2026-05-05-deterministic-earnings-calendar-design.md
    from tradingagents.agents.utils.calendar import compute_calendar
    calendar = compute_calendar(date, [ticker] + peers)
    (raw / "calendar.json").write_text(
        json.dumps(calendar, indent=2, default=str), encoding="utf-8"
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_researcher.py -v
```

Expected: all researcher tests pass (existing 6 + new 1 = 7).

Then full suite:

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 156 + 7 (calendar) + 1 (researcher addition) = 164 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_researcher.py
git commit -m "feat(researcher): write calendar.json alongside classification.json"
```

---

## Task 3: PM Pre-flight appends Reporting status block + adds Temporal anchor to system prompt

**Files:**
- Modify: `tradingagents/agents/managers/pm_preflight.py`
- Modify: `tests/test_pm_preflight.py`

- [ ] **Step 1: Write failing tests**

Open `tests/test_pm_preflight.py` and add these tests at the bottom:

```python
def test_pm_preflight_appends_calendar_block_to_brief(tmp_path):
    """If raw/calendar.json exists, pm_brief.md must end with a deterministic
    'Reporting status' block appended after the LLM-written content."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "calendar.json").write_text(_json.dumps({
        "trade_date": "2026-05-01",
        "_unavailable": [],
        "MSFT": {
            "last_reported": "2026-04-29",
            "fiscal_period": "FY26 Q3",
            "next_expected": "2026-07-25",
            "source": "yfinance",
        },
        "GOOGL": {
            "last_reported": "2026-04-22",
            "fiscal_period": "Q1 2026",
            "next_expected": "2026-07-23",
            "source": "yfinance",
        },
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# PM Brief: MSFT 2026-05-01\n\n## Peer set\n- GOOGL: hyperscaler peer\n")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    # The LLM content must come first
    assert brief.startswith("# PM Brief: MSFT 2026-05-01")
    # The Reporting status block must follow
    assert "## Reporting status (relative to trade_date 2026-05-01)" in brief
    # Each ticker must appear with its last-reported date and "already happened"
    assert "MSFT" in brief
    assert "FY26 Q3 reported 2026-04-29" in brief
    assert "GOOGL" in brief
    assert "Q1 2026 reported 2026-04-22" in brief
    assert "already happened" in brief
    # Both next_expected dates must appear
    assert "2026-07-25" in brief
    assert "2026-07-23" in brief


def test_pm_preflight_skips_calendar_block_when_calendar_missing(tmp_path):
    """If raw/calendar.json doesn't exist, pm_brief.md should contain only
    the LLM content. No fallback fabrication."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    # No calendar.json

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief content\n\n## Peer set\n- AAPL: peer\n")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "Reporting status" not in brief
    assert brief.startswith("# Brief content")


def test_pm_preflight_calendar_block_renders_unavailable_tickers(tmp_path):
    """A ticker marked unavailable in calendar.json should render '(yfinance unavailable)'."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "calendar.json").write_text(_json.dumps({
        "trade_date": "2026-05-01",
        "_unavailable": ["TICKERX"],
        "MSFT": {
            "last_reported": "2026-04-29",
            "fiscal_period": "FY26 Q3",
            "next_expected": "2026-07-25",
            "source": "yfinance",
        },
        "TICKERX": {"unavailable": True, "reason": "yfinance returned no earnings dates"},
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "TICKERX" in brief
    assert "yfinance unavailable" in brief.lower() or "(yfinance unavailable)" in brief


def test_pm_preflight_system_prompt_has_temporal_anchor():
    """Option A: PM Pre-flight _SYSTEM must include a Temporal anchor section
    instructing the LLM not to fabricate past-vs-future status."""
    from tradingagents.agents.managers.pm_preflight import _SYSTEM
    assert "Temporal anchor" in _SYSTEM or "trade date as \"today\"" in _SYSTEM
    assert "data to follow" in _SYSTEM or "already occurred" in _SYSTEM
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: 4 new tests fail.

- [ ] **Step 3: Add `_format_calendar_block` helper to `pm_preflight.py`**

Open `tradingagents/agents/managers/pm_preflight.py`. After the `_extract_peers` function (around line 95-100, before `create_pm_preflight_node`), add:

```python
def _format_calendar_block(raw_dir: str) -> str:
    """Format raw/calendar.json as a 'Reporting status' Markdown block for
    appending to pm_brief.md after the LLM call.

    Returns "" if calendar.json is missing or all tickers are unavailable —
    in which case downstream agents fall back to LLM judgment for temporal
    reasoning (same INDETERMINATE pattern as the classifier).
    """
    import json as _json
    cal_path = Path(raw_dir) / "calendar.json"
    if not cal_path.exists():
        return ""
    try:
        cal = _json.loads(cal_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError:
        return ""

    trade_date = cal.get("trade_date", "?")
    unavailable_set = set(cal.get("_unavailable", []))

    # Collect ticker rows in insertion order, skipping the bookkeeping keys
    rows = []
    for key, val in cal.items():
        if key in ("trade_date", "_unavailable"):
            continue
        if key in unavailable_set or val.get("unavailable"):
            rows.append(
                f"| {key} | (yfinance unavailable) | unknown | (yfinance unavailable) |"
            )
            continue
        last = val.get("last_reported", "?")
        period = val.get("fiscal_period", "?")
        nxt = val.get("next_expected") or "(unknown)"
        rows.append(
            f"| {key} | {period} reported {last} | already happened | {nxt} |"
        )

    if not rows:
        return ""

    table = "\n".join(rows)
    return (
        f"\n\n## Reporting status (relative to trade_date {trade_date})\n\n"
        "| Ticker | Most recent earnings | Status | Next expected |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        "*Use these dates verbatim. Do not write \"data to follow\" or "
        "\"upcoming\" for rows marked \"already happened\" — they happened "
        "before the trade date. Treat them as rear-view information that "
        "should inform fundamental and sentiment reasoning. The \"next "
        "expected\" dates are the forward catalyst windows.*\n"
    )
```

- [ ] **Step 4: Wire the append into the node body**

In the same file, find the `pm_preflight_node` function. Locate this block:

```python
        (raw_dir / "pm_brief.md").write_text(brief, encoding="utf-8")
        peers = _extract_peers(brief)
```

Replace with:

```python
        (raw_dir / "pm_brief.md").write_text(brief, encoding="utf-8")

        # Phase-6.2 temporal-anchor: append the deterministic earnings
        # calendar AFTER the LLM-written content so dates can never be
        # paraphrased. See docs/superpowers/specs/2026-05-05-deterministic-earnings-calendar-design.md.
        calendar_block = _format_calendar_block(state["raw_dir"])
        if calendar_block:
            with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
                f.write(calendar_block)
            brief = brief + calendar_block

        peers = _extract_peers(brief)
```

The `brief` local variable is also updated so the `peers = _extract_peers(brief)` call still operates on the full content (in case a future change adds peer-extraction logic that benefits from seeing the calendar block — though it doesn't today). The state's `pm_brief` field returned to the graph also reflects the appended content.

- [ ] **Step 5: Add Temporal anchor section to `_SYSTEM`**

In the same file, find the `_SYSTEM` constant. At the END of the existing `_SYSTEM` content (after the "Be concrete and falsifiable. No vague questions like ..." line), add:

```python

# Temporal anchor

Treat the trade date as "today". Events dated before it have already
occurred — never write them as "data to follow", "upcoming", or
"data to be reported". A "Reporting status" table will be programmatically
appended to your output listing the most-recent and next-expected earnings
dates for each ticker; those dates are authoritative and you do not need
to enumerate them yourself in the brief.
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: all PM Pre-flight tests pass (existing + 4 new).

Then full suite:

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 164 + 4 = 168 tests pass.

- [ ] **Step 7: Commit**

```bash
git add tradingagents/agents/managers/pm_preflight.py tests/test_pm_preflight.py
git commit -m "feat(pm-preflight): append Reporting-status block + Temporal anchor in _SYSTEM"
```

---

## Task 4: E2E validation on macmini

**Files:** none (operator step)

- [ ] **Step 1: Push + redeploy**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main && .venv/bin/pip install -e . --quiet'
ssh macmini-trueknot 'cd ~/tradingagents && git rev-parse --short HEAD'
```

Expected: HEAD matches the commit from Task 3.

- [ ] **Step 2: Refresh OAuth + run MSFT 2026-05-01**

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min. Verify:

```bash
ssh macmini-trueknot 'pgrep -fl tradingresearch | head -3'  # empty
```

- [ ] **Step 3: Inspect calendar.json**

```bash
ssh macmini-trueknot 'cat ~/.openclaw/data/research/2026-05-01-MSFT/raw/calendar.json'
```

Expected: a JSON dict with `trade_date: "2026-05-01"`, `_unavailable: []` (or similar list), and per-ticker entries for MSFT + each peer with `last_reported`, `fiscal_period`, `next_expected`, `source: "yfinance"`. For MSFT, `last_reported` should be a date in late April 2026.

- [ ] **Step 4: Inspect pm_brief.md tail**

```bash
ssh macmini-trueknot 'tail -25 ~/.openclaw/data/research/2026-05-01-MSFT/raw/pm_brief.md'
```

Expected: ends with the "## Reporting status" table. The table dates must match `calendar.json` byte-exactly.

- [ ] **Step 5: Check that TA v2 no longer says "data to follow" for past earnings**

```bash
ssh macmini-trueknot '
grep -in "data to follow\|to follow\|to be reported\|earnings.*\(late.April\|late-April\)" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals_v2.md
'
```

Expected: no match (or any matches are about events that genuinely are in the future, like the late-July Q4 print). The specific bug ("GOOGL/AMZN earnings (late April 2026, data to follow)") should be gone.

- [ ] **Step 6: Run a SECOND time and confirm calendar.json is byte-identical between runs**

```bash
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-calA'
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min. Then:

```bash
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-calB'
ssh macmini-trueknot '
diff -q \
  ~/.openclaw/data/research/2026-05-01-MSFT-calA/raw/calendar.json \
  ~/.openclaw/data/research/2026-05-01-MSFT-calB/raw/calendar.json
'
```

Expected: no diff (yfinance returns are stable for a given trade_date).

- [ ] **Step 7: Cleanup**

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-calA
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-calB
'
```

- [ ] **Step 8: Report findings**

If all checks pass:
- ✅ T4 complete. The temporal-anchor mitigation works.
- LLM no longer hallucinates past-vs-future status for peer earnings.
- The append-after-LLM design eliminates date paraphrasing.

If TA v2 still says "data to follow" for past events:
- The Option-A prompt anchor didn't fully bite. Tighten the language or add a QC Item 16 that explicitly checks for "data to follow" / "upcoming" verbiage applied to dates earlier than trade_date.

If calendar.json differs between runs:
- yfinance returned different data on the two pulls. Add caching with the trade_date as the key (out of scope for v1).

---

## Self-review notes

**Spec coverage:**
- ✅ Pure-Python `compute_calendar` module (Task 1)
- ✅ 6 unit tests covering past/future split, empty/exception/no-past edge cases, fiscal-period derivation for both calendar-year and non-calendar (MSFT) (Task 1)
- ✅ Researcher writes `raw/calendar.json` (Task 2)
- ✅ PM Pre-flight `_format_calendar_block` helper that appends to pm_brief.md after LLM call (Task 3)
- ✅ Option A `_SYSTEM` Temporal anchor section (Task 3)
- ✅ Graceful fallback when calendar.json missing or ticker unavailable (Task 1 module logic + Task 3 helper returns "")
- ✅ E2E validation: calendar.json byte-identical across runs + pm_brief.md ends with the table + TA v2 no longer says "data to follow" for past events (Task 4)

**Type / signature consistency:**
- `compute_calendar(trade_date: str, tickers: list[str]) -> dict[str, Any]` — defined in Task 1, called in Task 2 with `(date, [ticker] + peers)`, output structure (`{ticker: {last_reported, fiscal_period, next_expected, source}}` or `{ticker: {unavailable, reason}}`) is consumed by `_format_calendar_block` in Task 3.
- `_format_calendar_block(raw_dir: str) -> str` — defined in Task 3 step 3, called in Task 3 step 4.
- All three documented top-level keys (`trade_date`, `_unavailable`, per-ticker dicts) appear in test stubs (Task 2 + Task 3) and in the formatter logic.

**Placeholder scan:**
- No "TBD" / "implement later" / "similar to Task N" patterns.
- Each step shows exact code, exact file paths, exact verification commands, exact expected output.
- The fiscal-period derivation has explicit step-by-step examples in the test (`2026-04-29 MSFT → FY26 Q3`) and explicit edge-case logic in the implementation.

**Out-of-scope confirmation:** Macro calendar (Fed/CPI), holiday calendar, refresh-on-stale-yfinance — all explicitly listed in the spec as separate workstreams. Nothing in this plan extends into those.

**Rollback path:** Each task commits independently. Reverting Task 3 keeps `calendar.json` written but unused (pm_brief.md returns to LLM-only content). Reverting Task 2 leaves the calendar module orphan. Reverting Task 1 deletes the module entirely.
