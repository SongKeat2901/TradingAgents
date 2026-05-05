# Deterministic Earnings Calendar — Design

**Date:** 2026-05-05
**Status:** approved (user approved 2026-05-05)
**Predecessor:** [Deterministic Classifier](2026-05-04-deterministic-classifier-design.md)

## Goal

Eliminate LLM hallucination of past-vs-future status for peer earnings events. Run #6 of the deterministic-classifier validation (2026-05-04, trade date 2026-05-01) emitted a TA v2 report that said "GOOGL/AMZN earnings (late April 2026, data to follow)" — but those earnings had already happened ~7 days earlier. The LLM had no explicit "today is 2026-05-01" anchor, so it inferred temporal status from training-data context and got it wrong.

This spec adds a deterministic earnings calendar (per-ticker, per-peer) computed in the Researcher, written to `raw/calendar.json`, and APPENDED verbatim to `pm_brief.md` after the PM Pre-flight LLM call. Downstream agents read pm_brief.md and see the calendar context as ground truth.

This is Option B from the prior brainstorm. Option A (a small `_SYSTEM` prompt addition in PM Pre-flight reinforcing "events dated before trade_date have already happened") is the cheap follow-on layered ON TOP of B — bundled in this spec as the final task.

## Architecture

```
Researcher (existing)
  ├─ writes raw/reference.json (existing)
  ├─ writes raw/classification.json (existing — Phase 6.1)
  └─ writes raw/calendar.json (NEW — Phase 6.2)

PM Pre-flight node (modified)
  ├─ runs LLM → writes pm_brief.md (existing)
  ├─ Python APPENDS deterministic "## Reporting status" section to pm_brief.md
  └─ All downstream agents that read pm_brief.md (TA v1/v2, 4 analysts, etc.)
     see the calendar block verbatim.
```

The calendar.json is bit-identical across runs from identical inputs (deterministic Python from yfinance data — yfinance returns are stable for a given trade_date). The append-after-LLM approach guarantees the calendar's dates aren't paraphrased or re-rendered by the LLM, in the same spirit as the classifier's "use verbatim" injection.

## Components

### `tradingagents/agents/utils/calendar.py` (new module)

```python
def compute_calendar(trade_date: str, tickers: list[str]) -> dict:
    """Pull last-reported and next-expected earnings dates per ticker.

    Returns a dict:
      {
        "trade_date": "2026-05-01",
        "MSFT": {
          "last_reported": "2026-04-29",
          "fiscal_period": "FY26 Q3",
          "next_expected": "2026-07-25",
          "source": "yfinance",
        },
        "GOOGL": {...},
        ...
        "_unavailable": ["TICKERX"]   # tickers where yfinance returned no data
      }

    For each ticker, calls yfinance.Ticker(symbol).earnings_dates, splits
    past (Reported EPS not NaN, date < trade_date) vs future (date > trade_date),
    returns the most-recent past + first-future.

    If yfinance returns empty/error for a ticker, writes
      {ticker: {"unavailable": True, "reason": "..."}}
    and adds the ticker to the top-level "_unavailable" list. Downstream
    agents fall back to LLM judgment for that ticker (same INDETERMINATE
    pattern as the classifier).

    The "fiscal_period" string is derived from the calendar quarter of
    last_reported. For tickers with non-calendar fiscal years (MSFT FY=Jul-Jun,
    ORCL FY=Jun-May, etc.), uses a small lookup table mapping ticker to
    fiscal-year start month.
    """
```

Fiscal-year lookup table (handful of mega-caps):

```python
_NON_CALENDAR_FISCAL_YEARS = {
    "MSFT": {"fy_start_month": 7},   # FY runs Jul-Jun
    "ORCL": {"fy_start_month": 6},   # FY runs Jun-May
    "ADBE": {"fy_start_month": 12},  # FY runs Dec-Nov
    "CRM":  {"fy_start_month": 2},   # FY runs Feb-Jan
}
```

For tickers not in the table, falls back to calendar-year fiscal periods (Q1 = Jan–Mar, etc.).

### `tradingagents/agents/researcher.py` (modify)

After the existing classifier call, add:

```python
from tradingagents.agents.utils.calendar import compute_calendar
calendar = compute_calendar(date, [ticker] + peers)
(raw / "calendar.json").write_text(
    json.dumps(calendar, indent=2, default=str), encoding="utf-8"
)
```

### `tradingagents/agents/managers/pm_preflight.py` (modify)

Add helper:

```python
def _format_calendar_block(raw_dir: str) -> str:
    """Format raw/calendar.json as a 'Reporting status' Markdown section
    for appending to pm_brief.md after the LLM call.

    Returns an empty string if calendar.json is missing or all tickers
    are unavailable, in which case downstream agents fall back to LLM
    judgment (same pattern as the classifier).
    """
```

Output format:

```markdown
\n\n## Reporting status (relative to trade_date 2026-05-01)

| Ticker | Most recent earnings | Status | Next expected |
|---|---|---|---|
| MSFT | FY26 Q3 reported 2026-04-29 | already happened | 2026-07-25 |
| GOOGL | Q1 2026 reported 2026-04-22 | already happened | 2026-07-23 |
| AMZN | Q1 2026 reported 2026-04-30 | already happened | 2026-07-31 |
| ORCL | FY26 Q3 reported 2026-03-13 | already happened | 2026-06-15 |

*Use these dates verbatim. Do not write "data to follow" or "upcoming"
for rows marked "already happened" — they happened before the trade date.
Treat them as rear-view information that should inform fundamental and
sentiment reasoning. The "next expected" dates are the forward catalyst
windows.*
```

If a ticker is unavailable: row reads `| TICKERX | (yfinance unavailable) | unknown | (yfinance unavailable) |`.

In `pm_preflight_node`, after the existing `(raw_dir / "pm_brief.md").write_text(brief, ...)` call, add:

```python
calendar_block = _format_calendar_block(state["raw_dir"])
if calendar_block:
    with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
        f.write(calendar_block)
```

This appends after the LLM-written content. The LLM never gets the chance to paraphrase or rewrite the dates.

### PM Pre-flight `_SYSTEM` prompt addition (Option A — small)

After the existing PM Pre-flight `_SYSTEM` content, add:

```
# Temporal anchor

Treat the trade date as "today". Events dated before it have already
occurred — never write them as "data to follow" or "upcoming". A
"Reporting status" table will be appended to your output programmatically
listing the most-recent and next-expected earnings dates for each ticker;
those dates are authoritative and you do not need to enumerate them
yourself in the brief.
```

This addresses the case where the PM Pre-flight LLM might write its OWN paragraph mentioning peer earnings — without this anchor, it could hallucinate "GOOGL Q1 to be reported in late April" even though the appended calendar block says it already happened.

## Data flow

1. Researcher fetches reference + classification (existing) + calendar (new).
2. Researcher writes `raw/calendar.json`.
3. PM Pre-flight LLM produces the brief (with the new "Temporal anchor" section in its prompt influencing tone).
4. Python appends the deterministic calendar block to pm_brief.md.
5. All downstream agents that read `pm_brief.md` via `format_for_prompt` see the calendar block as ground truth.

## Tests

### Unit (`tests/test_calendar.py`)

- `test_compute_calendar_splits_past_and_future`: monkeypatch yfinance.Ticker.earnings_dates to return a DataFrame with mix of NaN and reported EPS rows; assert correct past/future split.
- `test_compute_calendar_handles_empty_yfinance_return`: monkeypatch to return empty DataFrame; assert ticker shows `unavailable: True` and is in `_unavailable` list.
- `test_compute_calendar_handles_yfinance_exception`: monkeypatch to raise; assert ticker shows unavailable.
- `test_fiscal_period_derivation_msft`: 2026-04-29 reported MSFT → "FY26 Q3" (Jan–Mar 2026 = MSFT's Q3 since FY = Jul-Jun).
- `test_fiscal_period_derivation_calendar_year_company`: 2026-04-22 reported GOOGL → "Q1 2026" (calendar year).
- `test_compute_calendar_includes_trade_date_in_output`: top-level `trade_date` field set.

### Researcher integration (`tests/test_researcher.py` extension)

- `test_researcher_writes_calendar_json`: stub `compute_calendar` (or stub yfinance), call `fetch_research_pack`, assert `raw/calendar.json` exists with documented schema.

### PM Pre-flight (`tests/test_pm_preflight.py` extension)

- `test_pm_preflight_appends_calendar_block_to_brief`: stub LLM with simple AIMessage; pre-populate `raw/calendar.json`; call node; assert `raw/pm_brief.md` ends with the formatted calendar block.
- `test_pm_preflight_skips_calendar_block_when_missing`: no `raw/calendar.json`; assert pm_brief.md contains only LLM content (graceful fallback).
- `test_format_calendar_block_renders_unavailable_tickers`: stub calendar.json with one ticker marked unavailable; assert that row reads "(yfinance unavailable)".

### E2E validation (manual)

Run MSFT 2026-05-01 once on macmini-trueknot. Verify:
- `raw/calendar.json` exists with last_reported + next_expected for MSFT, GOOGL, AMZN, ORCL.
- `pm_brief.md` ends with the appended Reporting status table; the table contents match `calendar.json` byte-exactly.
- TA v2 report does NOT contain "GOOGL/AMZN earnings (late April 2026, data to follow)" or any equivalent phrasing for events that already happened.
- Bonus: rerun and confirm `raw/calendar.json` is byte-identical between the two runs (yfinance return for a given trade_date should be stable).

## Failure modes

- **yfinance returns empty for a ticker**: ticker marked unavailable, calendar block shows the row with "(yfinance unavailable)". Downstream agents fall back to LLM judgment for that one ticker.
- **yfinance raises**: caught at the per-ticker level; same fallback. Other tickers in the call still get populated.
- **Wrong fiscal-period derivation**: the small lookup table (`_NON_CALENDAR_FISCAL_YEARS`) covers MSFT, ORCL, ADBE, CRM. Other non-calendar-FY companies will get their fiscal period labeled as calendar-year (e.g., Q1 2026 instead of FY26 Q3). Acceptable for v1; add to the table when a future ticker surfaces a mislabel.
- **The LLM writes its own date narrative anyway**: the appended calendar block is the authority. The Option-A prompt anchor reduces this risk; if it still happens at audit, escalate to QC Item 16 (verify pm_brief's narrative dates match calendar.json).

## Out of scope

- **Macro calendar** (Fed meetings, CPI/PPI, FOMC). Add only if a separate audit finding surfaces macro-event hallucination.
- **Holiday/trading-day calendar** (NYSE half-days, etc.). Same.
- **Refresh on stale data** (if yfinance returns last quarter's results that have since been updated). yfinance is the source of truth; we don't second-guess it.
- **Per-ticker fundamental ingestion that uses the calendar dates as filter** (e.g., "include only the most recent post-earnings filing"). Stays at the existing fundamentals fetch.

## Trade-offs accepted

- yfinance is the data source. Flaky for some tickers, fine for the mega-caps that are typical peers in research-grade tickers. The fallback to "(yfinance unavailable)" is graceful.
- The fiscal-period derivation uses a small hand-coded lookup. Doesn't scale to every non-calendar-FY company, but covers the typical hyperscaler/mega-cap set. Easy to extend.
- The append-after-LLM approach makes the calendar block the very last content in pm_brief.md. Visually distinct, but pm_brief.md is consumed via `format_for_prompt` which doesn't care about position. Fine for downstream LLM consumption.
- Option A (the prompt anchor) is bundled here even though brainstorm presented it separately. Both ship together because A complements B (B locks the dates; A reduces the chance the LLM writes its own conflicting narrative). Splitting into two PRs adds no value.

## Rollback path

Each piece is independently reversible:
1. Revert the PM Pre-flight `_format_calendar_block` append (1 commit) — pm_brief.md returns to LLM-only content.
2. Revert the Researcher `compute_calendar` call (1 commit) — calendar.json stops being written; existing brief append code finds no file and gracefully no-ops.
3. Revert the calendar.py module (1 commit) — clean slate.

The Option A prompt addition is a single line in `pm_preflight._SYSTEM`; trivially revertable.

## Estimated effort

- Calendar module + tests: ~2 hours (most of the work is in the fiscal-period derivation edge cases)
- Researcher integration: ~20 min
- PM Pre-flight append + tests: ~30 min
- Option A prompt anchor: ~5 min
- E2E validation: ~30 min
- **Total: ~3–3.5 hours**
