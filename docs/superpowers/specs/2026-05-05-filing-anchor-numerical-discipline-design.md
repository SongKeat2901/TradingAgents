# Filing-anchor + Numerical Discipline — Design

**Date:** 2026-05-05
**Status:** approved (user approved 2026-05-05)
**Predecessor:** [Deterministic Earnings Calendar](2026-05-05-deterministic-earnings-calendar-design.md)

## Goal

Stop two related LLM hallucinations that surfaced together in Run #2 of the Phase-6.2 validation (MSFT, trade_date 2026-05-01):

1. **Filing-pending hallucination.** TA v2 framed the most-recent 10-Q as "the mid-May 10-Q is the binary catalyst pending adjudication" — but MSFT files the 10-Q same-day as earnings (2026-04-29), so the document was already public on the trade date. The agents missed Azure +40% YoY and Commercial RPO +99% YoY → $627B, both in the filed 10-Q, because the pipeline never fetched it.
2. **Fabricated multi-decimal ratio.** The Fundamentals analyst cited "MSFT capex/revenue 5.4%" when the actual value computed from `financials.json` is ~37.3%. The analyst paraphrased a number it had no way to derive — `financials.json` had the underlying quarterly columns but the analyst never computed YoY from them.

Both have the same root cause: **the analyst tier had no ground-truth filing data on hand, and no discipline forcing it to derive numbers from raw data instead of paraphrasing**. This spec ships two coupled mitigations:

- **Filing-anchor (item 15):** Pre-flight fetches the most recent 10-Q/10-K from SEC EDGAR, writes `raw/sec_filing.md`, appends a one-line "filing already public" footer to `pm_brief.md`. Fundamentals + TA v2 + PM read the filing text. QC item 15 fails any narrative that calls a filing in `raw/` "pending" or "awaiting".
- **Numerical discipline (item 16):** Fundamentals gets a mandatory YoY pre-write step computing Revenue / OpInc / Capex YoY + Capex/Revenue ratio from `financials.json` quarterly columns. QC item 16 fails any multi-decimal ratio that doesn't trace verbatim to a `raw/` cell.

The two are bundled because (a) item 16 is only enforceable once item 15's `sec_filing.md` exists for analysts to cite, (b) the QC checklist change is a single 14→16 expansion, (c) the e2e validation run is shared.

## Architecture

```
PM Pre-flight node (modified)
  ├─ runs LLM → writes pm_brief.md (existing)
  ├─ Python APPENDS deterministic "## Reporting status" calendar block (Phase 6.2)
  ├─ Python fetches latest 10-Q/10-K from EDGAR (NEW — Phase 6.3 filing-anchor)
  │   └─ writes raw/sec_filing.md + appends "filing already public" footer
  └─ Downstream agents see filing text + calendar + footer in pm_brief.md.

Fundamentals analyst (modified)
  ├─ reads pm_brief.md, financials.json, peers.json, news.json (existing)
  ├─ reads sec_filing.md (NEW — added to file list)
  └─ system prompt has mandatory YoY pre-write step (NEW — Phase 6.3 numerical discipline)

TA v2 agent (modified)
  └─ reads sec_filing.md (NEW — added to file list)

Portfolio Manager (modified)
  └─ surfaces sec_filing.md content as "Most recent SEC filing" block in PM prompt

QC agent (modified)
  ├─ checklist 14 → 16 items
  ├─ item 15: no "pending"/"awaiting" framing for filings already in raw/sec_filing.md
  └─ item 16: multi-decimal claims must trace to a raw/ source cell
```

The filing fetch and write happens in PM Pre-flight (not Researcher) because EDGAR access is the primary ticker only — peers add EDGAR rate-limit pressure and Pre-flight already runs once per ticker. The `format_for_prompt` aggregator that all downstream agents use will pick up `sec_filing.md` from `raw_dir` automatically once it's listed in their file lists.

## Components

### `tradingagents/agents/utils/sec_edgar.py` (new module)

```python
def fetch_latest_filing(ticker: str, trade_date: str, max_text_chars: int = 60_000) -> dict:
    """Fetch the most recent 10-Q or 10-K filed on or before trade_date from
    SEC EDGAR (free, public, no auth).

    Returns either:
      {
        "ticker": "MSFT",
        "form": "10-Q" | "10-K",
        "filing_date": "2026-04-29",
        "accession_number": "0001193125-26-191507",
        "primary_document": "msft-20260331.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/.../msft-20260331.htm",
        "content": "<HTML-stripped plain text, truncated to max_text_chars>",
        "content_truncated": bool,
        "source": "sec.gov",
      }
    OR
      {"unavailable": True, "reason": "<short string>", "ticker": ticker}

    Three steps: (1) resolve ticker → CIK via small in-module cache or
    SEC's company_tickers.json directory; (2) fetch
    data.sec.gov/submissions/CIK<padded>.json and find the most recent
    10-Q/10-K with filingDate ≤ trade_date (look-ahead-bias safe);
    (3) fetch the primary document URL, strip HTML (drop <script>/<style>),
    truncate to max_text_chars.
    """

def format_for_prompt(filing: dict) -> str:
    """Render a filing dict as a Markdown block for raw/sec_filing.md.
    Returns "" if the filing is unavailable.
    Block contains form/date header, accession + URL, the temporal-anchor
    instruction ("treat as known data, NEVER as 'pending adjudication'"),
    and the verbatim filing text."""
```

CIK lookup uses a small in-module cache for typical research tickers (~30 mega-caps including MSFT/AAPL/GOOGL/AMZN/META/NVDA/etc.) plus fallback to SEC's directory. HTTP uses `urllib.request` (no new dep) with a proper User-Agent (SEC blocks default UA).

### `tradingagents/agents/managers/pm_preflight.py` (modify)

After the existing calendar-block append, add:

```python
from tradingagents.agents.utils.sec_edgar import fetch_latest_filing, format_for_prompt as format_sec_filing
try:
    filing = fetch_latest_filing(ticker, date)
except Exception:
    filing = {"unavailable": True, "reason": "fetcher raised", "ticker": ticker}
sec_filing_md = format_sec_filing(filing)
if sec_filing_md:
    (raw_dir / "sec_filing.md").write_text(sec_filing_md, encoding="utf-8")
    footer = (
        f"\n## Recent SEC filing (relative to trade_date {date})\n\n"
        f"- **{filing['ticker']} {filing['form']} filed {filing['filing_date']}** — "
        f"contents already public on the trade date; full text in raw/sec_filing.md. "
        f"Treat as **known data**, never as \"pending adjudication\" or \"awaiting filing\".\n"
    )
    with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
        f.write(footer)
    brief = brief + footer
```

The `try/except` guard catches any fetcher exception and degrades to the unavailable path — the rest of the pipeline never breaks because EDGAR is down.

### `tradingagents/agents/analysts/fundamentals_analyst.py` (modify)

Two changes:

1. Add `sec_filing.md` to the `format_for_prompt` file list.
2. Add a mandatory pre-write section to the system prompt:

```
## Mandatory pre-write step: YoY computation from financials.json

Before writing the report, locate the most recent reported quarter from
calendar.json (the "Reporting status" table in pm_brief.md) and find the
matching column in financials.json's quarterly time series. Then compute:

- Revenue YoY: (Q_latest - Q_same_quarter_prior_year) / Q_same_quarter_prior_year
- Operating income YoY (same formula)
- Capital expenditure YoY (same formula)
- Capex / revenue ratio for the latest quarter

These four numbers must appear verbatim in the "Sanity check on reported
numbers" section. The raw quarterly columns are present in financials.json
already — DO NOT invent ratios from memory.

## Mandatory pre-write step: read raw/sec_filing.md if present

If raw/sec_filing.md exists, it contains the verbatim text of the most
recent 10-Q or 10-K filed on or before trade_date — published, public
information. Quote specific numbers from it (Remaining Performance
Obligations, segment revenue and operating income, Azure / cloud growth
rates) and weave them into the report. Never write "awaiting filing" or
"pending adjudication" for a document that exists in raw/sec_filing.md.
```

### `tradingagents/agents/analysts/ta_agent.py` (modify)

Add `sec_filing.md` to the TA v2 `format_for_prompt` file list. The TA v1 pass intentionally skips it (TA v1 is short-form chart reading; loading the 10-Q would balloon its context for marginal value).

### `tradingagents/agents/managers/portfolio_manager.py` (modify)

Add a `sec_block` next to the existing `reference_block` and `technicals_block`:

```python
sec_block = ""
try:
    sec_path = Path(state.get("raw_dir", "")) / "sec_filing.md"
    if sec_path.exists():
        sec_block = (
            "\n\n**Most recent SEC filing (already public on trade date — "
            "treat as known data, never as 'pending adjudication'):**\n"
            f"{sec_path.read_text(encoding='utf-8')}\n"
        )
except OSError:
    pass
```

Inject `{sec_block}` into the PM prompt template alongside the existing blocks. The OSError guard is the same pattern already used elsewhere in the file.

### `tradingagents/agents/managers/qc_agent.py` (modify)

Two new checklist items appended to the existing 14:

```
15. Filing-anchor temporal correctness. If raw/sec_filing.md exists (the
most recent 10-Q or 10-K, already public on the trade date), no analyst
quote or PM claim may describe its contents as "pending", "awaiting
filing", "not yet disclosed", or as "the binary catalyst that will reprice
the trade". Filings already in raw/ are KNOWN DATA. The PM may legitimately
say a NEXT filing (e.g., the next 10-Q in 3 months) is the catalyst, but
not the one already on EDGAR. → FAIL on any "pending"/"awaiting" / "data
to follow" framing applied to the filing whose text is in raw/sec_filing.md.

16. Multi-decimal numerical claims trace to a specific source cell.
Any claim of the form "X% capex-to-revenue" / "X% margin" / "Y bps
compression" / "Zx multiple" must trace verbatim to a cell in
raw/financials.json, raw/sec_filing.md, raw/peers.json, or raw/reference.json.
Fabricated peer-comparison ratios that don't appear in any raw/ file → FAIL.
```

Update header counts (14 → 16) and the system instruction line that says "walk through each of the 14 items briefly".

## Data flow

1. Researcher writes `reference.json`, `classification.json`, `calendar.json` (existing).
2. PM Pre-flight LLM produces `pm_brief.md` with calendar block appended (Phase 6.2).
3. PM Pre-flight Python fetches latest filing → writes `raw/sec_filing.md` + appends "Recent SEC filing" footer to `pm_brief.md` (NEW).
4. Fundamentals analyst reads `pm_brief.md` + `financials.json` + `sec_filing.md`, computes YoY pre-write, weaves filing-specific numbers into the report.
5. TA v2 reads `sec_filing.md` for fundamental context to anchor pattern interpretation.
6. PM consumes its existing prompt + `sec_block` containing the full filing text.
7. QC reads PM draft + verifies items 1–14 (existing) + 15 (filing temporal) + 16 (numerical trace).

## Tests

### Unit (`tests/test_sec_edgar.py`)

- `test_fetch_latest_filing_happy_path`: monkeypatch `_http_get` to return stub submissions JSON + filing HTML; assert form / filing_date / content / script-tag stripping.
- `test_fetch_latest_filing_skips_filings_after_trade_date`: filing dated 2026-04-29, trade_date 2026-04-28 → unavailable (no look-ahead bias).
- `test_fetch_latest_filing_unknown_ticker`: ticker not in cache, not in directory → unavailable with "CIK not found".
- `test_fetch_latest_filing_network_failure`: `_http_get` returns None → unavailable with "unreachable".
- `test_fetch_latest_filing_invalid_trade_date`: malformed date → unavailable, no network call.
- `test_fetch_latest_filing_truncates_long_content`: 200KB stub → truncated to `max_text_chars`, `content_truncated: True`.
- `test_format_for_prompt_emits_block_with_temporal_anchor`: assert markdown contains "treat them as known data" + "NEVER as 'pending adjudication'" + form/date.
- `test_format_for_prompt_returns_empty_for_unavailable`: unavailable dict → `""`.

### PM Pre-flight integration (`tests/test_pm_preflight.py` extension)

- `test_pm_preflight_writes_sec_filing_md_when_filing_available`: stub `fetch_latest_filing` to return a happy-path dict; call node; assert `raw/sec_filing.md` exists with the formatted block AND `pm_brief.md` ends with the "## Recent SEC filing" footer.
- `test_pm_preflight_omits_sec_filing_when_unavailable`: stub `fetch_latest_filing` to return `{"unavailable": True, ...}`; assert `raw/sec_filing.md` does NOT exist and `pm_brief.md` has no "## Recent SEC filing" section (graceful degradation, no fabrication).
- `test_pm_preflight_handles_fetcher_exception`: stub `fetch_latest_filing` to raise; assert pipeline still completes, no `sec_filing.md` written.

### Fundamentals analyst (`tests/test_fundamentals_analyst.py` — new file)

- `test_fundamentals_prompt_includes_yoy_preamble`: assert system prompt contains "Mandatory pre-write step: YoY computation".
- `test_fundamentals_reads_sec_filing_md_when_present`: monkeypatch `format_for_prompt` to record requested files; assert `sec_filing.md` is in the list.

### TA v2 (`tests/test_ta_agent_v2.py` extension)

- `test_ta_v2_reads_sec_filing_md_when_present`: same pattern as Fundamentals test — assert `sec_filing.md` is in TA v2's file list (TA v1 must NOT include it).

### Portfolio Manager (`tests/test_pm_qc_checklist.py` extension)

- `test_portfolio_manager_includes_sec_block_when_sec_filing_md_present`: stub state with `raw_dir` containing `sec_filing.md`; capture rendered prompt; assert sec_block is present and contains the file's content.
- `test_portfolio_manager_omits_sec_block_when_sec_filing_md_missing`: same setup minus the file; assert no sec_block in prompt.

### QC agent (`tests/test_qc_agent.py` extension)

- `test_qc_checklist_has_16_items_and_filing_anchor_text`: load `_SYSTEM`; assert "16-item checklist" appears + items 15 and 16 contain their respective key phrases ("pending", "trace verbatim").
- `test_qc_fails_pm_draft_calling_filed_10q_pending`: stub state with `raw/sec_filing.md` present; PM draft contains "the 10-Q pending adjudication"; QC verdict should be FAIL with item-15-related feedback.

### E2E validation (manual — macmini-trueknot)

Run MSFT 2026-05-01 once. Verify:
- `raw/sec_filing.md` exists with MSFT 10-Q dated 2026-04-29 and contains Azure/RPO numbers.
- `pm_brief.md` ends with "## Recent SEC filing" footer naming the same 2026-04-29 10-Q.
- Fundamentals report has a "Sanity check on reported numbers" section with the four computed YoY ratios drawn from `financials.json` quarterly columns. The capex/revenue figure is in the 30–40% range (not a fabricated 5.4%).
- TA v2 report does NOT call the filing "pending" / "awaiting" / "binary catalyst pending adjudication".
- QC verdict is PASS, or if FAIL, the feedback cites a real item-15 or item-16 violation rather than a fabrication.

## Failure modes

- **EDGAR unreachable / 5xx.** `_http_get` returns `None`; `fetch_latest_filing` returns `{"unavailable": True, "reason": "EDGAR submissions endpoint unreachable"}`. PM Pre-flight skips writing `sec_filing.md` and the footer. Pipeline runs as if Phase 6.3 wasn't deployed for this run.
- **Ticker not in CIK cache and SEC directory call fails.** Same graceful unavailable path.
- **Filing primary doc is XBRL-only / unparseable.** HTML stripper returns near-empty text; analyst gets a near-empty `sec_filing.md`. Acceptable; the temporal anchor still fires (item 15) because the file exists with its header. v2 could add a fallback to the filing's HTML index page.
- **EDGAR returns a different "most recent 10-Q" than yfinance reports as latest earnings.** Both are pulled fresh per run; minor divergence (1–2 day filing-vs-earnings-call lag) is fine because the calendar block (Phase 6.2) and the filing footer (this spec) both cite their own dates verbatim.
- **The LLM still writes "filing pending" anyway.** QC item 15 catches it on the second pass; the existing PM retry loop drives correction.
- **The LLM writes a fabricated ratio anyway.** QC item 16 catches it. If item 16 itself misclassifies (false positive), the operator can manually override per the existing QC retry policy.

## Out of scope

- **Historical filings older than the most-recent 10-Q/10-K** (e.g., past four 10-Qs for a longitudinal capex trend). Add only if a finding surfaces this gap.
- **8-K (current report) ingestion.** Earnings releases and material-event 8-Ks are out of scope for v1; the 10-Q already covers the structured financial data the analysts need.
- **Filing-text caching.** Each PM Pre-flight run re-fetches; for production-grade research at scale, add a per-(ticker, accession_number) disk cache. v1 does not.
- **EDGAR XBRL ingestion** for structured financial fact extraction. v1 stays text-only because the existing `financials.json` already covers the structured side.
- **News-article filing-anchor.** News articles can also reference earnings as "upcoming" — out of scope; news_analyst quote-handling is a separate audit item.

## Trade-offs accepted

- Filing fetch happens in PM Pre-flight, not Researcher. Researcher already runs early and fast; injecting EDGAR there would add 5–15s to first-pass latency for every research run including those that don't need a full PM cycle. Pre-flight is run once-per-research and PM-tier latency is dominated by LLM calls anyway.
- Only the primary ticker gets a SEC filing fetched. Peers don't get `sec_filing.md` per-ticker. Reason: EDGAR rate limits + diminishing returns for peer narrative depth. Phase 6.2's calendar block already gives peer earnings dates.
- HTML→text stripping is naive (drops `<script>` and `<style>`, regexes whitespace). Tables in the filing flatten to whitespace-collapsed text. Acceptable because the analyst doesn't need cell-perfect rendering; numbers and section headers come through. v2 could use a structured parser.
- Item 16 is broad ("multi-decimal claims must trace to raw"). Risk of false-positive FAILs on legitimate computed ratios from `financials.json`. The Fundamentals YoY pre-write step (item 16's complement) gives the analyst a clear path: compute from `financials.json`, cite the formula. If item 16 turns out to over-fire in production, narrow the rule (e.g., to peer-comparison ratios specifically).
- The QC checklist count goes from 14 to 16 in a single change. Splitting into 14→15 then 15→16 commits adds no value — they're tested together.

## Rollback path

Each piece is independently revertable:

1. Revert PM Pre-flight `fetch_latest_filing` block (1 commit). `sec_filing.md` stops being written; downstream agents that read it find no file and gracefully no-op (`format_for_prompt` ignores missing files).
2. Revert the `sec_edgar.py` module + tests (1 commit). Clean slate.
3. Revert Fundamentals YoY pre-write text (1 commit). Removes the mandatory section from `_SYSTEM`; analyst reverts to prior behavior.
4. Revert QC items 15+16 (1 commit). Checklist goes back to 14 items.
5. Revert PM `sec_block` (1 commit). Identical degradation — file ignored.

If only one of the two mitigations needs rolling back, item 15 (filing-anchor) and item 16 (numerical trace) live in separate commits and are independently revertable.

## Estimated effort

- `sec_edgar.py` module + 8 unit tests: ~3 hours (HTTP/HTML edge cases dominate)
- PM Pre-flight wiring + 3 tests: ~30 min
- Fundamentals + TA v2 + PM wiring + 5 tests: ~1 hour
- QC items 15+16 + 2 tests: ~30 min
- E2E validation on macmini: ~30 min (single MSFT run)
- **Total: ~5–5.5 hours**

(Implementation is already pre-written; this estimate reflects what the work *would* have taken from scratch and is included for parity with prior specs.)
