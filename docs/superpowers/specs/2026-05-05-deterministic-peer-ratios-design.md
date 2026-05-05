# Deterministic Peer Ratios — Design

**Date:** 2026-05-05
**Status:** draft (pending user approval)
**Predecessor:** [Filing-anchor + Numerical Discipline](2026-05-05-filing-anchor-numerical-discipline-design.md)

## Goal

Stop the analyst tier from citing fabricated peer-comparison ratios when `raw/peers.json` already contains the underlying data. The Phase 6.3 audit + four MSFT 2026-05-01 e2e runs (`9717594` → `cb24edf` → `c5c41e4` → `0617182`) showed:

- Run #1 (`9717594`): PM cited `"GCP capex intensity 4.9% of revenue, below MSFT's 5.4%"` and `"AWS capex intensity 5.1%"` as load-bearing facts in the trim-vs-rotate-into-GOOGL recommendation. **Actual values from `raw/peers.json`: GOOGL 32.5% / AMZN 24.4% — a 6–7× magnitude error.**
- Runs #2–4 (`cb24edf`, `c5c41e4`, `0617182`): the PM softened the claim to *"inherited from prior debate, not revalidated, treat as approximate"* and the QC LLM accepted the caveat. The fabricated ratios persisted across three rounds of QC tightening because **the QC LLM doesn't have access to `raw/peers.json`** — it only sees the PM's written document and `raw/reference.json`. The QC can't verify whether the data is in fact available.

This is a **prompt-architecture mismatch**: a QC rule that says *"if raw/peers.json contains the data, must recompute"* cannot bite when the QC tier can't read peers.json. The Phase 6.2 (deterministic calendar) and Phase 6.3 (SEC filing fetch) playbook addressed analogous problems by **computing the authoritative values in pure Python and Python-appending them verbatim to `pm_brief.md` after the LLM call**, removing the LLM's chance to paraphrase.

This spec applies the same pattern to peer ratios.

## Architecture

```
PM Pre-flight node (modified)
  ├─ runs LLM → writes pm_brief.md (existing)
  ├─ Python APPENDS "## Reporting status" calendar block (Phase 6.2)
  ├─ Python fetches latest 10-Q + writes raw/sec_filing.md (Phase 6.3)
  ├─ Python computes peer ratios → writes raw/peer_ratios.json (NEW — Phase 6.4)
  └─ Python APPENDS "## Peer ratios (computed)" block to pm_brief.md (NEW)

PM / Fundamentals / TA v2 / debaters (consumers)
  └─ All read pm_brief.md via format_for_prompt; see authoritative peer ratios verbatim.

QC agent (no change needed)
  └─ Item 16(c) "Peer-comparison deltas reconcile" already requires that
     any peer-ratio claim reconcile with values in raw/peers.json. Once
     pm_brief.md contains the authoritative block, any LLM-side deviation
     becomes a deterministic 16(c) FAIL.
```

The block lands in `pm_brief.md` after the existing Phase 6.2 calendar block and Phase 6.3 SEC filing footer, in that fixed order. Downstream agents reading `pm_brief.md` via `format_for_prompt` see the appended ratios as ground truth.

## Components

### `tradingagents/agents/utils/peer_ratios.py` (new module)

```python
def compute_peer_ratios(peers_data: dict, trade_date: str) -> dict:
    """Compute authoritative peer ratios from raw/peers.json data.

    For each peer ticker present in peers_data, derive:
      - latest_quarter_capex_to_revenue: Q1 capex / Q1 revenue, in %
      - latest_quarter_op_margin: Q1 op income / Q1 revenue, in %
      - ttm_pe: parsed from `fundamentals` text block (PE Ratio (TTM): X)
      - forward_pe: parsed from `fundamentals` text block (Forward PE: Y)

    Returns:
        {
          "trade_date": "2026-05-01",
          "_unavailable": ["TICKERX"],   # tickers where computation failed
          "GOOGL": {
            "latest_quarter_capex_to_revenue": 32.5,
            "latest_quarter_op_margin": 36.1,
            "ttm_pe": 29.23,
            "forward_pe": 26.68,
            "source": "peers.json (yfinance via Q1 capex/revenue)",
          },
          "AMZN": {...},
          ...
        }

    Per-ticker fallback: if any computation fails (missing column, parse
    error, divide-by-zero), the ticker enters `_unavailable`. Other peers
    in the same call still get populated. Mirrors the Phase 6.2 calendar
    fallback pattern.
    """
```

### `tradingagents/agents/managers/pm_preflight.py` (modify)

After the existing Phase 6.3 SEC filing footer append block, add a third Python-append:

```python
from tradingagents.agents.utils.peer_ratios import compute_peer_ratios
peers_path = raw_dir / "peers.json"
if peers_path.exists():
    try:
        peers_data = json.loads(peers_path.read_text(encoding="utf-8"))
        ratios = compute_peer_ratios(peers_data, date)
        (raw_dir / "peer_ratios.json").write_text(
            json.dumps(ratios, indent=2, default=str), encoding="utf-8"
        )
        peer_block = _format_peer_ratios_block(ratios)
        if peer_block:
            with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
                f.write(peer_block)
            brief = brief + peer_block
    except (OSError, json.JSONDecodeError, KeyError):
        pass  # Graceful degradation; downstream agents fall back to LLM judgment
```

Helper `_format_peer_ratios_block(ratios)` renders a Markdown table:

```markdown


## Peer ratios (computed from raw/peers.json, trade_date 2026-05-01)

| Ticker | Q1 capex/revenue | Q1 op margin | TTM P/E | Forward P/E |
|---|---|---|---|---|
| GOOGL | 32.5% | 36.1% | 29.23x | 26.68x |
| AMZN | 24.4% | 11.0% | 32.58x | 41.32x |
| ORCL | 24.7% | 26.0% | 32.37x | 23.90x |
| CRM | 0.5% | 13.7% | 23.75x | 21.00x |

*Use these values verbatim. Do not cite "approximate" or "inherited from
prior debate" alternatives — these are the authoritative current-quarter
figures derived from yfinance data on the trade date. If you need to
make a peer-comparison claim, recompute deltas from this table, not from
memory.*
```

If a peer is in `_unavailable`, its row reads `(unavailable)` for the missing fields.

### Researcher / pm_brief consumers — no changes needed

The Fundamentals analyst, TA v2, PM, and debaters already read `pm_brief.md` via `format_for_prompt`. The appended block is automatically in their context.

The Fundamentals analyst's existing "Mandatory pre-write step: YoY computation from financials.json" can optionally be extended with a "Mandatory pre-write step: peer ratio recomputation from raw/peers.json" — but **this is not strictly required** because the deterministic block already provides the authoritative values. Let the LLM decide whether to lean on the block.

### QC agent (`tradingagents/agents/managers/qc_agent.py`)

Item 16(c) already requires peer-comparison deltas to reconcile with values *"elsewhere in the same document (analyst reports, debates) OR with values in raw/peers.json."* No change needed: `pm_brief.md` (which becomes part of the document) now carries authoritative peer ratios, so any LLM-side deviation becomes a 16(c) FAIL.

Optionally, a one-line addition to item 16(a) clarifying that *"the deterministic ## Peer ratios block in pm_brief.md is the authoritative source for peer ratios; recomputation from peers.json is not required when the block is present"* — but this is cosmetic and can wait.

## Data flow

1. Researcher writes `raw/peers.json` (existing — has full income_statement, cashflow, fundamentals text per peer).
2. PM Pre-flight LLM produces `pm_brief.md` (existing).
3. PM Pre-flight Python appends Phase 6.2 calendar block (existing).
4. PM Pre-flight Python fetches + writes `raw/sec_filing.md` + appends Phase 6.3 footer (existing).
5. PM Pre-flight Python computes `raw/peer_ratios.json` from `raw/peers.json` and appends `## Peer ratios` block to `pm_brief.md` (NEW — Phase 6.4).
6. Fundamentals analyst, TA v2, PM, debaters read `pm_brief.md` and see all three appended blocks as ground truth.
7. QC item 16(c) catches any peer claim in the PM draft that doesn't reconcile with the authoritative block.

## Tests

### Unit (`tests/test_peer_ratios.py`)

- `test_compute_peer_ratios_happy_path`: stub a peers_data dict with one quarter of revenue + capex + op income for GOOGL/AMZN; assert ratios match hand-computed values to 0.1%.
- `test_compute_peer_ratios_handles_missing_column`: peer has no Capital Expenditure row; assert peer enters `_unavailable` and others still populate.
- `test_compute_peer_ratios_handles_zero_revenue`: peer has revenue=0 (degenerate edge case); assert no ZeroDivisionError, peer enters `_unavailable`.
- `test_compute_peer_ratios_parses_pe_ratios_from_fundamentals_text`: stub `fundamentals` text with "PE Ratio (TTM): 29.23\nForward PE: 26.68"; assert both parse correctly.
- `test_compute_peer_ratios_handles_unparseable_pe`: malformed PE field; assert ratio dict carries `ttm_pe: None` rather than raising.
- `test_compute_peer_ratios_top_level_trade_date`: assert top-level `trade_date` field set + `_unavailable` list.

### PM Pre-flight integration (`tests/test_pm_preflight.py` extension)

- `test_pm_preflight_appends_peer_ratios_block`: stub `compute_peer_ratios` to return a happy-path dict; pre-populate `raw/peers.json`; call node; assert `raw/peer_ratios.json` exists AND `pm_brief.md` ends with the formatted "## Peer ratios" block AFTER the calendar + SEC filing blocks.
- `test_pm_preflight_skips_peer_ratios_when_peers_json_missing`: no `raw/peers.json`; assert no peer_ratios.json written, no peer block in pm_brief.md.
- `test_pm_preflight_handles_peer_ratios_compute_exception`: `compute_peer_ratios` raises; assert pipeline still completes.

### E2E validation (manual, on macmini)

Run MSFT 2026-05-01 once. Verify:
- `raw/peer_ratios.json` exists with documented schema. GOOGL capex/revenue ≈ 32.5%, AMZN ≈ 24.4%, ORCL ≈ 24.7%, CRM ≈ 0.5%.
- `pm_brief.md` ends with three appended blocks in this order: calendar table → SEC filing footer → peer ratios table.
- The PM's decision document either (a) cites peer capex from the new authoritative block (numbers match `peer_ratios.json` to 0.1%), or (b) doesn't cite peer capex at all. The "GOOGL 4.9% / AMZN 5.1% inherited from prior debate" framing must be **gone**.
- QC verdict PASSes (or FAILs with item 16(c) feedback explicitly citing the deterministic block).

## Failure modes

- **`raw/peers.json` is missing** (e.g., researcher failed for that fetch). PM Pre-flight catches the missing file and skips the entire peer-ratio block. No file written, no block appended. Downstream LLMs fall back to whatever they have.
- **Per-peer compute failure** (missing column, malformed JSON sub-tree, parse error on PE text). The ticker enters `_unavailable`; other peers still populate. The block renders the unavailable row as `(unavailable)`.
- **Zero or negative revenue** (degenerate edge case, e.g. early-stage IPO with zero revenue). Caught by the per-peer try/except; peer enters `_unavailable`.
- **`fundamentals` text block changes format** (yfinance schema drift). The PE-parse regex returns None for the affected field; rest of the row populates. v2 could add a more robust parser.
- **The LLM still cites a fabricated peer ratio anyway**. QC item 16(c) now has authoritative values to reconcile against in the same document — any deviation triggers a FAIL on the next pass.

## Out of scope

- **Annualized vs single-quarter capex.** v1 cites Q1 figures only because that's what `peers.json` exposes per quarter. Annualized peer ratios (4×Q-capex / 4×Q-revenue) would be a future v2 enhancement once we cross-validate against full-year data.
- **Peer enterprise-value / EV-EBITDA / debt-adjusted ratios.** Not needed for the Phase 6.3 hallucination class. Add when an audit surfaces a fabricated EV-based ratio.
- **Peer-of-peer cross-comparisons.** v1 only computes own-ticker-vs-each-peer. Cross-peer matrices (GOOGL vs AMZN directly) are out of scope.
- **Sector aggregates.** "Hyperscaler average capex/revenue" is more useful than per-peer for some narrative arcs, but adding it requires deciding what's "the sector" — out of scope for v1.

## Trade-offs accepted

- The block uses Q1 capex/revenue, not LTM. Q1 is what's in `peers.json`. LTM would require summing 4 quarters per peer, plus handling stub quarters at the start of the rolling window. Q1 is simpler and the analyst can annualize in narrative form if needed.
- The block lands in `pm_brief.md` after the SEC filing footer, making `pm_brief.md` ~30 lines longer. Acceptable: the file is read via `format_for_prompt`, position-agnostic.
- The PE parsing reads from a text-formatted `fundamentals` field (yfinance dump). A schema change in yfinance breaks the parse. Mitigation: per-field None fallback + `_unavailable` rollup. v2 could parse from a more structured source if/when yfinance changes layout.
- Item 16(a) caveat-wrapping rule remains in place even though the block makes it largely redundant. Cheap defense in depth.

## Rollback path

Each piece reverts independently:

1. Revert PM Pre-flight peer-block append (1 commit). `peer_ratios.json` keeps being written, but pm_brief.md stops getting the block.
2. Revert PM Pre-flight `compute_peer_ratios` call (1 commit). Block stops being written; the peer_ratios.py module stays as orphan.
3. Revert the `peer_ratios.py` module + tests (1 commit). Clean slate.

## Estimated effort

- `peer_ratios.py` module + 6 unit tests: ~2 hours (PE-string regex + per-peer fallback edge cases dominate)
- PM Pre-flight integration + 3 tests: ~30 min
- E2E validation on macmini: ~30 min (single MSFT 2026-05-01 run)
- **Total: ~3 hours**
