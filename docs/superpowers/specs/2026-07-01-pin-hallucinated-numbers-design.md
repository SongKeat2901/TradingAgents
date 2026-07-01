# Phase C — Pin Hallucinated Numbers (Rerun-Reduction, Phase 1 of 3)

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Reduce how often a cadence run trips the post-hoc Phase-7 validators, so fewer full top-down reruns are needed. This is **Phase C** of a three-phase rerun-reduction effort agreed with the user (C → A → B). Phases A and B are out of scope here and get their own specs.

---

## Background

When a cadence run (`tradingresearch --ticker T --date D`) fails validation, the whole pipeline must be re-run from the top (~20–35 min: re-fetch yfinance + SEC, recompute all deterministic blocks, re-run every LLM stage). The failure mode is structurally unfortunate: the LangGraph runs to completion and writes `decision.md`, and only *then* does a **separate, post-hoc, file-scanning validator** (`cli/research_validation.py` → `tradingagents/validators/*`) run. A blocking violation just sets exit code 3 + writes `validation_report.json`; there is no feedback into the graph and no retry loop. The operator (or `cadence-run` skill) re-invokes the whole thing fresh (confirmed: no checkpoint on the production path, no `raw/*.json` reuse, validators decoupled from the graph).

**The three-phase plan (user-chosen ordering C → A → B):**
- **Phase C (this spec):** make failures *rarer* — pin the numbers the LLM hallucinates so fewer runs trip the validator at all. Cheapest, lowest-risk, aligned with the existing deterministic-block pattern.
- **Phase A (later):** make reruns *cheap* — reuse `raw/*.json` + resume completed LLM stages (the `SqliteSaver` checkpointer already exists but is disabled on the production path).
- **Phase B (later):** make reruns *unnecessary* — move validators inline and self-correct the failing stage.

**The two failures we actually observed (MSFT 2026-06-30, two runs):**
1. **Price hallucination (dominant):** the LLM wrote *"Jun 29 close $359.90"* in an exit-tranche/stop context; the actual 2026-06-29 close was $368.57 (Δ$8.67). `phase_7_1_price_date`, MATERIAL/blocking.
2. **Net-debt definitional drift:** the fundamentals analyst invented a derived *"~$29B net-debt divergence"* figure while musing about net-cash vs yfinance's $8.16B. `phase_7_5_net_debt`, MATERIAL/blocking — even though the raw `net_debt.json` ($8.16B) is correct and was cited authoritatively.

## Findings that make this design work (verified against the validators)

- **The price validator's canonical `actual_close` comes from `raw/prices.json`'s `ohlcv` CSV, Close column (index 4)** — never a live re-fetch (`tradingagents/validators/price_date_validator.py:52-79,189-190`). That Close is `.round(2)` and already `drop_incomplete_session`-filtered upstream (`dataflows/y_finance.py:33,52`). **⇒ A pinned block built from the same `prices["ohlcv"]`, same column, same rounding agrees with the validator by construction** (well inside the flat **$0.50** tolerance, `price_date_validator.py:32`).
- **`raw/pm_brief.md` is NOT in the validator's `_FILES_TO_SCAN`** (`cli/research_validation.py:23-34`). ⇒ A block pinned there can never self-flag; it only helps insofar as downstream authors copy the numbers verbatim.
- **`prices` is already in memory at the block-append points** (`researcher.py:205,249`), so no new fetch is needed; append mechanics mirror `net_debt`/`accounting_ratios`/`latest_session` (`researcher.py:400-401,459-460,515-516`).
- **`phase_7_5_net_debt` flags `definitional_drift` only when a net-debt-labeled figure is outside tolerance of *every* one of ~8 canonical derivations** it builds from `raw/net_debt.json` (`net_debt_validator.py:27-39,666-679`; tol = 5% or $0.5B). ⇒ Forcing the analyst to only restate a canonical figure (not compute a novel derived one) directly reduces drift hits.

## Design principles

- **Deterministic-block pattern:** compute in Python, append to `pm_brief.md` as ground truth, the LLM quotes it verbatim. (Ref: `project_phase6_deterministic_blocks_pattern`.)
- **Source-alignment invariant (the crux):** the recent-closes block MUST derive from the exact same `raw/prices.json` `ohlcv` Close values the validator reads — same parse, same column, same rounding. Never a fresh yfinance call. This is what makes the pin actually prevent violations instead of relocating them.
- **Free-data honesty:** missing/unparseable prices → the block renders an explicit "recent closes unavailable" note, never fabricated rows; no pipeline crash.
- **Probabilistic success:** this reduces hallucination *frequency*; it is not a hard guarantee. Success is measured across cadence runs, not by a single unit assertion.

---

## Part 1 — `## Recent closes` ground-truth block (primary)

### Module: `tradingagents/agents/utils/recent_closes.py` (mirrors `latest_session.py`)

- `compute_recent_closes(prices: dict, trade_date: str, n: int = 10) -> dict`
  - Parse `prices["ohlcv"]` (CSV text; skip `#`-comment and header lines, exactly like `latest_session.py` / `price_date_validator.py:70`). Take the rows with `Date <= trade_date`, keep the last `n` (default **10**), most-recent-first. Close = column index 4, used verbatim (already `.round(2)`).
  - Return `{"trade_date": trade_date, "as_of": <latest date in table>, "rows": [{"date": d, "close": c}, ...], "source": "raw/prices.json ohlcv (Close, col 4)"}`. Empty/unparseable → `{"rows": [], "unavailable": True, "reason": ...}`.
- `format_recent_closes_block(rc: dict) -> str`
  - Render `## Recent closes (last N sessions, verbatim from raw/prices.json)` with a `| Date | Close |` table, most-recent-first.
  - Trailing mandate (verbatim intent): *"Any closing price you cite for a specific date MUST be quoted verbatim from this table (source: raw/prices.json Close). Do not state a close for a date not listed here — if you need an older close, say 'not in the recent-closes table' rather than estimating. This is the same source the validator checks."*
  - `unavailable` → return a short *"## Recent closes — unavailable (<reason>); do not cite specific-date closes"* string (honest, non-crashing).

### Wiring: `researcher.py`
- Immediately after the existing `latest_session` block append (~`researcher.py:515-516`), in the same style: `rc = compute_recent_closes(prices, date)`, write `raw/recent_closes.json` (`json.dumps(..., indent=2, default=str)`), `block = format_recent_closes_block(rc)`, append to `pm_brief_path`. Reuse the in-memory `prices` dict — no new fetch.
- Guard with the same try/except posture as the sibling blocks so a compute failure can't crash the run (append the "unavailable" note instead).

### Reuse note
`latest_session.py` already parses `ohlcv` rows; factor the row-parse so `recent_closes.py` and `latest_session.py` don't duplicate CSV parsing (either import `latest_session`'s row parser or a shared helper). Do NOT re-implement a third CSV parser.

---

## Part 2 — Net-debt commentary discipline (prompt-only)

- In `tradingagents/agents/analysts/fundamentals_analyst.py` (`_SYSTEM`): add an instruction that any net-debt / net-cash figure MUST be restated verbatim from the pinned `## Net debt` block (or one of the canonical derivations it shows) — the analyst may NOT compute and cite a *novel* derived net-debt/net-cash divergence number (e.g. "yfinance diverges by ~$29B"). If it wants to note the net-cash framing, it must use the block's own cells. Keep the existing free-data-honesty/verbatim tone.
- No code/validator change — this is prompt discipline to reduce `phase_7_5` `definitional_drift` frequency.

---

## Testing

- **Unit (`tests/test_recent_closes.py`):** with a fixture `prices` dict whose `ohlcv` CSV has ~15 dated rows: `compute_recent_closes` returns exactly the last 10 on-or-before `trade_date`, most-recent-first, closes verbatim from column 4; a `trade_date` mid-series excludes later dates (on-or-before boundary); fewer than `n` rows returns all available; empty/garbage `ohlcv` → `unavailable=True` with no exception. `format_recent_closes_block` contains `## Recent closes`, the mandate sentence, and renders the `unavailable` note when appropriate.
- **Prompt test (`tests/test_fundamentals_prompt.py`, extend):** assert the new net-debt-discipline sentence is present in `_SYSTEM`.
- **Source-alignment guard test:** a test that parses the SAME fixture `ohlcv` the way `price_date_validator._parse_prices_json` does (col 4) and asserts `compute_recent_closes` emits identical close strings for the overlapping dates — locks the block to the validator's source so a future refactor can't silently diverge them.
- **Real success measure (not a unit test):** fewer `phase_7_1_price_date` (and, for Part 2, `phase_7_5_net_debt`) blocking violations across the next cadence. Note in the run log; do not claim a guarantee.

## Out of scope

- Phase A (reuse `raw/*.json` + enable the existing checkpointer / resume completed LLM stages) — separate spec.
- Phase B (move validators inline + self-correct the failing stage) — separate spec.
- TA files (`raw/technicals*.md`) — already excluded from price-date extraction (`cli/research_validation.py:118-132`), so no pin needed there.
- Changing validator tolerances or the validator logic itself.
- Pinning non-close price levels (52-week high/low, intraday) — YAGNI; the observed failures are specific-date closes.
