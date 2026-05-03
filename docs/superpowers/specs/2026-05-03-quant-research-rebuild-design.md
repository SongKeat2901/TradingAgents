# Quant Research Rebuild — Tom-Lee-Style Rigor + Agentic Self-Correction

**Date:** 2026-05-03
**Status:** Draft
**Builds on:** Phases 1-5 of the TradingAgents-under-OpenClaw integration. Phase 5's hybrid claude-CLI architecture is the foundation; this spec extends the agent layer.

## Goal

Replace the current 4-analyst graph with a **data-grounded, quant-rigorous, self-correcting** multi-agent pipeline that produces equity research output with the precision of sell-side reports (Tom Lee / FundStrat style): explicit calculation chains, peer comparison matrices, identified technical levels with rationale, and a mandatory 12-month Bull/Base/Bear scenario table with probabilities and an Expected-Value computation.

## Stakeholder feedback that motivates this spec

> *"Already damn powerful, but I would love more numbers. 12 months bull/bear/neutral price, probability of each, then the EV of the stock in 12 months."*

> *"Quantitative analysis lacking. Example, details of $1.5B deal — how will it improve stock price (like $20m a year, then add EPS by how much, then increase share price by how much)."*

> *"Technical analysis lacking. Major lines need to be studied so that we crash into the bumps. e.g., 60-70K for BTC because previous cycle high was there."*

> *"Will be nice if there's competitor analysis... Visa vs Mastercard. One has how many transactions vs the other. For MSFT closest peers — Mag7 — and matrices for comparison. MSFT cash float is disgusting; how to compare with another Mag7 with tons of debt financing."*

> *"PM final — if PM doesn't agree, push it back for 1 or 2 more iterations? Like our superpower plugin self-learning and correction."*

## Non-goals

- Live trading execution. This stays research-only; hand-off to `ibkr-trader` is explicit.
- Replacing the existing trader / bull-bear / risk-team / PM personas. We keep all 12 personas; we add 2 new nodes (PM Pre-flight, TA Agent) and one deterministic data layer (Researcher).
- Multi-ticker batch runs. One ticker per invocation, as today.
- Streaming partial output. Same fire-and-forget model with PDF on completion.
- Tool-use (bind_tools) inside analyst nodes. We move to pre-fetched data; analysts read instead of fetch.

## Architecture

```
                                 START
                                   │
                                   ▼
        ┌─ PM Pre-flight (Sonnet via claude CLI) ──────────────────┐
        │  - validates ticker, reads memory log for past decisions │
        │  - identifies 2-4 peers for comparison                   │
        │  - sets framing: "what must this run answer"             │
        │  - writes raw/pm_brief.md                                │
        └──────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─ Researcher (Python, no LLM) ────────────────────────────┐
        │  - fetches all data using existing dataflows tools       │
        │  - per-ticker financials (yfinance, alpha_vantage)       │
        │  - per-peer financials (3-5 tickers from PM brief)       │
        │  - news (recent + global), insider transactions          │
        │  - 5-year price/volume history (long lookback for TA)    │
        │  - social/sentiment indicators                            │
        │  - writes raw/{financials,peers,news,prices,social}.json │
        └──────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─ TA Agent (Sonnet via claude CLI) ───────────────────────┐
        │  - reads raw/prices.json (5y history)                    │
        │  - identifies cycle highs/lows, swing points             │
        │  - Fibonacci retracements, volume profile zones          │
        │  - explains why crowds will trade each level             │
        │  - writes raw/technicals.md                              │
        └──────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─ 4 Analysts (Sonnet via claude CLI, parallel-safe) ──────┐
        │  Each reads raw/* (no tool calls). Strong prompts.       │
        │  - Market analyst → analyst_market.md                    │
        │      (consumes technicals.md; banters/extends)           │
        │  - Fundamentals → analyst_fundamentals.md                │
        │      (mandates: deal-math chains, peer matrix,           │
        │       capital structure compare with peers)              │
        │  - News → analyst_news.md (catalyst magnitude estimates) │
        │  - Social → analyst_social.md                            │
        └──────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                Bull / Bear debate (Sonnet, unchanged structure)
                                   │
                                   ▼
                Research Manager (Opus via claude CLI) ⇄ PM retry
                                   │
                                   ▼
                Trader (Sonnet)
                                   │
                                   ▼
                Risk team — Aggressive / Conservative / Neutral (Sonnet) ⇄ PM retry
                                   │
                                   ▼
        ┌─ PM Final (Opus via claude CLI) ─────────────────────────┐
        │  Pass 1: synthesize Bull/Base/Bear scenarios + EV +      │
        │          rating + execution plan                         │
        │  Pass 2: self-correct via QC checklist (in-place)        │
        │  Pass 3: if substantive disagreement, emit retry signal  │
        │          to research_manager OR risk_team (max 1 retry)  │
        │  Final: writes decision.md                               │
        └──────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                                  END
```

## Components

### 1. PM Pre-flight (new, Sonnet)

**Purpose:** opener role for the PM persona. Sets the research mandate before any analysis happens, identifies peers for comparison, and surfaces past lessons from the memory log.

**Reads:**
- `state.company_of_interest`, `state.trade_date`
- `~/.openclaw/data/memory/trading_memory.md` (existing reflection log)
- *(optional)* prior decision.md files for the same ticker if any

**Writes:** `<output_dir>/raw/pm_brief.md` containing:
- Ticker validation (trading day, sector, market cap class)
- **Business-model classification (REQUIRED, may override yfinance sector)** — see below.
- Peer set: 2-4 tickers with rationale ("MSFT's closest peers are GOOG, META, AAPL because all are mega-cap AI infrastructure operators with cloud businesses")
- Past-lesson summary: any prior decision on this ticker or similar pattern
- "What this run must answer": 3-5 specific questions the analysts must address

**Output schema:** structured Markdown with named sections that downstream nodes can grep.

#### Business-model classification (motivated by Flaw 1)

yfinance's sector tag is structurally misleading for some tickers. Example: MARA is tagged "Financial Services / Capital Markets" but is actually a Bitcoin miner; the resulting fundamentals analysis treated mark-to-market BTC holdings as a "loss on sale of investment securities," warping the entire bull/bear framing.

PM Pre-flight MUST emit, alongside the yfinance sector, an explicit **`actual_business_model`** sub-section that:

1. States the actual business model in plain English ("Bitcoin miner with co-located power generation").
2. Calls out any yfinance-tag mismatch.
3. Provides "interpretation rules for downstream analysts" — how to read specific line items given the actual model:

```markdown
## Business model classification

- yfinance sector: Financial Services / Capital Markets
- Actual business model: **Bitcoin miner with co-located power generation**

Interpretation rules for analysts:
- Revenue = (BTC mined × BTC price) + hosting + ancillary services. NOT
  net interest margin, NOT trading P&L.
- "Investment securities" line is largely BTC holdings marked to market.
  The Q4 "loss on sale of securities" is most likely a BTC price mark, NOT
  liquidation of a portfolio.
- Capex is ASIC and power infrastructure spend, NOT capital reallocation.
- "Cash + investments" must be reported with BTC separated from cash.
```

The fundamentals analyst's prompt is updated to read this section and quote the relevant rules in its first paragraph (and abide by them). The bull/bear researchers must respect these rules in their arguments.

### 2. Researcher (new, Python — no LLM)

**Purpose:** deterministic data acquisition. Replaces the bind_tools pattern in current analysts.

**Reads:**
- `<output_dir>/raw/pm_brief.md` (peer list)
- ticker, date from state

**Writes to `<output_dir>/raw/`:**

| File | Source | Content |
|---|---|---|
| `reference.json` | `get_stock_data` close on `trade_date` | Single source of truth for the **reference price** all agents must cite — see below |
| `financials.json` | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | All ticker financials, normalized |
| `peers.json` | Same tools, called per peer ticker | Per-peer financials, indexed by ticker |
| `news.json` | `get_news`, `get_global_news` | Recent + macro news |
| `insider.json` | `get_insider_transactions` | Insider activity |
| `social.json` | `get_news` (social variant) | Social sentiment |
| `prices.json` | `get_stock_data` with 5y lookback | OHLCV + indicators (`get_indicators`) |

#### Reference price — single source of truth (motivated by Flaw 2)

Past output had two prices for MARA on the same date: $11.99 in PM/technical sections, $11.41 in news/sentiment sections (one was the close, the other an intraday quote scraped from a news article). This is forbidden going forward.

`raw/reference.json` schema:

```json
{
  "ticker": "MARA",
  "trade_date": "2026-05-01",
  "reference_price": 11.99,
  "reference_price_source": "yfinance close 2026-05-01",
  "spot_50dma": 12.45,
  "spot_200dma": 14.20,
  "ytd_high": 18.30,
  "ytd_low": 8.92,
  "atr_14": 0.65
}
```

**Mandatory rule for every agent:** every dollar/price citation in any output **must** equal `reference_price` for "$TICKER at $trade_date" or use one of the named keys above (e.g. `spot_50dma`). Any other price quoted (e.g. from a news article) must carry an explicit time qualifier — `"$11.41 (article from 2026-04-29)"`. PM Pass-2 QC fails the report if a bare `"<TICKER> at <date>"` price citation diverges from `reference_price` by more than $0.01.

Implementation: a Python module `tradingagents/agents/researcher.py` with a single function `fetch_research_pack(state, peers) -> None` that writes the raw/ folder. No LangGraph node-class needed (uses existing tool functions directly, not bind_tools).

### 3. TA Agent (new, Sonnet)

**Purpose:** technical analysis ownership. Identifies the levels that matter and explains crowd psychology around them.

**Reads:**
- `raw/prices.json` (5y OHLCV)
- `raw/pm_brief.md` (for context — what's the run's framing?)

**Mandated output structure** (`raw/technicals.md`):

```markdown
## Major historical levels

| Level | Price | Type | Why crowds trade here |
|---|---|---|---|
| Prior cycle high | $560 | Resistance | 2024 swing high; large stop cluster above |
| 200-day SMA | $487 | Support | Long-term trend; institutional rebalancing zone |
| Fib 0.618 retrace | $446 | Support | Of $200→$560 leg; algo bounce zone |
| ...3-7 levels total |

## Volume profile zones

- Heaviest accumulation: $480-$510 (40% of YTD volume) — strong support
- Volume gap: $410-$430 (3% of YTD volume) — slip-through zone

## Current technical state
[narrative on RSI, MACD, moving average stack, divergences]

## Setup classification
[breakout / breakdown / consolidation / distribution / accumulation]

## Asymmetry
- Upside to next major resistance: $560 → +8%
- Downside to next major support: $446 → -14%
- Reward/risk: 0.6:1 (unfavourable)
```

### 4. Existing 4 Analysts (refactored, Sonnet)

Same names (market, fundamentals, news, social), but two changes:

**Change 1: data source.** No more `bind_tools`. Each analyst is given the relevant raw/ files as context in its prompt.

**Change 2: prompt rigor.** Each analyst's system prompt mandates concrete output structure:

#### Fundamentals analyst — additions
- **Deal math chain** (required when any deal/announcement is in news): `Deal size $X → annual revenue impact $Y (cite source assumptions) → EPS delta +$Z → at current 22x P/E this implies +$W per share`.
- **Peer comparison matrix** (always included; pulled from `raw/peers.json`):

```markdown
| Metric | Ticker | Peer1 | Peer2 | Peer3 | Ticker rank |
|---|---|---|---|---|---|
| Revenue (TTM) | $245B | $310B | $134B | $402B | 3rd |
| Revenue growth YoY | 12% | 8% | 22% | 5% | 2nd |
| Operating margin | 42% | 38% | 28% | 33% | 1st |
| Net debt / EBITDA | -1.2x | -0.4x | 0.8x | -0.9x | 1st (best) |
| Cash + ST investments | $98B | $82B | $24B | $128B | 2nd |
| P/E (TTM) | 31x | 24x | 28x | 29x | premium |
```

- **Capital structure framing** required (reflects stakeholder's "MSFT cash is disgusting vs Meta debt" point): explicit comparison of leverage / cash to peers.
- **Sanity-check section (motivated by Flaw 4).** Mandate a "Sanity check on reported numbers" section that audits any computed ratio that looks implausible:

```markdown
## Sanity check on reported numbers

| Metric | Reported | Implied math | Plausible? |
|---|---|---|---|
| Interest expense Q4 | $11M | On $3.65B debt = 1.4% effective rate | ❌ Implausibly low — likely excludes capitalized interest or convertibles |
| Cash + investments | $1.2B | Includes BTC holdings? | Flag for separation |
| Q4 loss on securities | -$200M | Per business-model rules, this is a BTC mark | Reclassify; not portfolio liquidation |
```

Any item flagged ❌ or "Flag for ..." must be addressed by the bull/bear researchers in debate. PM Pass-2 QC verifies that flagged items received responses.

- Final section: "What management needs to prove" — 3 falsifiable hurdles.

#### Market analyst — additions
- Reads `raw/technicals.md` (TA Agent output) as input.
- Required to **either accept TA Agent's level identification or challenge specific levels** with rationale.
- Required to map the technical setup onto a **trading playbook**: "if SPY breaks $500 on volume, expected next stop $487 200-DMA; if it holds, base for $510-520 retest."

#### News analyst — additions
- For every catalyst, mandated **magnitude estimate** ("Fed cut → +1% to S&P fair value via duration math; estimate $5 SPY upside").
- Cross-reference: cite when any peer in `raw/peers.json` is mentioned in the news context.

#### Social analyst — addition
- Sentiment must include numbers: "X% bullish per FinTwit volume in past 7d", not just "bullish."

### 4b. Bull / Bear researchers (existing, Sonnet — prompt rigor upgrade)

Motivated by Flaw 6: in past output, the bull leaned on Tesla/NextEra/Blackstone analogies that didn't survive scrutiny while burying the genuinely strong asset-value point (505 MW dispatchable power has scarcity value independent of execution). Bull/Bear prompts gain three explicit rules:

1. **Lead with your strongest argument.** First paragraph names the single most load-bearing fact for your case. No analogies in the lede.
2. **Analogies must survive scrutiny.** If you cite Tesla / NextEra / Blackstone / any precedent, you must include: (a) the relevant metric for the comparable, (b) why the analogy holds for this specific ticker, (c) the disanalogy and why it doesn't break the case. If you can't do all three, drop the analogy.
3. **Quantify the asymmetry.** Conclude with a specific dollar/percentage outcome AND a probability ("Bull case: $560 in 12 months, ~30% probability conditional on Q4 capex ROI clearing"). Vague directional claims ("upside is meaningful") are rejected by the Research Manager and require revision.

### 5. Research Manager (existing, Opus — gains retry-input handling)

Unchanged role: judges Bull/Bear debate, drafts investment plan with preliminary scenarios.

**New input:** when re-invoked via PM retry, the state contains `pm_feedback` text. The RM prompt is updated to read that field and address it explicitly.

### 6. Risk team (existing, Sonnet — gains retry-input handling)

Same change as RM: when re-invoked, must address `pm_feedback`.

### 7. PM Final (existing, Opus — gains retry loop + mandated structure)

**Three-pass model in V1:**

**Pass 1 — Draft synthesis.** PM produces a draft `decision.md` containing two mandatory new sections plus the existing memo:

```markdown
## Inputs to this decision (motivated by Flaw 7)

- Reference price: $<reference_price> (<reference_price_source>)
- Peers compared: <ticker_list with one-line rationale>
- Past decisions referenced: <ticker> <date> (<rating>; outcome <±pct>%, alpha <±pct>%)
  — invoked to argue <…>
- Memory-log lessons applied: <bullets with source line refs>
- Catalysts in window: <upcoming events with dates>
- Data freshness: <financials period>, <news through date>

## 12-Month Scenario Analysis

| Scenario | Probability | 12-Mo Price Target | Return | Key drivers |
|---|---:|---:|---:|---|
| Bull | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Base | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Bear | <pct>% | $<price> | <±pct>% | <named events / metrics> |

**Expected Value:** <calculation> = $<EV> (<±pct>% from spot $<spot>)
**Rating implication:** <BUY/HOLD/SELL> (<one-line bridge from EV to rating>)
```

The Inputs section makes `decision.md` self-contained — a stakeholder can read it cold without the analyst reports and understand the framing.

**Pass 2 — Self-correction QC checklist.** PM applies hard rules in-place. The full checklist (with the additions motivated by Flaws 2, 3, 5, 8):

1. Probabilities sum to exactly 100%.
2. All three price targets are specific dollar values (not ranges).
3. Each scenario lists at least one named, falsifiable catalyst.
4. Rating logically derives from EV.
5. Execution triggers are falsifiable (named price / level / date).
6. **(Flaw 8) Re-entry / upgrade triggers must be reachable in at least one scenario in the table.** Example: if Bull peaks at $14 but `decision.md` says "re-enter below $18," that's inconsistent — either revise the trigger or revise the scenario.
7. **(Flaw 2) Every bare `<ticker> at <trade_date>` price citation matches `reference_price` ± $0.01.** Other prices (article quotes, intraday) must carry an explicit time/source qualifier.
8. **(Flaw 3) Every cited analyst position has a verbatim ≤ 30-word quote attributed by section.** Statements like "Neutral's math, applied honestly, supports Sell" require a direct quote of the math from Neutral's report. If the cited claim is not in the source, the synthesis is invalid — re-synthesize.
9. **(Flaw 5) Cross-section numerical consistency.** Compare the same numerical claim (cash runway, target price, percentage move, etc.) across all analyst reports + debate transcripts. Any claim that appears with different values in different sections must be reconciled in `decision.md` under a "Reconciliation" subsection.
10. **(Flaw 4) Sanity-check flags from the fundamentals analyst are addressed** in either the bull/bear debate or the trader's plan. Flagged items cannot be silently ignored.
11. **Inputs section** (Flaw 7) is present and complete in `decision.md`.
12. Peer comparisons cite specific numbers.
13. All claimed numbers trace back to `raw/*.json` data.

If the draft fails any check, PM revises in-place and re-applies. Capped at one self-correction loop (always single LLM call total — one draft + one revise = 2 calls, reusable framework but capped to keep cost predictable).

**Pass 3 — Push-back retry (optional, max 1).** If PM has substantive disagreement with upstream synthesis, emits structured retry:

```json
{
  "approval": "retry",
  "target": "research_manager" | "risk_team",
  "feedback": "<specific instruction>"
}
```

State is mutated: `pm_retries += 1`, `pm_feedback = <text>`. LangGraph routes back to the target node. Target reruns with feedback in its prompt. Then comes back to PM Final.

If `pm_retries == 1`, PM cannot push back further on the second pass. It accepts the best-available output and notes remaining concerns in decision.md under "Caveats from PM."

## Data folder contract

Per-run directory now has both raw/ and the existing top-level reports:

```
<output_dir>/
├── raw/                                  ← new: shared input data
│   ├── pm_brief.md                       (PM pre-flight output)
│   ├── financials.json                   (ticker financials)
│   ├── peers.json                        (per-peer financials)
│   ├── news.json
│   ├── insider.json
│   ├── social.json
│   ├── prices.json                       (5y OHLCV + indicators)
│   └── technicals.md                     (TA Agent output)
├── analyst_market.md                     ← existing, refactored
├── analyst_fundamentals.md
├── analyst_news.md
├── analyst_social.md
├── debate_bull_bear.md
├── debate_risk.md
├── decision.md                           ← existing, with new mandated structure
├── state.json
└── research-<date>-<ticker>.pdf          ← Phase 5 artifact, unchanged
```

## State extensions

`AgentState` gains the following keys (all optional, default-None):

| Key | Type | Set by | Read by |
|---|---|---|---|
| `pm_brief` | str | PM Pre-flight | All downstream agents (in their prompts) |
| `peers` | list[str] | PM Pre-flight | Researcher |
| `raw_dir` | str | Researcher (after writing) | All agents from TA onward |
| `technicals_report` | str | TA Agent | Market analyst, all downstream |
| `pm_feedback` | str \| None | PM Final on retry | Research Manager / Risk team on rerun |
| `pm_retries` | int | LangGraph state machine | Conditional edge that caps retries |

## Conditional edges

Two new edges around PM Final:

```python
def pm_decision_router(state):
    """Decide whether PM is done, needs to retry, or hits the cap."""
    if state.get("pm_retries", 0) >= 1:
        return END               # cap reached, ship as-is
    if state.get("pm_approval") == "retry":
        return state["pm_retry_target"]  # "Research Manager" | "Risk team"
    return END

workflow.add_conditional_edges(
    "Portfolio Manager",
    pm_decision_router,
    {"Research Manager": "Research Manager", "Risk team": "Aggressive Analyst", END: END},
)
```

(Risk team retry routes back to the Aggressive Analyst node since that's how the existing risk debate loop enters.)

## Cost shape

Estimated per-run LLM call counts:

| Phase | Calls | Model |
|---|---:|---|
| PM Pre-flight | 1 | Sonnet |
| Researcher | 0 | (Python) |
| TA Agent | 1 | Sonnet |
| Market analyst | 1 | Sonnet |
| Fundamentals analyst | 1 | Sonnet |
| News analyst | 1 | Sonnet |
| Social analyst | 1 | Sonnet |
| Bull/Bear debate (1 round) | 2 | Sonnet |
| Research Manager | 1 (or 2 if retried) | Opus |
| Trader | 1 | Sonnet |
| Risk team (1 round) | 3 (or 6 if retried) | Sonnet |
| PM Final | 1-2 | Opus |
| **Total — best case** | ~14 | mixed |
| **Total — worst case (full retry)** | ~21 | mixed |

Runtime estimate at the existing 30s pacing: ~12-18 min.

## Migration path from current code

Implementation order, each phase independently shippable:

1. **Researcher (data fetcher)** — new file. Stages raw/ but downstream still uses bind_tools. Verify raw/*.json content is correct.
2. **PM Pre-flight** — new node. Adds peer determination. Writes pm_brief.md. Researcher consumes the peer list.
3. **TA Agent** — new node. Writes technicals.md. Market analyst still works as today (ignores it for now).
4. **Refactor each analyst** to read raw/ instead of bind_tools, one at a time:
   - Market analyst (now consumes technicals.md too)
   - Fundamentals analyst (gains peer-matrix prompt)
   - News analyst (gains catalyst-magnitude prompt)
   - Social analyst (gains numerical-sentiment prompt)
5. **PM Final** new prompt with mandatory scenario table + EV.
6. **PM correction loop** — self-correction first (Pass 2), then push-back retry (Pass 3).
7. **Tighten Research Manager / Risk team** for retry feedback handling.

Phase 5 graph regression tests pass at every step; the existing graph keeps producing decision.md throughout.

## Test strategy

| Concern | Test |
|---|---|
| PM Pre-flight emits a valid peer list for known tickers | Unit: stub the LLM, assert the brief Markdown structure and peer list extraction. |
| Researcher writes complete raw/ folder | Unit: stub yfinance/alpha_vantage tools; assert each expected JSON file exists and has expected schema. |
| TA Agent's level identification is structured | Unit: stub Sonnet output; assert `raw/technicals.md` has the required sections (Major Levels, Volume Profile, Setup Classification). |
| Each analyst reads raw/ and produces structured output | Unit: stub Sonnet; pass canned raw/; assert markdown sections present. |
| PM Final scenario table is mandated | Unit: stub Sonnet to produce output without scenarios; assert the QC self-correction triggers. |
| PM retry pushes back to Research Manager once | Integration: stub PM to emit retry signal; verify Research Manager re-runs with `pm_feedback` and PM evaluates again; verify second retry is capped. |
| End-to-end SPY/MSFT/MARA on real OAuth | Manual smoke after deploy. Compare quality of `decision.md` and `analyst_fundamentals.md` to current Phase 5 output. |

## Out of scope (deferred to future specs)

- Tool-bound analysts (data fetched within the analyst node by the LLM choosing). The pre-fetch + read pattern is cleaner.
- Cross-ticker portfolio reasoning (analyzing 2 tickers in one run).
- Real-time data via WebSockets / live market feeds.
- Backtesting harness — generating decisions across a calendar range and comparing to actual price action.
- Reflection-feedback loop to PM Pre-flight from realised outcomes (currently the memory log is read but not auto-updated; that's `ta.reflect_and_remember` which we deferred from Phase 1).
- Automatic peer set learning (today PM names peers; over time the system could observe which peers' data was actually used and refine).
- Hand-off integration into `ibkr-trader` for execution.

## Open questions

- **Q: Where does the PM brief surface to downstream agents?** Currently I propose passing it via state.pm_brief, included in each agent's prompt. Alternative: agents read `raw/pm_brief.md` directly via file I/O. State approach is more LangGraph-native; file approach matches the rest of raw/. Recommend **file approach** for consistency, but it means agents need a path-reading helper. To resolve during implementation.
- **Q: Should the PM correction loop's "self-correction" pass always run, or only when QC checklist fails?** Always-on is safer but adds 1 LLM call per run. Conditional is cheaper but might leak shoddy first-drafts. Recommend **always-on** in V1; profile and make conditional in V2 if cost matters.
- **Q: How does TA Agent know what "5y of price history" actually contains for a given ticker?** Some tickers IPO'd recently (e.g., MARA's float at $407 has limited deep history). Recommend the Researcher always pulls "max available" and TA Agent reasons over what's there. To document in TA Agent's prompt.
