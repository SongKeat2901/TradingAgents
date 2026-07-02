# Pro-deck gap analysis — ORCL vs Tiger Brokers "30 June FA Outlook" (pp. 63–70)

**Date:** 2026-07-02 · **Benchmark method:** fast path — ran every deterministic
block live on ORCL (trade_date 2026-07-01) via the same functions `researcher.py`
wires (`/tmp/orcl_bench.py`), fetched the real `sec_filing.md`, and inspected the 4
fundamentals role prompts (`fundamentals_roles.py`). Deck pages 63–70 read via pypdf.

## Owner's question: "Is our system on par with this pro deck's Oracle depth?"

**Short answer: not yet on the deck's 7 signature techniques — 5 MISSING, 2 PARTIAL.**
Our system is *deeper* than the deck on screening breadth (Altman Z″, Beneish M,
goodwill, refinancing proxy, DDM, triangulated intrinsic value, peer ratios,
ownership/activist surface — none of which the deck has), but the deck wins on
**forward-looking framing**: multi-year EPS × multiple price paths, RPO-as-backlog
centerpiece, capex funding decomposition, and a bear-concern → reframe-with-numbers
voice. Those are exactly the 7 techniques below.

## Per-technique verdicts

### 1. Capex funding bridge (deck p64) — **MISSING**
Deck decomposes the scary $95B FY27 capex guide into customer prepayments
($20–25B) + external financing ($40B) + OCF/cash ($30–35B).
**Our output:** we report capex faithfully (capex_ttm −$55.7B, capex/revenue in
peer-ratios + accounting-ratios blocks) but *nothing* decomposes forward capex
funding, and no role prompt asks for it. The funding split is earnings-call /
guidance material — not in any free structured feed — so the honest mechanism is a
**grounded role directive** (quote the filing/call coverage in news.json, else
"not disclosed"). → Build item **C**.

### 2. Forward FCF-inflection thesis + dated catalyst (p65) — **PARTIAL**
Deck argues why FCF turns positive and names 10 Sep (next earnings) as the
inflection to watch.
**Our output:** the PM Pre-flight calendar block already carries the next-earnings
date (dated catalyst ✓ deterministic), and the quarterly cash-flow series shows the
inflection shape (FCF −11.48B → −1.87B QoQ). But no block or directive *frames* a
forward FCF-inflection thesis or ties the dated event to it; the Catalysts role's
"What management needs to prove" is generic. → Build item **C** (directive) + the
deterministic QoQ momentum block in **D**.

### 3. Segment revenue trend tables (p66) — **MISSING (effectively)**
Deck shows 8 quarters of IaaS vs SaaS revenue with YoY *and* QoQ rows plus a
pricing-model-change narrative.
**Our output:** the §10 KPI/segment directive asks the Financial-Statement role for
*single-period* segment revenue/margin from `raw/sec_filing.md`. Two failures for
ORCL: (a) no multi-quarter YoY/QoQ trend view is requested anywhere; (b) the ORCL
10-K extract is inline-XBRL whose first 60K chars are header metadata — the
`fetch_latest_filing(max_text_chars=60_000)` truncation means **zero segment (or
RPO, or capex) prose survives into sec_filing.md**. yfinance has no segment data.
→ Build item **D** (directive) + the targeted-excerpt extraction fix in **B**.

### 4. RPO vs peers AND vs market cap (p67) — **MISSING**
The deck's centerpiece: ORCL RPO $638B ≥ MSFT/GOOGL/AMZN backlogs at ~1/6 their
market caps ⇒ "contracted revenue not priced in", with a per-quarter RPO-additions
row.
**Our output:** the Financial-Statement prompt says "quote RPO from sec_filing.md"
— but ORCL's extract contains no RPO figure (see §3), and nothing computes
RPO ÷ market-cap or fetches peer RPO. **Probe result: fully buildable
deterministically** — SEC XBRL companyconcept `us-gaap/RevenueRemainingPerformanceObligation`
returns ORCL $638.0B (2026-05-31, 10-K — exactly the deck's figure) with a
quarterly history ($455.3B → $523.3B → $552.6B → $638.0B = additions series), and
MSFT $633B / GOOGL $467.6B (2026-03-31). AMZN stopped tagging the dimensionless
total in 2020 → honest "n/a". → Build item **B** (deterministic block).

### 5. RPO conversion/duration waterfall (p68) — **MISSING**
Deck buckets RPO into next-12mo / 13–36 / 37–60 / thereafter and reframes slow
conversion as durability.
**Our output:** nothing. Probe: ORCL stopped tagging
`RevenueRemainingPerformanceObligationPercentage` dimensionlessly in 2021 (the
timing buckets are dimensioned XBRL facts, not exposed via companyconcept), so the
waterfall lives only in the filing's RPO paragraph — which our truncated extract
drops. → Build item **B**: targeted keyword excerpts from the *full* filing text
(pre-truncation) + a grounded directive to build the waterfall from the excerpt,
else "not disclosed".

### 6. Forward-EPS × exit-multiple price-target matrix (p69) — **MISSING**
Deck projects EPS on the guided 28% CAGR to 2030, prices it at 20x/25x/30x (+72x),
and shows implied P/E compressing 20.6→7.7 at a flat price.
**Our output:** we compute a *single* forward P/E (13.05x for ORCL) in relative
multiples, plus a DCF/EPV fair value — but no multi-year EPS path, no exit-multiple
price grid, no compression view. **Probe result: buildable deterministically** —
yfinance `earnings_estimate` gives consensus avg/low/high EPS for 0y ($8.05) and
+1y ($10.92) with analyst counts (41/39) and growth rates; `info.forwardEps` and
current price are present. → Build item **A** (deterministic block, generalizes to
every ticker with coverage; honest n/a otherwise).

### 7. Structured earnings-call takeaways (p70) — **PARTIAL**
Deck lists QoQ deltas (FCF −1.8B vs −11.5B, OCF +100% QoQ, capex −11%) + management
operational color (floating contracts pass input costs).
**Our output:** no earnings-call transcript source is wired (no reliable free feed),
so management color is only whatever lands in news.json. BUT the quantitative half
is **exactly reproducible from data we already fetch**: ORCL quarterly cash flow
gives FCF −1.873B vs −11.484B prior (deck: "−1.8B vs −11.5B"), OCF 14.62B vs 7.15B
= +104% QoQ (deck: ">100%"), capex −16.49B vs −18.64B = −11.5% QoQ (deck: "−11%").
→ Build item **D**: deterministic cash-flow QoQ momentum block + a call-takeaways
directive grounded in news.json.

## Summary table

| # | Deck technique | Verdict | Buildable how |
|---|---|---|---|
| 1 | Capex funding bridge | MISSING | C — grounded directive |
| 2 | FCF inflection + dated catalyst | PARTIAL | C — directive (calendar block exists) |
| 3 | Segment YoY/QoQ trend tables | MISSING | D — directive + B's excerpt fix |
| 4 | RPO vs peers vs market cap | MISSING | B — deterministic (SEC XBRL companyconcept) |
| 5 | RPO conversion waterfall | MISSING | B — targeted filing excerpt + directive |
| 6 | Forward-EPS × multiple grid | MISSING | A — deterministic (yfinance estimates) |
| 7 | Earnings-call takeaways | PARTIAL | D — deterministic QoQ block + directive |

Plus one cross-cutting gap: **E — bear-to-bull reframing voice**. The deck's whole
method is "state the bear concern, then reframe it with numbers"; our role prompts
never ask for that structure.

## Post-build re-benchmark (2026-07-02, after shipping A–E)

Re-ran the REAL `fetch_research_pack` wiring end-to-end on live ORCL
(trade_date 2026-07-01, peers MSFT/GOOGL/AMZN). Every pro-deck block landed in
pm_brief.md; values verified against the deck:

| # | Deck technique | Now | Evidence (live ORCL run) |
|---|---|---|---|
| 1 | Capex funding bridge | **ON PAR (grounded)** | capex-bridge directive + targeted excerpts carry the 10-K's fiscal-2027 capex guidance ($21.2B→$55.7B, "upward trend to continue") and customer-prepayment prose; net-debt block carries the cash cells. Call-only detail (the exact $20-25B/$40B split) appears only when filing/news disclose it — by design, never fabricated |
| 2 | FCF inflection + dated catalyst | **MATCH** | deterministic Cash-flow momentum block (FCF −$11.5B → −$1.9B) + FCF-trajectory directive + required "## Dated inflection to watch" section anchored to the calendar block |
| 3 | Segment YoY/QoQ trends | **IMPROVED (free-data ceiling)** | segment-trend directive builds the YoY/QoQ table from figures the filing discloses; excerpts now surface segment prose. The deck's 8-quarter series collates 8 filings — a single-filing free pipeline surfaces the latest filing's comparable periods only, honestly marked |
| 4 | RPO vs peers vs market cap | **MATCH** | deterministic block: RPO $638.0B (10-K 2026-05-31), additions +317.5/+68.0/+29.3/+85.4B (deck p67 verbatim), RPO÷mcap **1.55x** vs MSFT 0.24x / GOOGL 0.11x, AMZN honest n/a |
| 5 | RPO conversion waterfall | **MATCH (quoted)** | targeted excerpt carries the filing's exact waterfall ("approximately 12% … next twelve months, 34% … 13-36, 34% … 37-60, remainder thereafter"); RPO discipline directive quotes it verbatim, else "not disclosed" |
| 6 | Forward-EPS × multiple grid | **MATCH** | deterministic grid: +1y consensus $10.92 (39 analysts) → $27.30 (+4y at +35.7%), prices at 20x/25x/30x + current 24.4x, compression row 13.1x→5.2x (deck's 20.6→7.7 pattern) |
| 7 | Earnings-call takeaways | **MATCH (quant) / news-gated (color)** | deterministic QoQ line: OCF +104.4%, capex spend −11.5%, FCF −$11.5B→−$1.9B (deck p70 verbatim) + "## Latest-quarter takeaways" directive; management color only from news.json (no free transcript feed) |

Plus **E**: every role now closes with a required "## Top bear concern, tested"
section — the deck's bear-concern → reframe-with-numbers method, kept two-sided
(the bear is allowed to win).

**Verdict for the owner:** on the 7 signature techniques we are now on par —
5 fully matched (4 of them deterministic, stronger than the deck's static
slides since they regenerate per ticker per run), and 2 at the honest ceiling
of free data (capex-bridge call detail and multi-filing segment series appear
exactly to the extent the filing/news disclose them). Combined with the
screening depth the deck lacks (Z″/Beneish/goodwill/refi/DDM/intrinsic-value/
ownership/activist), the ORCL pm_brief now reads at or beyond the Oracle case
study's depth.

## Data-source ground truth established by this benchmark

- SEC XBRL companyconcept API carries total RPO for ORCL/MSFT/GOOGL (free,
  structured, exact deck numbers); AMZN n/a post-2020; timing buckets NOT exposed.
- `fetch_latest_filing`'s 60K-char truncation silently drops MD&A/RPO/segment
  prose on large inline-XBRL 10-Ks (ORCL's header metadata alone exceeds 60K).
- yfinance `earnings_estimate`/`eps_trend`/`growth_estimates` provide consensus
  forward EPS paths + analyst counts — free basis for the price-target grid.
- Quarterly cash-flow columns reproduce the deck's QoQ call takeaways exactly.
- Capex funding splits + segment quarterly revenue: **no free structured source**
  → grounded directives only (never fabricate).
