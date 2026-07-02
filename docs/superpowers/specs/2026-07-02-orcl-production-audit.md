# ORCL production-run audit — 2026-07-01 (Phase 4 of the free-finish goal)

**Audited:** 2026-07-02 · **Run:** `~/tkresearch/preaudit/2026-07-01-ORCL/` (produced by
production `~/tradingagents` at e9b21e5, `deep=claude-opus-4-8` / `quick=claude-sonnet-4-6`,
both via CLI; QC passed with 0 retries; 48-page branded PDF generated).
**Benchmarks:** (a) FA-101 checklist (per `2026-07-02-fa101-coverage-audit.md` mapping),
(b) Tiger Brokers "30 June FA Outlook" Oracle case study pp. 63–70 (`~/30-jun-fa-outlook.pdf`,
re-read via pypdf for this audit).
**Method:** every dimension checked against `raw/pm_brief.md` evidence + downstream usage in
`analyst_fundamentals.md` / `debate_*.md` / `decision.md` / `decision_executive.md` / the PDF;
key deterministic figures independently re-verified (see §3).

## Verdict: **SATISFIED** — with two code-level hardening fixes shipped (§4)

The ORCL report is on-par-or-beyond the deck's Oracle case study on every
free-data-achievable dimension, every block is sourced or carries an honest n/a, and the
new debt-maturity ladder carries Oracle's real year-by-year schedule, independently
verified verbatim against the live SEC filing. No report content is missing, broken,
thin, or fabricated. The only defects found were **outside the report**: 6 false-positive
MATERIAL blockers in the post-hoc `validation_report.json` (validator regex scope bugs —
all six underlying report claims verified correct) and a directive/file-list mismatch in
the Financial-Statement role. Both fixed with TDD (§4); the regenerated validation report
is clean (0 violations).

## 1. Pro-deck Oracle case study (pp. 63–70) — dimension by dimension

| # | Deck technique (page) | Verdict | Evidence in this run |
|---|---|---|---|
| 1 | Capex funding bridge (p64: $95B = prepayments + $40B financing + OCF/cash) | **MATCH (grounded)** | pm_brief carries the release's verbatim funding excerpts ("raised $43B debt + $5B equity… expects ~$40B in FY27 incl. $20B ATM…", "$75B prepaid/customer-supplied hardware", the Net-Cash-Outlay table $55,663M − $3,345M − $4,592M = $47,726M). analyst_fundamentals §"Capex funding bridge" decomposes FY26 capex from those disclosed cells and marks the un-itemized ~$15.7B "funding split not disclosed" — never fabricated. The deck's exact $20-25B/$40B/$30-35B FY27 split is call-only detail; ours quotes what the release/filing disclose (by design) |
| 2 | FCF inflection + dated catalyst (p65: 10 Sep inflection) | **MATCH** | Deterministic Cash-flow-momentum block: FCF −$2.9B→−$0.4B→−$10.0B→−$11.5B→−$1.9B, latest QoQ OCF +104.4%, capex −11.5% (deck p70's numbers verbatim). "Dated inflection to watch: 2026-09-09 Q1 FY27" section names the one metric (58–64% USD cloud guide); decision.md builds the whole post-catalyst tree on it |
| 3 | Segment YoY/QoQ trend tables (p66: 8 quarters IaaS/SaaS) | **PARTIAL — free-data ceiling (by design)** | Revenue-stream table (Cloud $34.0B +39%, IaaS $18.1B +77%, SaaS $15.9B +11%, Software −1%…) + trend table (IaaS +77% FY vs +93% Q4 = **accelerating**; SaaS stable; Software decline widening) — all release-disclosed comparable periods, honestly marked. The deck's 8-quarter series hand-collates 8 filings; a deterministic multi-release segment parser was evaluated and rejected in the Phase-1 design (no structured free source — segment XBRL facts are dimensioned, not exposed via companyconcept/companyfacts; text-parsing 8 heterogeneous releases per arbitrary ticker is fabrication-prone). The acceleration *conclusion* matches the deck's |
| 4 | RPO vs peers vs market cap (p67) | **MATCH (deterministic, beyond deck)** | XBRL RPO block: total $638.0B (10-K 2026-05-31, = deck), 9-quarter history with additions (+317.5/+68.0/+29.3/+85.4B = deck p67), **RPO÷mcap 1.55x** vs MSFT 0.22x / CRM 0.51x / IBM 0.26x, SAP honest n/a. RPO YoY +363%, ÷TTM revenue 9.47x, ÷cloud revenue 18.8x. Peer set differs from the deck's (MSFT/CRM/SAP/IBM vs deck's hyperscalers) per the classifier — the centerpiece "contracted revenue ≥ peers at a fraction of the market cap" lands identically |
| 5 | RPO conversion/duration waterfall (p68: 12/34/34/20) | **MATCH (quoted verbatim)** | `raw/sec_filing.md` targeted excerpt carries the filing's exact sentence ("…approximately 12 % as revenues over the next twelve months , 34 % over the subsequent month 13 to month 36 , 34 % … month 37 to month 60 and the remainder thereafter") — quoted in analyst_fundamentals, debate (both sides), decision.md §Data freshness, and the "slow burn = durability + years-2-5 dependency" framing appears on both bull and bear sides |
| 6 | Forward-EPS × exit-multiple grid (p69) | **MATCH (deterministic)** | Grid block: consensus FY $8.05 (41 analysts) / +1y $10.92 (39) → $27.30 at +35.7%/yr, priced at 20x/25x/30x + current 24.4x, **compression row 13.1x→5.2x** (deck's 20.65→7.69 pattern). Extrapolation labeled "NOT company guidance". Used downstream: fundamentals' "cheap-if-delivered" read, bull's $200–218 case, bear's "compression only if the path materializes" |
| 7 | Earnings-call takeaways (p70) | **MATCH (quant) / release-grounded (color)** | Quant half verbatim-reproduced (OCF +104.4% QoQ, capex −11.5%, FCF −$11.5B→−$1.9B = deck p70). Color: "Management color (earnings release)" quotes the 8-K Ex-99.1 narrative ($75B prepaid/BYOC — the deck's p70 prepayment theme; Multicloud DB +404%; Oracle Health double-digit guide) with honest "no attributed CEO/CFO quote in the exhibit". Deck's call-only stats (65% of RPO increase prepaid, 1GW Q1 delivery) are transcript-only — no free feed; correctly absent, not faked |
| 8 | **Debt maturity ladder (new, this goal's Phase 1)** | **MATCH — real schedule, independently verified** | See §3. pm_brief block quotes the 10-K note verbatim; Risk section renders the full table (FY27 $7,210M / FY28 $10,145M / FY29 $5,500M / FY30 $7,250M / FY31 $9,750M / Thereafter $90,250M / **Total $130,105M**) + analysis (FY28 heaviest near year, 3.1x cash-covered; 69% of load in "Thereafter" → RPO-conversion dependency); present in the PDF; cited by Conservative risk analyst and both bear-tests. **Not n/a — the real year-by-year ladder** |
| E | Bear-concern → reframe-with-numbers voice | **MATCH** | Three "Top bear concern, tested" sections (one per fundamentals role scope), each two-sided — e.g. "funded, not resolved — a genuine reframe on liquidity, but not a clean rebuttal on leverage"; the FY26 −$23.7B FCF concern is allowed to *stand* (Z″ 0.85, ROIC<WACC, incremental ROIC 5.2%) and becomes the RM's decisive argument. Never spun |

## 2. FA-101 checklist — all sections real, sourced, or honest-n/a

| FA-101 area | Status | pm_brief / report evidence |
|---|---|---|
| §1 top-down, peers | MATCH | classification + peer set with rationale; macro engine runs separately (daily) |
| §2 statements (rev/margins/EPS/BS/CF) | MATCH | accounting-ratios block; net-debt block ($98.25B authoritative + $124.30B economic anchor, reconciled in decision.md #3); cash-flow momentum |
| §3 ratio families (profitability/liquidity/leverage/efficiency/return-of-capital) | MATCH | full table; honest n/a: DIO, CCC (line-item absent in yfinance for ORCL) |
| §4 growth (CAGRs, operating leverage, incremental ROIC, forward consensus) | MATCH | Rev CAGR 10.48%, EPS CAGR 23.83%, op-leverage 1.4x, **incremental ROIC 5.2%** (load-bearing in the decision); FCF CAGR honest n/a |
| §5 valuation (relative + intrinsic + DDM) | MATCH | relative-multiples (peer-median n/a cells honest), intrinsic DCF/EPV/reverse-DCF $91.48 base with **DIVERGE flag addressed** in decision.md (Reconciliation #2 + Caveats), DDM honest n/a (g ≥ CoE, correctly invalid) |
| §6 qualitative (moat, mgmt, governance, capital allocation) | MATCH | Porter-style Competitive-Quality section, all filing-quoted; capital-allocation shift ($206M buybacks vs $55.7B capex) with ROIC-vs-WACC discipline verdict; board independence honestly "not determinable from available free filings" (proxy content not fetched) |
| §7 distress/manipulation screens | MATCH | Z″ 0.85 Distress (structurally explained, not hand-waved), Beneish −2.57 normal, goodwill 146.5%-of-equity elevated flag, refi proxy moderate + **full ladder** |
| §8 consensus/surprise/short interest/ownership | MATCH | surprise history 5-of-8 beat, sentiment block (39 analysts, targets $155–400), 13F holders table, insider net-sell $147.3M with 10b5-1/CEO-transition context; 13D/13G **honest n/a — transient**: "EDGAR full-text search unreachable" during the run; the fetcher re-tested live on 2026-07-02 and returns ORCL 13G rows (fail-open worked as designed) |
| §9 filings | MATCH | 10-K (2026-06-22) fetched + targeted excerpts; 8-K Ex-99.1 release fetched verbatim; 8-K event surface + DEF 14A pointer; Form 4 insider detail |
| §10 segments/backlog/KPIs | MATCH | segment tables (§1 #3), RPO deep-dive (§1 #4/#5); NRR/ARPU/LTV-CAC honestly "not disclosed" |

## 3. Anti-fabrication spot-checks (all passed)

- **Debt ladder vs live SEC:** re-fetched `orcl-20260531.htm` (accession 0001193125-26-277521)
  from sec.gov during this audit — "Fiscal 2027 $ 7,210 / 2028 10,145 / 2029 5,500 / 2030 7,250 /
  2031 9,750 / Thereafter 90,250 / Total $ 130,105" — **verbatim match** with `raw/debt_maturity.json`,
  the pm_brief block, the Risk-section table, and the PDF.
- **RPO waterfall provenance:** decision.md's "~12% / 34% / 34%" traced to `raw/sec_filing.md`
  line 50 (the filing's spaced "12 %" formatting initially hid it from a naive grep — it is there,
  verbatim).
- **Ladder ↔ balance sheet consistency:** FY2027 principal $7,210M = the net-debt block's
  Current Debt $7.20B; refi-proxy's $33,847M = Total − Long-Term Debt (different, broader basis) —
  explicitly disambiguated in the Risk section rather than conflated.
- **EV tie-out:** $508.7B = mcap $410.47B (2,880.471M sh × $142.50) + net debt $98.25B — ties
  across relative-multiples block, decision.md, and exec summary.
- **ND/EBITDA discrepancy handled, not hidden:** RM's 3.22x (yfinance-EBITDA basis) vs
  decision's derived 2.94x (EBIT+D&A from named quarterly cells) — carried as Reconciliation #1
  with the derivation shown.
- **Deck-number cross-checks:** RPO $638B/additions/waterfall %, FCF −$1.9B vs −$11.5B,
  OCF +104%, capex −11.5%, FY26 EPS $7.63 non-GAAP, guidance $8.05/$90B — all equal the deck's
  figures from independent free sources (XBRL / 8-K / 10-K), i.e. the free pipeline reproduces
  the paid deck's evidence base.

## 4. Defects found (outside the report) — fixed with TDD, same day

1. **`validation_report.json` shipped with 6 spurious MATERIAL "blocking" violations** —
   all six underlying report claims verified correct:
   - 5× `net_debt definitional_drift`: the 8-K's supplemental term "**Net Cash Outlay** for
     Capital Expenditures" (quoted verbatim per the earnings-release directive) matched the
     `net cash` position regexes. Fix: `(?!\s+outlay)` lookahead in both patterns
     (`net_debt_validator.py`, Phase 9.2) + regression tests on the exact ORCL prose.
   - 1× `peer_metric wrong_peer_metric`: ORCL's own "Net debt/EBITDA 3.22x" bound to MSFT via
     lookback ("behind MSFT's 46.3%…" — SAP/CRM/IBM slash-rejected, subject outside the window).
     Fix: consumed-comparator guard — a ticker immediately followed by its own inline value
     ("MSFT's 46.3%", "IBM at 3.34x") is no longer a binding target
     (`peer_metric_validator.py`, Phase 9.2) + tests (FP repro + false-negative guard).
   - Adversarially reviewed per repo policy (validator-reviewer agent) before commit.
   - Re-ran the fixed validators against the run: **0 total / 0 blocking**;
     `validation_report.json` regenerated (original archived as `validation_report.orig.json`).
2. **Financial-Statement role directive/file mismatch:** the "Latest-quarter takeaways"
   directive sources call commentary from `news.json`, but `_FILES_FINANCIAL` never included
   it — the designed news-color path could never trigger (the run honestly wrote "no
   earnings-call commentary…"). Fix: `news.json` (≈5KB) added to `_FILES_FINANCIAL` + tests.

**Why no full ORCL pipeline re-run:** the goal's fix-loop triggers on report deficiencies vs
the two benchmarks; none were found — both fixes live outside report-content generation
(post-hoc validator; additive role input). The validator fix was verified end-to-end against
this run's real prose (6 → 0); a fresh ~30-min Opus run would only re-roll LLM variance on an
already-passing report. The news.json wiring takes effect from the next scheduled run onward.

## 5. Residual notes (accepted, not defects)

- Segment 8-quarter series: free-data ceiling by design decision (see §1 #3).
- 13D/13G n/a in this run: transient EDGAR outage, fail-open honest; fetcher verified healthy.
- Deck's MSFT RPO 627B vs our 633B: different sources (deck: Bloomberg/10-Q read; ours: XBRL
  companyconcept) — ours is the structured verbatim figure.
- Goodwill block prints a raw float (`62261000000.0`) — cosmetic; correctly cited as $62,261M
  downstream.
