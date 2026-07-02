# Pro-deck parity — running progress note

Owner-facing status for the PRO_DECK_GOAL.md work (benchmark ORCL vs Tiger
Brokers deck pp. 63–70, then build techniques A–E). Updated after every step.

## Status: Task 1 DONE · A, B, C DONE — building D

| Step | State | Notes |
|---|---|---|
| Read pro deck pp. 63–70 | ✅ | via pypdf; 7 techniques catalogued |
| ORCL benchmark (fast path) | ✅ | all deterministic blocks run live on ORCL 2026-07-01; sec_filing.md + role prompts inspected |
| Gap analysis doc | ✅ | `2026-07-02-pro-deck-gap-analysis.md` — 5 MISSING, 2 PARTIAL |
| A. Forward-EPS × multiple grid | ✅ | `eps_scenario.py` (15 unit tests, TDD); wired in researcher + Financial-Statement role directive; live-smoked ORCL (compression 13.1x→5.2x) + MSFT |
| B. RPO deep-dive | ✅ | `rpo_backlog.py` deterministic block (SEC XBRL): ORCL $638B, QoQ additions match deck (+317.5/+68/+29.3/+85.4B), RPO÷mcap 1.55x vs MSFT 0.24x/GOOGL 0.11x, AMZN honest n/a; + targeted-excerpt fix in `sec_edgar.py` (waterfall "12%/34%/34%" paragraph now survives truncation); + Financial-Statement RPO discipline. 18 unit tests, TDD |
| C. Capex bridge + FCF inflection + dated catalyst | ✅ | 3 grounded directives: capex-funding-bridge + FCF-trajectory (Financial-Statement role) and a required "## Dated inflection to watch" section (Catalysts role, anchored to the calendar block's Next-expected date). Smoke: ORCL excerpts carry the fiscal-2027 capex-guidance + prepayment prose the directives quote. 5 unit tests, TDD |
| D. Segment trends + call takeaways | ⏳ | directive + deterministic cash-flow QoQ block |
| E. Bear-to-bull reframing voice | ⏳ | structural directive |
| Re-benchmark ORCL + update gap analysis | ⏳ | |

## Key findings so far

- We are NOT yet on par with the deck's 7 signature techniques (5 missing, 2
  partial) — but we exceed it on screening breadth (Z″/M-score/goodwill/refi/DDM/
  intrinsic value/ownership). The gap is forward-looking framing, not rigor.
- ORCL total RPO $638B is free + structured via SEC XBRL companyconcept — the
  deck's centerpiece table is fully reproducible (MSFT $633B, GOOGL $467.6B,
  AMZN honest n/a).
- Bug-class finding: `fetch_latest_filing` truncates at 60K chars → ORCL's
  inline-XBRL 10-K loses ALL MD&A/RPO/segment prose (header metadata alone
  \>60K). Fix planned in B via targeted keyword excerpts from full text.
- Deck p70's QoQ call takeaways reproduce exactly from our quarterly cashflow
  (FCF −1.87B vs −11.48B; OCF +104% QoQ; capex −11.5% QoQ).

## Blockers

None.

## Environment notes (dev clone)

- `openpyxl` was missing from the dev `.venv` (collection error in
  `test_update_research_summary.py`) — installed 2026-07-02.
- Pre-existing, host-specific failure:
  `test_claude_cli_chat_model.py::test_invoke_calls_claude_cli_and_returns_aimessage`
  expects argv `["claude", ...]` but this host resolves the full nvm path.
  Fails on a clean main checkout too — NOT caused by this work.
