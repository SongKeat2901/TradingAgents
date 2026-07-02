# Earnings-source progress — SEC 8-K press release + FMP stretch (2026-07-02)

Tracking doc for `EARNINGS_RELEASE_GOAL.md`: add the free SEC 8-K earnings press
release (Exhibit 99.1) as a sourced input (guidance + capex-funding structure +
management color), then optionally the FMP free-tier call transcript.

## Status

| Step | State |
|---|---|
| 1. Fetcher: latest item-2.02 8-K → Ex-99.x → text + targeted excerpts | ✅ shipped (`80c05ca`) |
| 2. Researcher wiring: raw/earnings_release.{json,md} + pm_brief block | ✅ shipped (`ee1588d`) |
| 3. Role citations: Fin-Statement (guidance/funding) + Catalysts (mgmt quotes) | ✅ shipped |
| 4. Live smoke: ORCL ($40B financing, 27–29% guidance, RPO) + honest-n/a name | ✅ passed |
| 5. Merge → main → push | ✅ done (`e2ae396` on main) |
| 6. STRETCH: FMP free-tier earnings-call transcript | ❌ skipped — needs a paid feed (see verdict) |

## Evidence log

- **Fetcher shipped (`80c05ca`, TDD)**: `fetch_earnings_release` +
  `format_earnings_release_block/md` in `sec_edgar.py`; 16 new unit tests
  (doc picker, happy path, look-ahead guard, honest n/a for no-results-8-K
  names, index/exhibit fail-soft, truncation-vs-excerpts contract, block
  formatting). Full suite 947 passed.
- **Researcher wiring shipped (`ee1588d`, TDD)**: raw/earnings_release.json
  always written (records honest-n/a too), raw/earnings_release.md only when
  there is prose, pm_brief gets the release block; own try/except (fail-open,
  3 wiring tests incl. raise-path).
- **Role citations shipped (TDD)**: Financial-Statement quotes guidance +
  capex/financing funding structure verbatim-or-'not disclosed' (capex bridge
  now release-sourced, not call-only); Catalysts gets a required
  '## Management color (earnings release)' section (attributed CEO/CFO quotes
  only). Both roles read raw/earnings_release.md. Suite: 953 passed.
- **Live smoke passed (real EDGAR, 2026-07-02):**
  - **ORCL**: found the 2026-06-10 8-K (items 2.02,8.01,9.01), exhibit
    `orcl-ex99_1.htm`, 39.6K chars. pm_brief block verified to carry: Q1-FY27
    guidance "Total Revenues are expected to grow from 27% to 29%", the
    "$40 billion through a combination of debt and equity financing including
    its previously announced $20 billion at-the-market equity issuance", and
    RPO "$553 billion to $638 billion" (head snippet). NOTE: ORCL's actual
    Ex-99.1 prints NO attributed CEO/CFO quotes (the goal doc over-promised
    that detail) — management color there is an unattributed "AI Market and
    Technology Evolution" narrative; the Catalysts role's honest fallback
    covers it.
  - **MSFT**: 2026-04-29 8-K Ex-99.1 — CEO-quote excerpt captured verbatim
    ("…said Satya Nadella, chairman and chief executive officer…"). This
    smoke caught a real bug (fixed `083a2f7`): the generic 'expect' keyword
    exhausted the excerpt budget before the exec-quote keywords ran.
  - **BABA** (foreign filer, no results-8-K): honest n/a — "no item-2.02
    (results) 8-K on/before trade_date". No fabrication path.
- Also fixed a pre-existing host-dependent test (`94d307c`):
  `test_claude_cli_chat_model` cmd assertion depended on whether `claude`
  was on the pytest env's PATH.

## FMP stretch verdict — needs a paid feed

Two independent blockers (2026-07-02):

1. **No `FMP_API_KEY` in the environment** — the goal says skip rather than
   block (and never hardcode/commit a key). Nothing was wired.
2. **The free tier doesn't include transcripts anyway.** FMP's current
   packaging puts earnings-call transcripts on the **Ultimate tier
   (~$149/mo)**; the free plan is a 250-requests/day EOD sandbox, and
   Starter/Premium don't carry transcripts either (per FMP's pricing docs
   and third-party review, findmymoat.com — FMP's own pricing page 403s
   robots). So even with a free key, `/api/v3/earning_call_transcript` would
   not serve the full Q&A.

**Conclusion: full call-transcript Q&A needs a paid feed.** Not scraping
Motley Fool / Seeking Alpha (ToS). Mitigation shipped in this same unit: the
8-K press release now covers guidance, funding structure, RPO, and (where
printed) management quotes for free — the remaining uncovered slice is the
analyst Q&A back-and-forth only.

## What shipped (summary for the phone)

- `sec_edgar.fetch_earnings_release()` — latest item-2.02 8-K → Ex-99.1 →
  stripped text + targeted excerpts (guidance / CEO+CFO quotes / RPO /
  financing / capex / expects). Fail-soft n/a everywhere.
- Researcher writes `raw/earnings_release.{json,md}` + a `## Latest earnings
  release (SEC 8-K Ex-99.1)` pm_brief block (fail-open).
- Financial-Statement role: guidance + capex/financing funding structure
  quoted verbatim-or-'not disclosed' (capex bridge now release-sourced).
  Catalysts role: required `## Management color (earnings release)` section
  (attributed quotes only).
- Live-smoked ORCL ($40B financing + $20B ATM, 27–29% guidance, RPO
  $553B→$638B), MSFT (Nadella quote verbatim — smoke caught + fixed an
  excerpt-budget bug), BABA (honest n/a). Unit suite: 954 passed.
