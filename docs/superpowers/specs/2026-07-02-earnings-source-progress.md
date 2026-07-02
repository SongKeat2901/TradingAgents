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
| 4. Live smoke: ORCL ($40B financing, 27–29% guidance, RPO) + honest-n/a name | pending |
| 5. Merge → main → push | pending |
| 6. STRETCH: FMP free-tier earnings-call transcript | pending |

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
- Also fixed a pre-existing host-dependent test (`94d307c`):
  `test_claude_cli_chat_model` cmd assertion depended on whether `claude`
  was on the pytest env's PATH.

## FMP stretch verdict

- (not yet attempted)
