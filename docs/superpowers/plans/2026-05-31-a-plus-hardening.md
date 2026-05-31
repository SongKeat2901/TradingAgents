# Goal: tune the whole stack so the researcher produces A+ reports natively

Set 2026-05-31. Driving condition (session goal / Stop hook):
> researcher tuned for accurate data — no falsified/hallucinated data, no missing
> data, correct cover-page wording, customer-facing report with no leaked agent
> instructions. Manually audit each report after research, one at a time, retune
> until A+, then register. Ignore runs older than 1 week. All fresh (≤1 week) runs
> must be A+. Remove all failed runs. Register only A+ reports.

"Fresh" = trade_date ≥ 2026-05-24 (the 05-26 / 05-28 / 05-29 cohorts, 15 runs).
Older (≤ 2026-05-23) → ignore entirely.

## Audit baseline (2026-05-31, full 35-run sweep — see chat)
Dominant failure = **Tier 11 peer-ratio fabrication** (11 of 12 F's): invented/inflated
peer Forward & TTM P/E, peer net-debt cells, often falsely cited "per raw/peer_ratios.json".
Second mode (AAPL 05-29): fabricated "Note N" citations to an XBRL stub with no readable prose.

Fresh-cohort grades: A+ = GOOGL, AMKR, MSTR, SOUN, STM, ONDS(zero-discrepancy);
need work = AAPL(F), ORCL(F), RKLB(B), TSM(B), AAOI(A), ASX(A), FUTU(A), INTC(A), TIGR(A).

## Root causes in code
1. `peer_metric_validator` only **gates Telegram**, doesn't fix the report; and its
   Phase 8.2 listing-context guard (lines ~353-367) over-skips the comparison-table
   shape where fabrication actually happens → false negative.
2. Peer numbers are **LLM-authored**. Per repo philosophy (deterministic blocks >
   prompt rules), they must be Python-authoritative.
3. PDF cover leaks architecture: "simulated multi-agent decision pipeline",
   "Multi-agent research pipeline · {model_label}".
4. PDF **appendices render raw markdown verbatim** (raw/ paths, QC item numbers,
   "≤30 words paraphrased", agent roles) — heavy customer-facing leakage; polish
   pass only runs on front-of-document sections.

## Plan (phases)
- [x] P0 Skill/default + ops: --output-dir default → TK Research/preaudit; trading-research
      SKILL.md repointed (done earlier this session).
- [ ] P1 Customer-facing PDF hardening (no mini; pure-fn tests):
      - cover wording: drop architecture/"simulated" leak, customer phrasing.
      - apply leak-scrub (strip-directives + agentic-vocab + generic raw/<f> catch-all)
        to ALL rendered sections incl. appendices. Raw .md on disk stays verbatim (audit).
- [x] P2 Peer-number structural fix (no falsified data) — output-side corrector:
      snaps peer-metric values to peer_ratios.json before PDF + validators.
      Handles markdown TABLE columns (where the real P/E fabrications live) and
      prose "<TICKER> <METRIC> <value>". VERIFIED on 2026-05-21-MSFT: fixes all 4
      documented Fwd P/E fabrications. Wired in research.py; writes
      raw/peer_corrections.json. Commits 80c7cb3, ad017f2.
- [ ] P2b Make the VALIDATOR table+listing aware for DETECTION (gate A+). The bare
      slash-listing "GOOGL 27.51x / 32.46% capex / ..." (unlabeled ttm-vs-fwd P/E)
      is not auto-corrected; it must be flagged so nothing fabricated ships silently.
- [ ] P3 Filing-attribution guard: when sec_filing.md is an XBRL stub (no readable
      statements), forbid/flag "Note N" + statement-line citations to it.
- [ ] P4 Missing-data guard: fail loudly / mark N/A honestly when financials.json or
      sec_filing.md absent (no silent gaps presented as data).
- [x] Deployed P1+P2 to mini (HEAD d04365b). Added TRADINGRESEARCH_NO_TELEGRAM
      opt-out so audit re-runs don't auto-deliver to the customer chat.
- [x] 6 already-A+ fresh runs regenerated through the hardened stack (PDF leak-scrub
      + corrector), verified leak-free (0 markers in extracted PDF text) and
      corrector-clean, copied to ~/Documents/TK Research/final/, and recorded in
      REGISTER.md: GOOGL, ONDS (05-26), AMKR, MSTR, SOUN, STM (05-29).
      (Caught + fixed a corrector $-field bug that had cosmetically mutated MSTR/SOUN;
       restored sign glyphs. Commit 3f638af.)
- [~] P5 Re-run loop on macmini, ONE fresh ticker at a time — IN PROGRESS.
      First re-run kicked: TIGR 2026-05-29 → preaudit/. Then audit Tier 1-14 → A+?.
      Remaining fresh non-A+ to process: AAPL(F, needs P3 filing-guard),
      ORCL(F, fabricated CRM peer not in set), RKLB(B), TSM(B), AAOI(A), ASX(A),
      FUTU(A), INTC(A), TIGR(A, running now).
- [ ] P3 filing-attribution guard (AAPL) + forbidden-peer handling (ORCL) before
      re-running those two.
- [ ] P6 Remove failed (non-A+) runs after each is superseded by an A+ re-run or
      deemed unfixable; REGISTER.md updated as A+ reports land (6 entries so far).

## Notes
- Edits only on this MacBook; macmini pulls. SKILL.md lives on the mini (not in repo).
- Re-run cost ~30 min/ticker on the mini; OAuth 8h TTL (refresh before runs).
- Audit framework: project_2026-05-06_cadence_runs_pending_audit.md (Tier 1-14).
