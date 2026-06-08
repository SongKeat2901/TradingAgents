# Bot cadence-followup — design

**Date:** 2026-06-08
**Status:** Approved design, pre-implementation
**Author:** SK + Claude

## Problem

The cadence follow-up workflow — monitor preaudit runs → QC each to grade A →
promote preaudit→Drive `final/` → update the Research Summary gsheet — currently
runs only inside a Claude Code session on the laptop, driven by the human + the
repo skills (`cadence-run`, `report-auditor`, `publish-report`, `update-summary`).
The TrueKnot OpenClaw bot on `macmini-trueknot` cannot do this follow-up; it only
has `trading-research` (run a research), `research-summary` (read reports),
`scanner`, and the IBKR skills.

Goal: let SK **DM `@TrueKnotBot` privately** and have it autonomously QC the
latest cadence and publish the passes — no laptop session required.

## Decisions (locked with SK, 2026-06-08)

| Decision | Choice |
|---|---|
| Autonomy | **Full** — QC → promote → gsheet, no human approval gate |
| Mode | **Live only** — no dry-run mode is built |
| Batch selection | **Conversational** — bot infers the newest preaudit batch |
| Channel | **DM only**, SK-only (existing allowlist Telegram `8311935445`) |
| gog token expired at run time | **QC live + DM the re-auth URL + hold the Drive/gsheet writes**, finish writes after SK re-auths |
| Audit trail | **Per-ticker DM** verdict + final batch summary |
| Approach | **C — Hybrid**: deterministic Python orchestrator that auto-dismisses known-safe flag patterns; bot LLM adjudicates only residual/novel flags |
| wk24 ownership | wk24 (currently running) **stays with the laptop session**; the skill **debuts on the next cadence**, so two actors never `mv` the same preaudit dir. Bot can target any past batch by name once deployed. |
| Code location | Orchestrator in the repo (`cli/cadence_followup.py` + console script), deployed to the mini via `deploy-mini`; thin `SKILL.md` in the bot workspace — same split as `trading-research`. |

## Why Approach C (hybrid)

Under full autonomy with **no dry-run gate**, the known-safe flag dismissals must
be **deterministic**, not re-guessed by the LLM each run. This cadence surfaced a
recurring reality: every blocking validation flag so far (6 across 5 tickers) was a
**validator parser false positive**. Codifying those signatures in Python means the
bot can't accidentally treat a known-FP as a real defect (or vice-versa). The LLM is
reserved for flags that match **no** known pattern — genuinely novel cases that
deserve judgement — and for composing the human-readable DM summary.

(Approach A = pure SKILL.md was rejected: LLM re-litigates known FPs every run with
no determinism. Approach B = pure Python was rejected: the residual semantic triage
genuinely needs LLM reading and is not worth fully codifying.)

## Architecture

```
DM "follow up / QC + publish the latest cadence"  (SK, private)
  → cadence-followup SKILL.md  (bot workspace; conversational trigger; SK-DM only)
    → cadence-followup CLI  (repo, deployed to mini):
        1. detect newest preaudit batch
        2. for each completed ticker:
             a. read validation_report.json + raw/{intrinsic_value,peer_ratios,financials}.json
             b. FP-classifier → auto-dismiss known-safe flags
             c. residual blocking flags → emit as "needs-adjudication" records
             d. grade (provisional): A iff no residual + no real defect
        3. gog token guard:
             valid   → for each grade-A ticker: idempotent publish + gsheet write
             expired → skip all writes, collect re-auth URL
        4. emit machine-readable result (JSON) to stdout
    → bot LLM:
        - adjudicate each "needs-adjudication" record against raw/ (real vs FP)
        - finalize grade per ticker
        - compose per-ticker + batch DM summary (incl. any held writes + re-auth URL)
```

## Components

Each is independently testable with a single clear responsibility.

### 1. Batch detector
Newest `trade_date` under `~/tkresearch/preaudit/` with at least one completed run
(`decision.md` present), **excluding archived `*.*` dirs** (the `.pre-cadence` /
`.run` glob bug — always filter `! -name "*.*"`). Returns the date + the list of
completed ticker run-dirs.

### 2. FP-classifier (the deterministic core)
Input: one run's `validation_report.json` blocking violations + the run's `raw/`.
For each blocking violation, classify as **auto-dismiss (known-safe)**,
**correct-by-design**, or **needs-adjudication**. Known patterns (catalogued from
wk24):

- **`phase_7_5_net_debt` / definitional_drift — $-figure grab.** The claimed dollar
  value is a *different* balance-sheet/cash-flow quantity that happens to sit near
  the words "net debt/net cash" (e.g. a buyback authorization `$163B`, or a
  cash-flow line "net cash from operating activities −$2,540M" = FCF). Dismiss when
  the `match_text` shows the claimed value is labelled as something other than a
  net-debt/net-cash *balance* (buyback / operating cash flow / FCF), and the run's
  actual cited net debt matches yfinance `raw/intrinsic_value.json:inputs.net_debt`.
- **`phase_7_1_price_date` / wrong_close — from→to mis-pair.** `match_text` is a
  "decline/move **from** $A (date) **to** $B" construction and the validator paired
  the date with $B (the endpoint) instead of $A. Dismiss on the **structural** check:
  the validator's `actual_close` equals $A (the *from*-value) while its
  `claimed_price` equals $B (the *to*-value), confirming the cross-wire — no fresh
  yfinance call needed. (Optional stronger check: $A matches yfinance for the
  cited date; keep it out of the hot path.)
- **`phase_7_3_peer_metric` / wrong_peer_metric — "respectively" mis-map.** Two
  tickers then "X and Y respectively"; the validator mapped the metric to the wrong
  ticker. Dismiss when the claimed value matches the *other* ticker's cell in
  `raw/peer_ratios.json` (i.e. the "respectively" order is correct).
- **`skipped_non_usd_reporter` (MINOR).** Correct-by-design for non-USD reporters
  (TWD/HKD/EUR ADRs) — never a defect.
- **Plausibility-gate FV suppression.** `intrinsic_value.json:skipped_methods`
  carrying an "eps mis-scaled / P/E∉[3,60] / diverges implausibly" reason is the
  guard working as intended — correct-by-design, not a defect. (Note: it also
  suppresses *genuine* high-multiple names like MRVL ~91x; that's a known
  conservative tradeoff, not a run defect.)

Anything not matching a known pattern → **needs-adjudication** (escalate to LLM).
The classifier is **conservative**: when a pattern *almost* matches but a cross-check
fails, it escalates rather than auto-dismisses.

### 3. LLM adjudicator (bot)
For each needs-adjudication record, the bot reads the cited file + `raw/` and decides
real-vs-FP, exactly as the human does today. A confirmed **real** defect → ticker is
**not** graded A (held in preaudit, surfaced in the DM for manual fix).

### 4. Grader
Grade A iff: zero confirmed real defects AND zero unresolved needs-adjudication.
Foreign-ADR / unprofitable "not computable" intrinsic value is **not** a defect.

### 5. Idempotent publisher
Reuses `publish-report` logic. Per grade-A ticker:
- PDF → shared `pdf/` Drive folder `1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`, **by file
  ID** via `~/gsheet-tool/pdf_ids.tsv` (`ticker<TAB>fileId`): if ticker present,
  trash old id + upload + overwrite row; if absent, upload + append. Never
  name-search (broken on this account). Parse the manifest in Python (never
  `IFS=$"\t"` / `grep -P` — macOS-broken).
- Promote: `mv` preaudit run → `final/wk NN YYYY/<date>-<ticker>/` (create the
  ISO-week folder if absent). **This is the only destructive step and runs only on
  grade A.**

### 6. gsheet updater
Reuses `update-summary` / register-update logic to refresh the Research Summary
sheet `1VJowGGdxjCPd0jMpZVHJlC-C6aEspf1iJWOVPH0T7dk` (one row per ticker: rating,
price, EV, move, notes). Read `FORMATTED_VALUE`; write via `gog sheets update`
(not the REST values endpoint); sanitise NaN→"".

### 7. Token guard
Before any write: `gog auth list | grep trueknotsg` and a cheap probe; treat an
`invalid_grant` as expired. **Expired** → skip components 5 & 6 entirely, collect the
re-auth instruction + URL, set `writes_held=true` in the result. QC (components 1–4)
always runs regardless of token state.

### 8. DM reporter (bot)
Composes the per-ticker verdict lines and the batch summary, including: graded A &
published, held-for-manual-fix (with the real-defect detail), and — if
`writes_held` — the re-auth URL and the list of A-grade tickers awaiting publish.

## Data flow & result contract

The CLI emits a single JSON object to stdout the bot consumes:

```json
{
  "trade_date": "YYYY-MM-DD",
  "batch_size": 21,
  "completed": 21,
  "token_valid": true,
  "writes_held": false,
  "reauth_url": null,
  "tickers": [
    {
      "ticker": "AAPL",
      "grade": "A",
      "auto_dismissed": [{"phase": "phase_7_5_net_debt", "reason": "buyback $163B mislabeled as net debt"}],
      "needs_adjudication": [],
      "real_defects": [],
      "published": true,
      "promoted_to": "final/wk 24 2026/2026-06-05-AAPL"
    }
  ]
}
```

`needs_adjudication` records carry `{phase, file, line_no, claimed, match_text}` so
the bot can adjudicate without re-parsing.

## Error handling & safety

- **No dupes:** idempotent-by-file-ID publish; re-running the whole follow-up is safe.
- **Destructive op gated:** `mv` to `final/` only on grade A; flagged tickers stay
  in preaudit.
- **Token expiry graceful:** QC always completes; writes pause; SK gets the URL.
- **Partial batch:** the detector only includes completed runs; an in-flight cadence
  is followed-up for whatever has landed (safe to re-run as more land).
- **Conservative classifier:** ambiguous flags escalate to the LLM, never silently
  auto-dismiss.

## Testing

- **FP-classifier unit tests** using wk24 as known-answer fixtures: AAPL (2 net-debt
  FPs), AMKR (peer "respectively" FP), ASX (price-date from→to FP + non-USD skip),
  INTC (net-cash/FCF FP), FUTU/IFNNY (non-USD skip + plausibility suppression =
  correct-by-design) must all classify as dismiss/correct; a **synthetic real
  defect** (e.g. a genuinely wrong peer ND/EBITDA) must classify as
  needs-adjudication (NOT dismissed).
- **Batch detector** test: a fixture preaudit tree with a mix of completed,
  in-flight, and archived `*.*` dirs returns only the newest date's completed runs.
- **Idempotent publisher** test: manifest present vs absent paths; manifest parsed in
  Python; no name-search.
- **Token guard** test: valid vs `invalid_grant` → `writes_held` toggles, QC still
  runs.

## Out of scope

- Fixing the underlying validator false positives (separate work; do not touch
  mid-cadence).
- Running the research cadence itself (that's `trading-research` / `cadence-run`).
- Any dry-run mode (explicitly declined).
- WhatsApp/admin agent (off-limits).

## Open dependency (operational, not a blocker)

The bot needs `GOG_KEYRING_PASSWORD` in its execution env to run `gog`. Source of
truth is `OpenClawOps/trueknot-runbook.md`; wiring it into the bot's environment is
an implementation task. The ~weekly gog OAuth re-auth remains an unavoidable human
step (mini browser, trueknotsg) — handled by the token-guard graceful path.
