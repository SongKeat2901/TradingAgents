---
name: cadence-followup
description: >-
  CANONICAL handler for "follow up the cadence", "QC and publish the latest
  cadence", "publish the passes", "finalize the latest research batch / wk NN".
  Runs the deterministic QC + publish orchestrator over the newest preaudit
  batch and posts a per-ticker verdict + batch summary. DM (SK) only. Distinct
  from `trading-research` (runs a NEW research) and `research-summary` (reads
  existing reports).
---

# Cadence follow-up (autonomous QC -> publish)

Invoke when SK DMs anything matching: "follow up the cadence", "QC + publish the
latest cadence", "finalize the batch / wk NN", "publish the passes". **DM only.**

## How to run

1. Determine the **week folder** (the cadence number is SEQUENTIAL, not the ISO
   week — 2026-06-02 was wk23 and 2026-06-05 is wk24 although they share an ISO
   week). If SK names it ("wk24"), use `wk 24 2026`. If SK says only "the latest
   cadence", run the CLI once with `--no-write` first, read the `week` it infers
   (highest existing folder), and confirm with SK whether this batch joins that
   week or starts the next one before promoting.

2. Run the orchestrator (it auto-detects the newest preaudit batch and prints a
   JSON contract — parse it, do NOT re-derive grades yourself):

       GOG_KEYRING_PASSWORD="$GOG_KEYRING_PASSWORD" \
         ~/tradingagents/.venv/bin/tradingcadencefollowup --week "wk NN 2026"

   Omit `--week` only for a read-only look; promotion needs an explicit week.

3. Read the contract fields:
   - `tickers[]`: each has `grade` ("A"/"HOLD"), `auto_dismissed` (known validator
     false positives the classifier already cleared), `needs_adjudication` (novel
     flags — see step 4), `published`, `promoted_to`, `error`.
   - `revalidated`: list of tickers whose stale `validation_report.json` was
     refreshed before grading (decision.md was newer — a hand-correction self-heal).
   - `summary_update_pending`: `true` when at least one ticker was published — the
     bot must do the gsheet digest (step 6b).
   - `writes_held` / `reauth_url`: gog token expired (step 5).
   - `week_required`: no week resolved — supply `--week` and re-run.
   - `token_valid`: `null` means a `--no-write` run (not checked).

4. For each ticker with non-empty `needs_adjudication`: open the cited `file` at
   `line_no` and the relevant `raw/*.json`, decide real-vs-false-positive the way
   the human does (net-debt $-grab, price-date from->to, peer "respectively", or a
   genuine error). The CLI already held these out of promotion (grade HOLD). If it
   is genuinely a real defect, leave it unpublished and call it out for manual fix.
   If it is in fact a false positive the classifier missed, note the pattern (so it
   can be added to the classifier later) — do NOT hand-promote unless you are sure.

5. If `writes_held` is true with a `reauth_url`: the gog token is expired. Post the
   re-auth instruction to SK, list the grade-A tickers awaiting publish, and STOP
   (do not retry writes until SK confirms re-auth). QC results are still complete.

6. Compose the DM:
   - Per ticker: `<T>: <grade>` + one line (published -> `promoted_to`; HOLD ->
     the defect; error -> the `error` string).
   - Batch summary: N graded A & published, M held, any re-auth / week needed.
   - If `revalidated` is non-empty, note which tickers had their validation
     auto-refreshed (stale report self-healed).

6b. **If `summary_update_pending` is `true`**: digest the just-published reports
   and update the Research Summary gsheet
   (ID `1VJowGGdxjCPd0jMpZVHJlC-C6aEspf1iJWOVPH0T7dk`).
   - For each published ticker, read `decision.md` to extract: **Rating**,
     **Reference price**, **12-month EV** (both % upside and $ target), and a
     **1-line thesis note**.
   - Write to the sheet via the `update-summary` skill / `gog sheets update` method
     (group rows by rating band; apply colour bands per rating: A+/Buy = green,
     Hold = amber, Avoid/Sell = red). This LLM-reading step cannot be automated
     deterministically — the CLI deliberately leaves it to the bot.
   - The CLI no longer calls `update_register.py` (the script does not exist and
     the digest requires LLM extraction anyway). The bot owns this step.

## Guardrails

- **Idempotent:** re-running cannot create Drive duplicates (publish is by file ID
  via `~/gsheet-tool/pdf_ids.tsv`).
- **Promote (`mv` to final/<week>) happens only for grade A** and only when a week
  is set + token valid; a held ticker is never touched. `promote` refuses to
  overwrite an already-promoted dir (FileExistsError -> surfaced as the ticker
  `error`, batch continues).
- **Never run against a batch the laptop session is actively QC-ing.** wk24
  (2026-06-05) was finalized by the laptop; this skill is for cadences from wk25
  onward. If unsure which batch is live, ask SK.
- Requires `GOG_KEYRING_PASSWORD` in the environment for the Drive/sheet writes.
