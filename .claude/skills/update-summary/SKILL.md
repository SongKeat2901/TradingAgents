---
name: update-summary
description: >-
  Digest the finalized TK Research reports and update the native TrueKnot
  Research Summary Google Sheet (which lives inside final/ in the trueknotsg
  My Drive) via the gog CLI on macmini-trueknot. Use after promoting reports
  to final/, or when asked to "update the research summary / the sheet". One
  running portfolio table: one row per ticker (rating, price, EV, move, notes).
disable-model-invocation: true
---

# Update Research Summary (Google Sheet via gog)

The customer-facing summary is a **native Google Sheet**, not the old xlsx.
It lives **inside `final/`** in the trueknotsg Google Drive and is edited with
the `gog` CLI on `macmini-trueknot` (account `trueknotsg@gmail.com`).

- **Spreadsheet ID:** `1VJowGGdxjCPd0jMpZVHJlC-C6aEspf1iJWOVPH0T7dk`
- **gog keyring password:** the trueknot keyring pw — do NOT hardcode it here;
  read it from `OpenClawOps/trueknot-runbook.md` (`GOG_KEYRING_PASSWORD`). Export
  it in the shell for every gog call: `export GOG_KEYRING_PASSWORD=<pw>`.
- Base path: `~/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research`.

## Steps

1. **Verify the gog token (it expires ~every 7 days — unverified-app rule).**
   ```bash
   ssh macmini-trueknot 'export GOG_KEYRING_PASSWORD=<pw>; /opt/homebrew/bin/gog auth list 2>&1 | grep trueknotsg'
   ```
   If a later gog call returns `oauth2: "invalid_grant"`, re-auth (browser on the
   mini, signed in as trueknotsg — the 127.0.0.1 callback only resolves there):
   ```bash
   ssh macmini-trueknot 'export NVM_DIR=$HOME/.nvm; . $NVM_DIR/nvm.sh; GOG_KEYRING_PASSWORD=<pw> /opt/homebrew/bin/gog auth add trueknotsg@gmail.com --services gmail,calendar,drive,docs,contacts,sheets,tasks,people'
   ```
   Surface the printed URL; user opens it in the **mini's** browser → Advanced → Allow.

2. **Digest the finalized reports.** For each run under `final/wk NN YYYY/<date>-<ticker>/`:
   - `decision.md` → `Reference price:** $<price> (yfinance close of <date>)` (the authoritative spot).
   - `decision_executive.md` → rating (bold after `## Rating and Trading Plan`, or
     `rating the name **X**`), and the 12-month EV ($ and %) — phrasing varies
     ("probability-weighted EV is $X, +Y%", "12-month expected value of $X (+Y%)",
     "expected value is +Y%"); grab the EV sentence and read it, compute EV$ = price×(1+EV%)
     when only % is given. Notes = primary catalyst + key trigger (1 line).
   - Latest close + move-since-report: `~/tradingagents/.venv/bin/python` yfinance
     (for same-day reports move≈0; only multi-day-old rows show real moves).

3. **Write the rows** (group by rating: Overweight, Hold, Underweight; stale carryovers last):
   ```bash
   gog sheets update <SID> 'Sheet1!A1' --input USER_ENTERED --values-json '[[...],[...]]' -a trueknotsg@gmail.com
   ```
   Columns: Report Date · Ticker · Rating · Price @ Report ($) · EV 12-Mo ($) ·
   EV 12-Mo (%) · Current Close ($) · Move Since (%) · Notes · Report PDF.

4. **Color bands** (legend at rows ~28-34): green=Overweight, yellow=Hold,
   red=Underweight, grey=stale carryover, orange=re-running/pending.
   ```bash
   gog sheets format <SID> 'Sheet1!A2:J2' --format-json '{"backgroundColor":{"red":0.78,"green":0.91,"blue":0.79}}' --format-fields userEnteredFormat.backgroundColor -a trueknotsg@gmail.com
   ```

5. **Confirm**: `gog sheets get <SID> 'Sheet1!A1:F6' -p` and report the sheet URL.

## Notes
- The build/format helper scripts from the initial build live in `~/gsheet-tool/`
  on the mini (build_sheet.py / format_sheet.py / legend.py) — adapt rather than
  rewrite. See memory `reference_trueknot_research_gsheet`.
- The sheet is owned by + private to trueknotsg; viewers must be signed in as
  trueknotsg (or share via `gog drive permissions`).
- gog runs on the mini only ([[feedback_google_stuff_only_on_mini]]); never the laptop.
