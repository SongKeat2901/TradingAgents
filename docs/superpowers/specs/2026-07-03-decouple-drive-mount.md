# Decouple promote/publish from the GUI Google-Drive mount (Option A)

**Shipped 2026-07-03.** The mini is a shared Mac: when its GUI login switches away
from `trueknot` (e.g. to THTimber), the Google Drive *for Desktop* mount at
`~/Library/CloudStorage/GoogleDrive-…` unmounts, breaking any op that touches that
path. The 2026-07-01 cadence promotion stalled on exactly this. Fix: no publishing
op may depend on the mount.

## Design

- **Canonical published store = LOCAL**: `~/tkresearch/final/wk NN YYYY/<date>-<TICKER>/`.
  Promote = `mv` from `~/tkresearch/preaudit/…` into it. Always available, no mount.
- **Drive copy = gog API only, by file-ID**: the PDF (and only the PDF) is uploaded
  to the flat shared `pdf/` folder (`1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`) with
  `gog drive upload`, tracked in `~/gsheet-tool/pdf_ids.tsv` (`ticker<TAB>fileId`).
  Replacement is upload-new → overwrite manifest row → `gog drive delete <oldId> -y`
  (gog ≥0.31 has no `trash` subcommand; `delete` moves to trash). Never name-search,
  never write through the mount.
- **Sheets** (Trading Plan `1ZLq9…`, Research Summary `1VJow…`) were already
  mount-independent via gog by spreadsheet ID; their reports source moves to the
  local final tree.
- gog account is `shianpin@trueknot.sg` (identity migrated from trueknotsg@gmail.com
  ~2026-06-26; note the `shianpin` spelling in `gog auth list`).

## Changes

| Where | Change |
|---|---|
| `tradingagents/cadence/publish.py` | `drive trash` → `drive delete <id> -y` (trash was a silent no-op on gog v0.31); `gog_token_valid` default account → shianpin |
| `cli/cadence_followup.py` | `FINAL_BASE` → `~/tkresearch/final`; `ACCOUNT` → `shianpin@trueknot.sg` |
| `tests/cadence/` | contract tests: FINAL_BASE local + mount-free, delete-not-trash call shape |
| `.claude/skills/publish-report` | local final canonical, delete-not-trash, shianpin account |
| `.claude/skills/cadence-run` | runs land via `TK_RESEARCH_BASE=$HOME/tkresearch`; promote to local final |
| `.claude/skills/update-summary` | reports base = local final; account refs |
| `.claude/skills/macro-run` | `~/tkresearch` first `--reports-dir` (rglob covers preaudit + final); account |
| `ops/com.trueknot.macrodaily.plist` | `GOG_ACCOUNT` → shianpin |
| mini `~/tkresearch/publish-2026-07-01.sh` | `FINAL` → `~/tkresearch/final/wk 27 2026`; `-a shianpin@trueknot.sg` |
| mini `~/gsheet-tool/update_summary.py` | `FINAL` → `~/tkresearch/final`; `ACCT` → shianpin |
| mini `~/gsheet-tool/refresh_trading_plan.sh` | `--reports-dir ~/tkresearch`; `GOG_ACCOUNT` → shianpin |
| mini installed macrodaily plist | GOG_ACCOUNT + reports-dirs off the dead mount (daily job had been failing since 06-26) |

## Invariants

- Promote/publish/sheet-update never read or write `~/Library/CloudStorage/…`.
- The Drive `final/` tree (shianpin My Drive) is a **convenience mirror only**; the
  wk ≤27 history there is not consulted by any tooling.
- `pdf_ids.tsv` remains the single source of truth for published Drive files
  (no-duplicates rule: replace by ID, never re-upload blind).
