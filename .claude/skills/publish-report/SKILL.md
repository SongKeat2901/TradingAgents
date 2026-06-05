---
name: publish-report
description: >-
  Idempotently publish a completed research PDF to the shared TK Research pdf/
  Drive folder, register it in the ID manifest, and re-run the macro engine so the
  ticker joins the Trading Plan. Use when asked to "publish <ticker>", "add
  <ticker> to the plan", or after a fresh research run lands (e.g. MARA, COIN).
disable-model-invocation: true
---

# Publish a research PDF → Drive + manifest + Trading Plan

Enforces the **no-duplicates** rule (publish by known file ID, never name-search —
search is broken on this account). See `feedback_no_duplicates_idempotent_publish`
and `project_macro_regime_engine_deployed` memories. All ops on macmini-trueknot as
`trueknotsg@gmail.com`; gog needs `GOG_KEYRING_PASSWORD` in env (value on the mini).

Key IDs:
- Shared **pdf/** folder (PDFs now live flat here): `1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`
- Manifest: `~/gsheet-tool/pdf_ids.tsv` (`ticker<TAB>fileId`)
- Trading Plan sheet: `1ZLq9HuyU0AAzREECpBGpamDBVpbbjq9V8joHqQHp-cw`

## Steps

1. **Locate the run** + its PDF: e.g. `~/tkresearch/preaudit/<date>-<TICKER>/research-<date>-<TICKER>.pdf`.
   Confirm `decision.md` + the PDF exist (a finished run).

2. **Publish idempotently** (per ticker):
   - If the ticker is **already** in the manifest: trash the old file
     (`gog drive trash <oldId> -a trueknotsg@gmail.com`), upload the new PDF, capture
     the NEW id, and **overwrite** the manifest row.
   - If **absent**: `gog drive upload <pdf> --parent 1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs -a trueknotsg@gmail.com -j`,
     capture `file.id`, and **append** `printf 'TICKER\tID\n' >> ~/gsheet-tool/pdf_ids.tsv`.
   - Parse the manifest in Python (never `IFS=$"\t"` / `grep -P` — broken on macOS).

3. **Regenerate intrinsic** for the new ticker (so it uses the current guarded code):
   `ssh macmini-trueknot '.venv/bin/python /tmp/regen_intrinsic.py'`.

4. **Re-run the macro engine** (use the `macro-run` skill) — the new ticker joins the
   plan (it must have a run dir under one of the `--reports-dir` bases, e.g. `~/tkresearch`).

5. **Verify**: read the sheet's ticker + Research columns and confirm the new row is
   present and linked:
   `gog sheets get <sheet-id> "A6:Q30" -a trueknotsg@gmail.com -p` → row exists, col Q is a `drive.google` link.

## Notes
- New PDFs go to the **pdf/ root** (the folder was flattened so colleagues see them
  directly), NOT the old `wk 23 2026` subfolder.
- `--manifest-scope` means the manifest IS the plan's watchlist: adding here adds to the
  plan; to REMOVE a ticker, delete its manifest row and re-run (see how FUTU was dropped).
