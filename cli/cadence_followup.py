"""Autonomous cadence follow-up: QC the newest preaudit batch and publish passes.

Emits a JSON result contract on stdout for the OpenClaw `cadence-followup` skill
to consume (adjudicate residual flags + compose the DM). Deterministic core lives
in tradingagents/cadence/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tradingagents.cadence.batch import find_latest_batch, load_run
from tradingagents.cadence.grader import grade_run
from tradingagents.cadence import publish as pub

DEFAULT_PREAUDIT = Path.home() / "tkresearch" / "preaudit"
PDF_PARENT = "1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs"
MANIFEST = Path.home() / "gsheet-tool" / "pdf_ids.tsv"
ACCOUNT = "trueknotsg@gmail.com"
FINAL_BASE = (Path.home() / "Library/CloudStorage"
              / "GoogleDrive-trueknotsg@gmail.com"
              / "My Drive/True Knot/TK Research/final")
VENV_PY = str(Path.home() / "tradingagents" / ".venv" / "bin" / "python")
SUMMARY_SCRIPT = str(Path.home() / "gsheet-tool" / "update_summary.py")

_WEEK_RE = re.compile(r"^wk (\d+) (\d{4})$")


def _maybe_revalidate(run_dir: Path) -> bool:
    """Re-run the phase-7 validators when decision.md is newer than
    validation_report.json (a hand-correction left the report stale). Returns
    True if it revalidated. Best-effort: never raises into the batch."""
    rd = Path(run_dir)
    rep = rd / "validation_report.json"
    dec = rd / "decision.md"
    try:
        if dec.is_file() and (not rep.is_file()
                              or dec.stat().st_mtime > rep.stat().st_mtime):
            from cli.research_validation import (
                run_phase_7_validators, write_validation_report)
            write_validation_report(str(rd), run_phase_7_validators(str(rd)))
            return True
    except Exception:
        pass
    return False


def _highest_week(final_base: str) -> str | None:
    """The highest existing 'wk N YYYY' folder (by year then week). Fallback only
    when --week is not supplied; the cadence number is sequential, not the ISO week."""
    try:
        entries = os.listdir(final_base)
    except OSError:
        return None
    weeks = []
    for d in entries:
        m = _WEEK_RE.match(d)
        if m:
            weeks.append((int(m.group(2)), int(m.group(1)), d))
    return max(weeks)[2] if weeks else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preaudit-base", default=str(DEFAULT_PREAUDIT))
    ap.add_argument("--final-base", default=str(FINAL_BASE))
    ap.add_argument("--week", default=None,
                    help='target final/ week folder, e.g. "wk 24 2026". Required to '
                         "promote; the cadence number is SEQUENTIAL, not the ISO week. "
                         "If omitted, the highest existing week folder is used.")
    ap.add_argument("--no-write", action="store_true",
                    help="QC + emit contract without any Drive/sheet writes (no gog call)")
    ap.add_argument("--no-revalidate", action="store_true",
                    help="Skip the mtime check that re-runs phase-7 validators when "
                         "decision.md is newer than validation_report.json")
    args = ap.parse_args(argv)

    date, run_dirs = find_latest_batch(Path(args.preaudit_base))
    result = {
        "trade_date": date, "batch_size": len(run_dirs), "completed": len(run_dirs),
        "token_valid": None, "writes_held": False, "week_required": False,
        "reauth_url": None, "week": args.week, "tickers": [], "revalidated": [],
    }
    if not date:
        print(json.dumps(result, indent=2))
        return 0

    # token_valid is None under --no-write (not checked), else the real probe.
    token_ok = None if args.no_write else pub.gog_token_valid(ACCOUNT)
    result["token_valid"] = token_ok

    week = args.week or _highest_week(args.final_base)
    result["week"] = week

    writes_enabled = (not args.no_write) and bool(token_ok) and bool(week)
    if not args.no_write and not token_ok:
        result["writes_held"] = True
        result["reauth_url"] = (
            "re-auth required: run `gog auth add %s --services sheets,drive` "
            "in the mini browser" % ACCOUNT)
    elif not args.no_write and token_ok and not week:
        result["writes_held"] = True
        result["week_required"] = True

    for rd in run_dirs:
        if not args.no_revalidate:
            ticker_name = Path(rd).name.split("-", 3)[-1] if "-" in Path(rd).name else Path(rd).name
            if _maybe_revalidate(rd):
                result["revalidated"].append(ticker_name)
        run = load_run(rd)
        rv = grade_run(run)
        row = {
            "ticker": rv.ticker, "grade": rv.grade,
            "auto_dismissed": [{"phase": v.phase, "reason": v.reason}
                               for v in rv.auto_dismissed],
            "needs_adjudication": [{"phase": v.phase, "reason": v.reason, **v.detail}
                                   for v in rv.needs_adjudication],
            "published": False, "promoted_to": None, "error": None,
        }
        if writes_enabled and rv.grade == "A":
            pdf = Path(run.run_dir) / ("research-%s-%s.pdf" % (run.trade_date, run.ticker))
            if not pdf.is_file():
                row["error"] = "PDF missing"
            else:
                try:
                    pub.publish_pdf(run.ticker, pdf, MANIFEST,
                                    parent=PDF_PARENT, account=ACCOUNT)
                    dest = pub.promote(Path(run.run_dir), Path(args.final_base), week)
                    row["published"] = True
                    row["promoted_to"] = str(dest)
                except Exception as exc:  # degrade: one bad run never aborts the batch
                    row["error"] = "%s: %s" % (type(exc).__name__, exc)
        result["tickers"].append(row)

    # Research Summary gsheet: deterministically re-rendered from the promoted
    # reports by ~/gsheet-tool/update_summary.py (reliable rating/EV/price
    # extraction — no LLM needed; supersedes the old "flag for the bot" path).
    # Runs after promotion so the sheet always reflects the latest published set;
    # a failed render leaves summary_update_pending=True so it isn't lost.
    published_any = any(t["published"] for t in result["tickers"])
    if writes_enabled and published_any:
        updated = pub.refresh_summary_sheet(
            python=VENV_PY, script=SUMMARY_SCRIPT, account=ACCOUNT)
        result["summary_updated"] = updated
        result["summary_update_pending"] = not updated
    else:
        result["summary_updated"] = False
        result["summary_update_pending"] = published_any

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
