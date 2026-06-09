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
REGISTER_PY = str(Path.home() / "gsheet-tool" / "update_register.py")
VENV_PY = str(Path.home() / "tradingagents" / ".venv" / "bin" / "python")

_WEEK_RE = re.compile(r"^wk (\d+) (\d{4})$")


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
    args = ap.parse_args(argv)

    date, run_dirs = find_latest_batch(Path(args.preaudit_base))
    result = {
        "trade_date": date, "batch_size": len(run_dirs), "completed": len(run_dirs),
        "token_valid": None, "writes_held": False, "week_required": False,
        "reauth_url": None, "week": args.week, "tickers": [],
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

    if any(t["published"] for t in result["tickers"]):
        try:
            pub.refresh_summary_sheet(python=VENV_PY, script=REGISTER_PY, account=ACCOUNT)
        except Exception as exc:
            result["summary_refresh_error"] = "%s: %s" % (type(exc).__name__, exc)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
