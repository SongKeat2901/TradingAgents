"""Autonomous cadence follow-up: QC the newest preaudit batch and publish passes.

Emits a JSON result contract on stdout for the OpenClaw `cadence-followup` skill
to consume (adjudicate residual flags + compose the DM). Deterministic core lives
in tradingagents/cadence/.
"""
from __future__ import annotations

import argparse
import json
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
              / "GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/final")
REGISTER_PY = str(Path.home() / "gsheet-tool" / "update_register.py")
VENV_PY = str(Path.home() / "tradingagents" / ".venv" / "bin" / "python")


def _iso_week_folder(date_str: str) -> str:
    import datetime as _dt
    y, m, d = (int(x) for x in date_str.split("-"))
    wk = _dt.date(y, m, d).isocalendar().week
    return f"wk {wk} {y}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preaudit-base", default=str(DEFAULT_PREAUDIT))
    ap.add_argument("--no-write", action="store_true",
                    help="QC + emit contract without Drive/sheet writes (test/safety)")
    args = ap.parse_args(argv)

    date, run_dirs = find_latest_batch(Path(args.preaudit_base))
    result = {"trade_date": date, "batch_size": len(run_dirs),
              "completed": len(run_dirs), "token_valid": None,
              "writes_held": False, "reauth_url": None, "tickers": []}
    if not date:
        print(json.dumps(result, indent=2))
        return 0

    token_ok = (not args.no_write) and pub.gog_token_valid(ACCOUNT)
    result["token_valid"] = token_ok
    if not args.no_write and not token_ok:
        result["writes_held"] = True
        result["reauth_url"] = ("re-auth required: run `gog auth add "
                                f"{ACCOUNT} --services sheets,drive` in the mini browser")

    week = _iso_week_folder(date)
    for rd in run_dirs:
        run = load_run(rd)
        rv = grade_run(run)
        row = {
            "ticker": rv.ticker, "grade": rv.grade,
            "auto_dismissed": [{"phase": v.phase, "reason": v.reason}
                               for v in rv.auto_dismissed],
            "needs_adjudication": [{"phase": v.phase, "reason": v.reason, **v.detail}
                                   for v in rv.needs_adjudication],
            "published": False, "promoted_to": None,
        }
        can_write = rv.grade == "A" and token_ok and not args.no_write
        if can_write:
            pdf = Path(run.run_dir) / f"research-{run.trade_date}-{run.ticker}.pdf"
            if pdf.is_file():
                pub.publish_pdf(run.ticker, pdf, MANIFEST,
                                parent=PDF_PARENT, account=ACCOUNT)
                dest = pub.promote(Path(run.run_dir), FINAL_BASE, week)
                row["published"] = True
                row["promoted_to"] = str(dest)
        result["tickers"].append(row)

    if any(t["published"] for t in result["tickers"]):
        pub.refresh_summary_sheet(python=VENV_PY, script=REGISTER_PY, account=ACCOUNT)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
