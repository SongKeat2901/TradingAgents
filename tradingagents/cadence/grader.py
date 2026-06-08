from __future__ import annotations

from tradingagents.cadence.models import FlagDisposition, RunData, RunVerdict
from tradingagents.cadence.fp_classifier import classify_run_flags


def grade_run(run: RunData) -> RunVerdict:
    verdicts = classify_run_flags(run)
    auto_dismissed = [v for v in verdicts
                      if v.disposition in (FlagDisposition.DISMISS,
                                           FlagDisposition.CORRECT_BY_DESIGN)]
    # Only MATERIAL/blocking NEEDS_ADJUDICATION flags hold the grade.
    blocking_open = [v for v in verdicts
                     if v.disposition is FlagDisposition.NEEDS_ADJUDICATION
                     and (v.detail.get("severity") != "MINOR")]
    grade = "A" if not blocking_open else "HOLD"
    return RunVerdict(ticker=run.ticker, grade=grade, flag_verdicts=verdicts,
                      auto_dismissed=auto_dismissed, needs_adjudication=blocking_open)
