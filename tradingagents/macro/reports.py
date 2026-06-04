"""Read each report's base EV from a research run dir.

Reuses cli.daily_followup.parse_research (the existing, battle-tested
decision.md parser) so we have a single source of truth for the regexes. Adds
a percentage view and a scenario-weighted fallback when no explicit EV line
exists.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from cli.daily_followup import parse_research, Scenario

logger = logging.getLogger(__name__)


@dataclass
class BaseEV:
    ticker: str
    research_date: str
    rating: str
    reference_price: float
    ev: float | None
    scenarios: list[Scenario]
    hard_stop: float | None
    run_dir: str = ""          # the run dir this was parsed from (for raw/ lookups)


def load_base_ev(run_dir: Path) -> BaseEV | None:
    try:
        parsed = parse_research(Path(run_dir))
    except OSError:
        return None
    if not parsed:
        return None
    return BaseEV(
        ticker=parsed["ticker"],
        research_date=parsed["research_date"],
        rating=parsed["rating"],
        reference_price=parsed["reference_price"],
        ev=parsed["ev"],
        scenarios=parsed["scenarios"],
        hard_stop=parsed["hard_stop"],
        run_dir=str(run_dir),
    )


def load_intrinsic(run_dir: str | Path) -> dict | None:
    """Read raw/intrinsic_value.json → {fair_value, margin_of_safety_pct, profile}.

    `fair_value` is the base-case per-share intrinsic value; it is None for
    foreign ADRs and unprofitable names where the DCF/EPV model is skipped
    (the report then relies on the scenario EV). Returns None if the file is
    absent or unreadable.
    """
    if not run_dir:
        return None
    p = Path(run_dir) / "raw" / "intrinsic_value.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fv = d.get("fair_value") or {}
    return {
        "fair_value": fv.get("base"),
        "margin_of_safety_pct": d.get("margin_of_safety_pct"),
        "profile": d.get("profile"),
    }


def _scenario_weighted_target(be: BaseEV) -> float | None:
    num = den = 0.0
    for sc in be.scenarios:
        if sc.probability is None or sc.target is None:
            continue
        num += sc.probability * sc.target
        den += sc.probability
    return (num / den) if den else None


def scenario_ladder(be: BaseEV) -> dict[str, float | None]:
    """Bull/base/bear scenario targets keyed by lowercase scenario name."""
    out: dict[str, float | None] = {"bull": None, "base": None, "bear": None}
    for sc in be.scenarios:
        key = (sc.name or "").lower()
        if key in out and sc.target is not None:
            out[key] = sc.target
    return out


def ev_pct(be: BaseEV) -> float | None:
    """12-mo EV as a fraction of reference price. Uses the explicit EV line if
    present, else the scenario-probability-weighted target."""
    ev_abs = be.ev if be.ev is not None else _scenario_weighted_target(be)
    if ev_abs is None:
        logger.warning("ev_pct: no EV or usable scenarios for %s (%s)",
                       be.ticker, be.research_date)
        return None
    if not be.reference_price:
        return None
    return (ev_abs - be.reference_price) / be.reference_price


def latest_runs(base_dirs) -> dict[str, BaseEV]:
    """Newest BaseEV per ticker across one or more run-dir base trees.

    `base_dirs` is a single path or a list of paths. Run dirs may sit directly
    under a base (preaudit) or nested deeper (final/wk NN YYYY/<date>-<ticker>/);
    we locate them by finding every state.json at any depth and taking its parent.

    Tie-break: per ticker we keep the run with the latest (research_date, then
    state.json mtime). The mtime tie-break matters when the same trade_date was
    written into more than one base — the most recently written copy (e.g. a
    corrected rerun) wins over an earlier same-date original.
    """
    if isinstance(base_dirs, (str, os.PathLike)):
        base_dirs = [base_dirs]
    best: dict[str, tuple] = {}   # ticker -> ((date, mtime), BaseEV)
    for base in base_dirs:
        for state_file in Path(base).rglob("state.json"):
            be = load_base_ev(state_file.parent)
            if not be:
                continue
            try:
                mtime = state_file.stat().st_mtime
            except OSError:
                mtime = 0.0
            key = (be.research_date, mtime)
            prev = best.get(be.ticker)
            if prev is None or key > prev[0]:
                best[be.ticker] = (key, be)
    return {ticker: payload[1] for ticker, payload in best.items()}
