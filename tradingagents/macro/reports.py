"""Read each report's base EV from a research run dir.

Reuses cli.daily_followup.parse_research (the existing, battle-tested
decision.md parser) so we have a single source of truth for the regexes. Adds
a percentage view and a scenario-weighted fallback when no explicit EV line
exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logging

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
    )


def _scenario_weighted_target(be: BaseEV) -> float | None:
    num = den = 0.0
    for sc in be.scenarios:
        if sc.probability is None or sc.target is None:
            continue
        num += sc.probability * sc.target
        den += sc.probability
    return (num / den) if den else None


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


def latest_runs(base_dir: Path) -> dict[str, BaseEV]:
    """Newest BaseEV per ticker across all run dirs under base_dir.

    Run dirs may sit directly under base_dir (preaudit) or nested one level
    deeper under week buckets (final/wk NN YYYY/<date>-<ticker>/). We locate
    them by finding every state.json at any depth and taking its parent.
    """
    out: dict[str, BaseEV] = {}
    for state_file in Path(base_dir).rglob("state.json"):
        be = load_base_ev(state_file.parent)
        if not be:
            continue
        prev = out.get(be.ticker)
        if prev is None or be.research_date > prev.research_date:
            out[be.ticker] = be
    return out
