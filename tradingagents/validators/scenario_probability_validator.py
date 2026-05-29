"""Validate PM Bull/Base/Bear scenario targets+probabilities against the
deterministic raw/forward_probabilities.json anchor (Phase 8.x).

Phase 8.x is the same pattern as the existing Phase-7.x validators: the
Researcher's deterministic forward-distribution model writes the canonical
scenario table, the PM is instructed to use it verbatim, and this validator
mechanically catches any drift. MATERIAL violations gate Telegram delivery.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PROB_TOLERANCE = 0.05   # 5 percentage points
_TARGET_TOLERANCE_USD = 0.50   # match the existing price-date tolerance

# Match `| <scenario> | <a> | <b> |` rows where one of {a, b} is a percentage
# and the other is a dollar value. Tolerant of whitespace.
_ROW = re.compile(
    r"\|\s*(?P<scenario>Bull|Base|Bear)\s*\|"
    r"\s*(?P<a>[^|]+?)\s*\|"
    r"\s*(?P<b>[^|]+?)\s*\|",
    re.IGNORECASE,
)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DOLLAR = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


@dataclass(frozen=True)
class ScenarioViolation:
    severity: Literal["MATERIAL", "MINOR"]
    type: Literal["scenario_probability_drift", "scenario_target_drift"]
    scenario: str
    claimed: float
    anchor: float
    match_text: str


def _parse_row(scenario: str, a: str, b: str) -> tuple[float | None, float | None]:
    """Return (probability_fraction, target_dollars). The two cells can appear
    in either order — try both."""
    prob: float | None = None
    target: float | None = None
    for cell in (a, b):
        if prob is None:
            m = _PCT.search(cell)
            if m:
                try:
                    prob = float(m.group(1)) / 100.0
                    continue
                except ValueError:
                    pass
        if target is None:
            m = _DOLLAR.search(cell)
            if m:
                try:
                    target = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
    return prob, target


def validate_scenario_probabilities(decision_text: str, run_dir) -> list[ScenarioViolation]:
    """Check that the PM's scenario rows in decision.md match the deterministic
    Bull/Base/Bear anchor in raw/forward_probabilities.json.

    Returns [] when the anchor file is missing (no enforcement possible) OR the
    decision text contains no recognisable scenario rows. MATERIAL on any
    probability drift > 5 pp or target drift > $0.50 (the same close-match
    tolerance the price-date validator uses).
    """
    anchor_path = Path(run_dir) / "raw" / "forward_probabilities.json"
    if not anchor_path.exists():
        return []
    try:
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))["scenarios"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []

    vios: list[ScenarioViolation] = []
    seen: set[str] = set()
    for m in _ROW.finditer(decision_text):
        scen = m.group("scenario").lower()
        if scen in seen:
            continue
        seen.add(scen)
        if scen not in anchor:
            continue
        prob_claimed, target_claimed = _parse_row(scen, m.group("a"), m.group("b"))
        prob_anchor = anchor[scen].get("probability")
        target_anchor = anchor[scen].get("target")
        match_text = m.group(0).strip()
        if prob_claimed is not None and prob_anchor is not None:
            if abs(prob_claimed - prob_anchor) > _PROB_TOLERANCE:
                vios.append(ScenarioViolation(
                    severity="MATERIAL",
                    type="scenario_probability_drift",
                    scenario=scen,
                    claimed=prob_claimed,
                    anchor=prob_anchor,
                    match_text=match_text,
                ))
        if target_claimed is not None and target_anchor is not None:
            if abs(target_claimed - target_anchor) > _TARGET_TOLERANCE_USD:
                vios.append(ScenarioViolation(
                    severity="MATERIAL",
                    type="scenario_target_drift",
                    scenario=scen,
                    claimed=target_claimed,
                    anchor=target_anchor,
                    match_text=match_text,
                ))
    return vios
