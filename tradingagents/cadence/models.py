from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FlagDisposition(Enum):
    DISMISS = "dismiss"                 # known-safe validator false positive
    CORRECT_BY_DESIGN = "correct"       # e.g. non-USD reporter skip
    NEEDS_ADJUDICATION = "adjudicate"   # novel — escalate to the bot LLM


@dataclass
class FlagVerdict:
    phase: str
    disposition: FlagDisposition
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class RunData:
    ticker: str
    trade_date: str
    run_dir: str
    validation: dict
    intrinsic_value: dict
    peer_ratios: dict
    financials: dict
    reference_price: float | None


@dataclass
class RunVerdict:
    ticker: str
    grade: str                       # "A" or "HOLD"
    flag_verdicts: list[FlagVerdict]
    auto_dismissed: list[FlagVerdict]
    needs_adjudication: list[FlagVerdict]
