"""Phase 7 post-output validators (2026-05-08).

After the LLM pipeline writes its outputs (decision.md / decision_executive.md
/ debate_*.md / technicals_v2.md), the validators in this package
mechanically verify every numerical / quote / date claim against the
deterministic raw/ ground truth. Each validator is a pure-Python check
that runs in milliseconds and returns a structured list of violations.

This is the symmetric counterpart to the deterministic INPUT blocks
(Phase 6.2-6.9). The INPUT blocks pin authoritative cells before the
LLM writes; the OUTPUT validators verify the LLM didn't drift.

Phase 7.1: claim extractor + price/date validator (this commit).
Phase 7.2: quote attribution validator (planned).
Phase 7.3: peer-metric source check (planned).
Phase 7.4: Telegram delivery gate on validators passing (planned).
"""

from tradingagents.validators.claim_extractor import (
    DateCloseClaim,
    extract_date_close_claims,
)
from tradingagents.validators.price_date_validator import (
    Violation,
    validate_date_close_claims,
)

__all__ = [
    "DateCloseClaim",
    "extract_date_close_claims",
    "Violation",
    "validate_date_close_claims",
]
