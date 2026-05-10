"""Shared helpers for Phase 7 validators.

Phase 7.8 consolidation: each validator (claim_extractor, price_date,
quote_attribution, peer_metric, net_debt) had its own copy of `_line_no`
and the net-debt validator had peer-ticker scope helpers that were
useful to other validators too. Centralising here prevents drift
across validators (e.g. when v1.5 added `:` to the peer-ticker
delimiter, only net_debt got it; with this module the change is
visible to all).
"""

from __future__ import annotations

import re


def line_no(text: str, char_offset: int) -> int:
    """Return 1-indexed line number for a character offset in `text`."""
    return text[:char_offset].count("\n") + 1


# Peer-ticker pattern: matches an uppercase 2-5-letter token followed
# by a delimiter. Delimiters: `'s` (possessive), whitespace, `:` (table-
# row form, RMBS 2026-05-08 fix). Used by validators that need to detect
# whether a claim's surrounding prose attributes a metric to a peer
# ticker rather than the main ticker.
PEER_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}(?:'s|\s|:)")


# Common uppercase 2-5-letter tokens that aren't tickers — accumulate
# new entries here as false-positive context surfaces during audits.
_NON_TICKER_TOKENS = frozenset({
    "USD", "GAAP", "ARR", "CC", "EBITDA", "FCF", "AI", "PM", "TA",
    "CEO", "CFO", "API", "FY", "OCF", "NTM", "TTM", "SOTP", "NDR",
    "RPO", "SBC", "MA", "II", "QC", "RM", "SEC", "USA", "NYC",
    # Phase 7.8 additions: surfaced during historical audits but never
    # tagged before because they only mattered for the peer-attribution
    # path. Kept here so adding new tokens stays a one-line change.
    "GM", "EPS", "OPEX", "COGS", "DPS", "BPS", "NAV", "CD",
})


def claim_attributed_to_other_ticker(
    match_text: str,
    main_ticker: str | None,
) -> bool:
    """Heuristic: is the claim's surrounding prose attributing it to a
    DIFFERENT ticker than the main one?

    Returns True when at least one peer-ticker token appears in
    `match_text` and the set of tokens is not solely the main ticker.
    Returns False when no ticker is found (assume main-ticker claim) or
    when only the main ticker is mentioned.

    Filters out common uppercase non-ticker tokens (USD, EBITDA, FCF,
    etc.) so they don't trigger spurious peer attribution.
    """
    main_upper = (main_ticker or "").upper()
    found_tickers: set[str] = set()
    for m in PEER_TICKER_PATTERN.finditer(match_text):
        # Rebuild bare ticker (strip the delimiter the pattern matched)
        token = re.match(r"[A-Z]{2,5}", m.group(0)).group(0)
        if token in _NON_TICKER_TOKENS:
            continue
        found_tickers.add(token)

    if not found_tickers:
        return False  # no ticker context — assume main-ticker claim
    if found_tickers == {main_upper}:
        return False  # only main ticker mentioned — validate
    return True  # at least one other ticker present — peer-attributed
