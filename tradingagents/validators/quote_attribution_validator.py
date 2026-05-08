"""Validate italic-quoted analyst quotes against source agent output.

The COIN 2026-05-08 audit surfaced TA Agent v2 emitting a fake
attribution block::

    **Revision 1 — Market Analyst:** *"COIN closed the session at $206.50
    on 14.39M shares — roughly 1.8–2x the trailing daily average — after
    the 10-Q filing on May 7."*

But analyst_market.md (Market Analyst's actual output) said no such
thing — its real prose was *"close $192.96 on 8.99M shares"*. TA v2
fabricated the quote AND the attribution.

This validator extracts italic-quoted text with agent attribution from
LLM-authored markdown and verifies the load-bearing numerical claims in
each quote actually appear in the attributed agent's source file. We
compare *numerical fingerprints* (dollar amounts, percentages, share
counts) rather than prose verbatim — minor wording drift is acceptable
(LLM paraphrase), but if NONE of the quote's distinctive numbers appear
in the source, the attribution is fabricated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Mapping from agent-name (lowercase, normalised) to the source file we
# expect their actual output to live in.
_AGENT_TO_FILE: dict[str, str] = {
    "market analyst": "analyst_market.md",
    "news analyst": "analyst_news.md",
    "social analyst": "analyst_social.md",
    "social media analyst": "analyst_social.md",
    "social sentiment analyst": "analyst_social.md",
    "fundamentals analyst": "analyst_fundamentals.md",
    "aggressive analyst": "debate_risk.md",
    "conservative analyst": "debate_risk.md",
    "neutral analyst": "debate_risk.md",
    "bull researcher": "debate_bull_bear.md",
    "bear researcher": "debate_bull_bear.md",
    "research manager": "debate_bull_bear.md",
    "ta agent": "raw/technicals.md",
    "ta agent v2": "raw/technicals_v2.md",
    "trader": None,  # Trader output isn't a separate file; skip
    "portfolio manager": None,  # PM is the doc itself; skip self-attribution
}


@dataclass(frozen=True)
class AttributedQuote:
    """An italic quote with explicit agent attribution."""

    quote_text: str
    agent_name: str
    file: str
    line_no: int
    expected_source_file: str | None  # None = agent has no separate file


@dataclass(frozen=True)
class QuoteViolation:
    severity: Literal["MATERIAL", "MINOR"]
    type: Literal["fabricated_quote", "agent_source_missing"]
    file: str
    line_no: int
    agent_name: str
    quote_excerpt: str  # first 100 chars of the quote
    expected_source_file: str | None
    distinctive_numbers: list[str]
    matches_in_source: list[str]


# Pattern A: `**Agent Name:** *"..."*`
_PATTERN_PARA_ATTR = re.compile(
    r"\*\*(?P<agent>[A-Za-z][A-Za-z\s]{2,40}?)(?:\s*[:—–-]+|\s+\(.*?\))?\*\*"
    r"\s*\*\"(?P<quote>[^\"]+)\"\*",
)

# Pattern B: `*"..."*` followed by `(Agent Name, ...)` or `(Agent Name)`
# (post-quote parenthetical attribution — the COIN-style fabrication shape)
_PATTERN_POST_PAREN_ATTR = re.compile(
    r"\*\"(?P<quote>[^\"]+)\"\*"
    r"\s*\(\s*(?P<agent>[A-Za-z][A-Za-z\s]{2,40}?)"
    r"(?:[,;)]|\s+(?:transcript|paragraph|opening|closing|draft))",
    re.DOTALL,
)

# Pattern C: `*"..."* — Agent Name`
_PATTERN_DASH_ATTR = re.compile(
    r"\*\"(?P<quote>[^\"]+)\"\*"
    r"\s*[—–-]+\s*(?P<agent>[A-Za-z][A-Za-z\s]{2,40}?)(?:\.|$|\n)",
)

# Numerical fingerprints to extract from a quote — distinctive, easy to
# match against source prose.
_PATTERN_NUMBER = re.compile(
    r"\$[\d,]+(?:\.\d+)?[BM]?"  # $192.96, $4.08B, $14M
    r"|"
    r"[\d,]+(?:\.\d+)?M"        # 14.39M (shares / users)
    r"|"
    r"[\d.]+%"                   # 38.5%, 24.2%
    r"|"
    r"[\d.]+x"                   # 12x, 1.5x (ratios)
)


def _line_no(text: str, char_offset: int) -> int:
    return text[:char_offset].count("\n") + 1


def _normalise_agent_name(raw: str) -> str:
    """Lowercase + strip + collapse whitespace + drop common decorations."""
    s = raw.strip().lower()
    # Strip trailing punctuation that may have leaked in
    s = re.sub(r"[:.,;—–-]+\s*$", "", s).strip()
    # Collapse runs of whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def extract_attributed_quotes(text: str) -> list[AttributedQuote]:
    """Find italic-quoted text with agent attribution in the surrounding prose.

    Returns deduplicated list ordered by document position. Quote text is
    stripped of the surrounding `*"..."*` decorators.
    """
    if not text:
        return []

    seen: set[tuple[int, str]] = set()
    quotes: list[AttributedQuote] = []

    for pat in (_PATTERN_PARA_ATTR, _PATTERN_POST_PAREN_ATTR, _PATTERN_DASH_ATTR):
        for m in pat.finditer(text):
            agent_raw = m.group("agent").strip()
            agent = _normalise_agent_name(agent_raw)
            if agent not in _AGENT_TO_FILE:
                continue  # skip unknown / non-agent attribution
            quote_text = m.group("quote").strip()
            line_no = _line_no(text, m.start())
            key = (line_no, quote_text[:50])
            if key in seen:
                continue
            seen.add(key)
            quotes.append(AttributedQuote(
                quote_text=quote_text,
                agent_name=agent,
                file="",  # filled by caller
                line_no=line_no,
                expected_source_file=_AGENT_TO_FILE[agent],
            ))

    # Sort by line number for stable output
    quotes.sort(key=lambda q: q.line_no)
    return quotes


def extract_distinctive_numbers(quote_text: str) -> list[str]:
    """Pull numerical fingerprints from a quote for source verification."""
    return _PATTERN_NUMBER.findall(quote_text)


def validate_attributed_quotes(
    quotes: list[AttributedQuote],
    run_dir: Path,
    min_distinctive_numbers: int = 2,
    threshold_match_ratio: float = 0.0,
) -> list[QuoteViolation]:
    """Verify each quote's distinctive numbers appear in the source file.

    Threshold logic:
    - If quote has FEWER than `min_distinctive_numbers` distinctive numbers,
      skip verification (too short to fingerprint reliably).
    - If quote has ≥`min_distinctive_numbers` numbers and `threshold_match_ratio`
      of them fail to appear in source → MATERIAL fabricated_quote.
      Default `threshold_match_ratio = 0.0` means "ALL distinctive numbers
      must be missing" — strict, no false positives. Operator can lower
      threshold for tighter matching.
    - If source file doesn't exist → MINOR agent_source_missing (caller
      can ignore for skipped analysts like Trader / Portfolio Manager).
    """
    violations: list[QuoteViolation] = []
    # Per-source-file content cache to avoid re-reading per quote
    source_cache: dict[str, str] = {}

    for q in quotes:
        if q.expected_source_file is None:
            continue  # skip analysts with no separate source file

        if q.expected_source_file not in source_cache:
            path = run_dir / q.expected_source_file
            if not path.exists():
                violations.append(QuoteViolation(
                    severity="MINOR",
                    type="agent_source_missing",
                    file=q.file,
                    line_no=q.line_no,
                    agent_name=q.agent_name,
                    quote_excerpt=q.quote_text[:100],
                    expected_source_file=q.expected_source_file,
                    distinctive_numbers=[],
                    matches_in_source=[],
                ))
                source_cache[q.expected_source_file] = ""  # avoid re-warn
                continue
            source_cache[q.expected_source_file] = path.read_text(encoding="utf-8")

        source_text = source_cache[q.expected_source_file]
        if not source_text:
            continue  # already flagged above

        numbers = extract_distinctive_numbers(q.quote_text)
        if len(numbers) < min_distinctive_numbers:
            continue  # too few fingerprints to be confident

        matches = [n for n in numbers if n in source_text]
        match_ratio = len(matches) / len(numbers)

        if match_ratio <= threshold_match_ratio:
            violations.append(QuoteViolation(
                severity="MATERIAL",
                type="fabricated_quote",
                file=q.file,
                line_no=q.line_no,
                agent_name=q.agent_name,
                quote_excerpt=q.quote_text[:100],
                expected_source_file=q.expected_source_file,
                distinctive_numbers=numbers,
                matches_in_source=matches,
            ))

    return violations


def render_quote_violations_text(violations: list[QuoteViolation]) -> str:
    if not violations:
        return "QUOTE VALIDATION PASS: 0 violations"
    lines = [f"QUOTE VALIDATION FAIL: {len(violations)} violation(s)"]
    for v in violations:
        loc = f"{v.file or '?'}:{v.line_no}"
        lines.append(f"  [{v.severity}] {loc}  {v.type}  ({v.agent_name})")
        if v.type == "fabricated_quote":
            lines.append(
                f"    quote: \"{v.quote_excerpt}{'...' if len(v.quote_excerpt) >= 100 else ''}\""
            )
            lines.append(
                f"    distinctive numbers: {v.distinctive_numbers} "
                f"(matches in {v.expected_source_file}: {v.matches_in_source or 'NONE'})"
            )
        elif v.type == "agent_source_missing":
            lines.append(
                f"    expected source: {v.expected_source_file} (file missing)"
            )
    return "\n".join(lines)
