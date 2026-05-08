"""Extract date+close claims from LLM-authored markdown for verification.

The COIN 2026-05-08 audit surfaced fabricated `<DATE> close $X.XX` claims
across decision.md / decision_executive.md / technicals_v2.md / debate
documents. Examples that need to be caught:

  - "the May 8 session subsequently closed at $206.50"
  - "May 8 close $206.50 on 14.39M shares"
  - "technicals through the 2026-05-08 close (per prices.json: open
     $205.31, high $210.47, low $202.81, close $206.50, volume 14,390,000)"
  - "May 7 reference; May 8 closed $206.50"

The extractor uses multiple narrow patterns rather than one greedy regex
because LLM prose varies. False-negatives are acceptable (other
validators will catch the same drift later); false-positives waste
operator time and erode trust in the validator.

Each extracted claim carries:
  - date_raw: the date string as written ("May 8", "2026-05-08")
  - price: the dollar amount as written
  - file/line context for the violation report
  - match_text: the surrounding sentence (for human review)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date, datetime
from typing import Iterator


_MONTH_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


@dataclass(frozen=True)
class DateCloseClaim:
    """A claim of the form `<DATE> close $X.XX` found in LLM output."""

    date_raw: str          # "May 8", "2026-05-07", etc.
    date_iso: str | None   # resolved to "YYYY-MM-DD" if possible
    price: float           # the dollar amount
    match_text: str        # surrounding sentence for context
    line_no: int           # 1-indexed line in the source document
    file: str | None = None  # filled in by the caller


def _resolve_iso(date_raw: str, anchor_year: int) -> str | None:
    """Resolve a date string to ISO format `YYYY-MM-DD`.

    Accepts:
      - "2026-05-07" → "2026-05-07"
      - "May 8" → uses anchor_year ("2026-05-08")
      - "May 8, 2026" → "2026-05-08"

    Returns None if ambiguous or unparseable.
    """
    s = date_raw.strip()
    # ISO format
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return _date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    # "Month DAY" or "Month DAY, YEAR"
    m = re.fullmatch(
        r"(\w+)\s+(\d{1,2})(?:,?\s+(\d{4}))?",
        s,
    )
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else anchor_year
        month = _MONTH_TO_NUM.get(month_name)
        if month is None:
            return None
        try:
            return _date(year, month, day).isoformat()
        except ValueError:
            return None
    return None


_PATTERN_DATE_CLOSE = re.compile(
    # group "date": ISO `YYYY-MM-DD` OR `Month DAY[, YEAR]?`
    r"(?P<date>"
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?"
    r")"
    # bridge: up to ~80 chars, no newline, no period (no sentence break)
    r"(?P<bridge>[^\n.]{0,80}?)"
    # close|closed|closes|closing (word boundary), then optional connecting
    # words like "at" / "was" / "the session at"; allow $ to follow directly
    r"\b(?:close|closed|closes|closing)\b"
    r"(?:\s+(?:at|was|the\s+session\s+(?:at|was)|to|near))?"
    r"\s*\$(?P<price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


_PATTERN_PRICES_JSON_TUPLE = re.compile(
    # Matches "(per prices.json: open $X, high $X, low $X, close $X.XX, volume X)"
    # which is the most explicit fabrication shape from the COIN run.
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+close[^\n]{0,30}?"
    r"\(per prices\.json[^\n]*?close\s+\$(?P<price>[\d,]+\.?\d*)",
    re.IGNORECASE,
)


# Phase 7.1 v2 (Fix #11): when the bridge between an outer date and the
# close contains ANOTHER date reference (parenthetical or semicolon-
# separated), the close belongs to the INNER date, not the outer.
# Examples from the MSFT 2026-05-08 run:
#   - "Jan 7, 2026 peak before the FY26 Q2 earnings crash (Jan 29, 2026:
#     close $432.51 on 128.9M shares)"  →  close belongs to Jan 29
#   - "intraday high range of May 7 session; Apr 22 close at $432.92"
#     →  close belongs to Apr 22
# This pattern matches a date occurring anywhere; we scan the bridge for
# the LAST occurrence (closest to the close) and use that.
_PATTERN_DATE_ONLY = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?",
    re.IGNORECASE,
)


def _line_no(text: str, char_offset: int) -> int:
    """1-indexed line number for a character offset."""
    return text[:char_offset].count("\n") + 1


def extract_date_close_claims(text: str, anchor_year: int = 2026) -> list[DateCloseClaim]:
    """Scan markdown text for `<DATE> ... close $X.XX` patterns.

    `anchor_year` resolves "May 8"-style dates without explicit year. Defaults
    to 2026 since that's the active research date range; callers can override
    when scanning historical reports.

    Returns a list of structured claims (one per match). Order matches the
    document order so the violation report can list them in reading order.
    """
    if not text:
        return []

    claims: list[DateCloseClaim] = []
    seen: set[tuple[int, int]] = set()  # (line_no, char_offset) — dedupe overlapping patterns

    for pat in (_PATTERN_DATE_CLOSE, _PATTERN_PRICES_JSON_TUPLE):
        for m in pat.finditer(text):
            try:
                price = float(m.group("price").replace(",", ""))
            except (ValueError, AttributeError):
                continue
            line_no = _line_no(text, m.start())
            key = (line_no, m.start())
            if key in seen:
                continue
            seen.add(key)
            date_raw = m.group("date")

            # Phase 7.1 v2: if the bridge contains another date reference
            # (parenthetical or semicolon-separated), bind the close to the
            # LAST date in the bridge — that's the date semantically nearest
            # the close. Drops false positives like "Jan 7 peak (Jan 29:
            # close $432.51)" pairing Jan 7 with $432.51.
            try:
                bridge = m.group("bridge")
            except IndexError:
                bridge = ""
            if bridge:
                inner_dates = list(_PATTERN_DATE_ONLY.finditer(bridge))
                if inner_dates:
                    date_raw = inner_dates[-1].group(0)

            iso = _resolve_iso(date_raw, anchor_year)
            # Capture surrounding sentence (up to 120 chars) for human review
            ctx_start = max(0, m.start() - 20)
            ctx_end = min(len(text), m.end() + 20)
            match_text = text[ctx_start:ctx_end].replace("\n", " ").strip()
            claims.append(DateCloseClaim(
                date_raw=date_raw,
                date_iso=iso,
                price=price,
                match_text=match_text,
                line_no=line_no,
            ))

    return claims


def iter_date_close_claims_in_files(file_paths: list[str], anchor_year: int = 2026) -> Iterator[DateCloseClaim]:
    """Convenience: scan multiple files and yield claims with `file` filled in.

    Skips files that don't exist. Logs nothing — callers decide what to do
    with the structured output.
    """
    from pathlib import Path
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in extract_date_close_claims(text, anchor_year=anchor_year):
            yield DateCloseClaim(
                date_raw=claim.date_raw,
                date_iso=claim.date_iso,
                price=claim.price,
                match_text=claim.match_text,
                line_no=claim.line_no,
                file=str(path),
            )
