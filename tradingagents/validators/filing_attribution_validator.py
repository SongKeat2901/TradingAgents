"""Phase 9 P3 — filing-attribution validator.

The 2026-05-29 AAPL run graded F because it cited "Note 2 / Note 5 / Note 7"
as if quoting prose footnotes of the 10-Q — but that filing is inline-XBRL
encoded and carries NO readable footnote prose. The deterministic SEC-filing
block already warns the LLM ("Do NOT cite specific Note numbers ... any
citation that pretends to read the Note's text is fabricated attribution"),
but the LLM ignored it. This validator catches the violation post-output so
it gates A+ delivery.

Detection: a filing is an XBRL stub when raw/sec_filing.md contains the
"XBRL ENCODING WARNING" marker. When it is, any `Note <N>` citation in the
report is fabricated attribution to a footnote that isn't readable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from tradingagents.validators._helpers import line_no as _line_no

_XBRL_STUB_MARKER = "XBRL ENCODING WARNING"
# "Note 5", "Note 15", "Note 14 — Subsequent Events", "Note 2a" ...
_NOTE_RE = re.compile(r"\bNote\s+\d+[A-Za-z]?\b")


@dataclass(frozen=True)
class FilingAttributionViolation:
    severity: Literal["MATERIAL"]
    type: Literal["fabricated_note_citation"]
    file: str
    line_no: int
    match_text: str


def filing_is_xbrl_stub(sec_filing_text: str | None) -> bool:
    """True when the filing is inline-XBRL with no readable footnote prose."""
    return bool(sec_filing_text) and _XBRL_STUB_MARKER in sec_filing_text


def validate_filing_attribution(
    text: str, file_label: str, sec_filing_text: str | None
) -> list[FilingAttributionViolation]:
    """Flag `Note <N>` citations when the filing is an XBRL stub.

    Returns [] when the filing is NOT a stub (readable filings legitimately
    carry numbered prose footnotes a report may cite)."""
    if not text or not filing_is_xbrl_stub(sec_filing_text):
        return []
    violations: list[FilingAttributionViolation] = []
    for m in _NOTE_RE.finditer(text):
        ln = _line_no(text, m.start())
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        if le == -1:
            le = len(text)
        violations.append(FilingAttributionViolation(
            severity="MATERIAL",
            type="fabricated_note_citation",
            file=file_label,
            line_no=ln,
            match_text=text[ls:le].strip()[:160],
        ))
    return violations


def render_filing_attribution_violations_text(
    violations: list[FilingAttributionViolation],
) -> str:
    if not violations:
        return "FILING ATTRIBUTION PASS: 0 violations"
    lines = [f"FILING ATTRIBUTION FAIL: {len(violations)} violation(s)"]
    for v in violations:
        lines.append(
            f"  [{v.severity}] {v.file}:{v.line_no}  {v.type}\n    text: {v.match_text}"
        )
    return "\n".join(lines)
