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


_QUOTE_RE = re.compile(r"[\"“”‘’«»]([^\"“”‘’«»]{15,})[\"“”‘’«»]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _citation_substantiated(line: str, sec_filing_text: str) -> bool:
    """True when the Note-citing line carries a verbatim quote that is
    actually present in the filing text.

    RKLB 2026-05-28: the fetcher tagged the 10-Q with the XBRL-warning header
    yet the body DID contain the cited note prose ("Remaining backlog totaled
    $ 2,219,756 ..."). A citation whose quoted text is found in the filing is
    substantiated, not fabricated — so it must not be flagged. A genuinely
    fabricated quote (AAPL 2026-05-29) is absent and still flags. Matching is
    whitespace- and punctuation-tolerant (the filing renders "$ 2,219,756"
    while the report writes "$2,219,756K"): we look for a contiguous run of
    the quote's letters/spaces in the filing."""
    sec_alpha = _norm(re.sub(r"[^A-Za-z ]", " ", sec_filing_text)).lower()
    for q in _QUOTE_RE.findall(line):
        # Prefer the alphabetic phrase BEFORE the first digit — number
        # formatting differs between the filing ("$ 2,219,756") and the report
        # ("$2,219,756K"), but the surrounding prose ("Remaining backlog
        # totaled") matches verbatim.
        prefix = re.split(r"\d", q, 1)[0]
        for candidate in (prefix, q):
            alpha = _norm(re.sub(r"[^A-Za-z ]", " ", candidate)).lower()
            if len(alpha) >= 15 and alpha[:60] in sec_alpha:
                return True
    return False


def validate_filing_attribution(
    text: str, file_label: str, sec_filing_text: str | None
) -> list[FilingAttributionViolation]:
    """Flag `Note <N>` citations when the filing is an XBRL stub.

    Returns [] when the filing is NOT a stub (readable filings legitimately
    carry numbered prose footnotes a report may cite). A Note citation whose
    verbatim quote is found in the filing is substantiated and not flagged
    even under the XBRL-stub heuristic (the warning header can co-exist with
    readable note prose)."""
    if not text or not filing_is_xbrl_stub(sec_filing_text):
        return []
    violations: list[FilingAttributionViolation] = []
    for m in _NOTE_RE.finditer(text):
        ln = _line_no(text, m.start())
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.end())
        if le == -1:
            le = len(text)
        line = text[ls:le]
        if sec_filing_text and _citation_substantiated(line, sec_filing_text):
            continue
        violations.append(FilingAttributionViolation(
            severity="MATERIAL",
            type="fabricated_note_citation",
            file=file_label,
            line_no=ln,
            match_text=line.strip()[:160],
        ))
    return violations


# Ordered transforms that remove a fabricated "Note N" prose attribution while
# keeping the (XBRL-readable) number and any legitimate filing reference. Applied
# ONLY when the filing is an XBRL stub.
_NOTE_STRIP_TRANSFORMS: list[tuple[str, str]] = [
    # "raw/sec_filing.md Note 7" → "raw/sec_filing.md" (PDF scrub later → "the 10-Q text")
    (r"\braw/sec_filing\.md\s+Note\s+\d+[A-Za-z]?\b", "raw/sec_filing.md"),
    # "per Subsequent Event Note 15" / "per Note 5" → "per the 10-Q"
    (r"\bper\s+(?:[A-Z][a-zA-Z ]*?\s+)?Note\s+\d+[A-Za-z]?\b", "per the 10-Q"),
    # "Note 5 discloses/shows/states/reports/details" → "the 10-Q discloses ..."
    (r"\bNote\s+\d+[A-Za-z]?\s+(disclos\w+|show\w+|state\w+|report\w+|detail\w+)\b",
     r"the 10-Q \1"),
    # bare parenthetical "(Note 5)" / "( Note 5 )" → drop entirely
    (r"\s*\(\s*Note\s+\d+[A-Za-z]?\s*\)", ""),
    # catch-all: any remaining "Note 5" → "the 10-Q"
    (r"\bNote\s+\d+[A-Za-z]?\b", "the 10-Q"),
]


def strip_fabricated_note_citations(
    text: str, sec_filing_text: str | None
) -> tuple[str, int]:
    """Remove fabricated `Note N` prose attributions when the filing is an
    XBRL stub. The cited NUMBERS are XBRL-readable and stay; only the
    unreadable-Note-prose citation is rewritten to a generic 10-Q reference.

    Returns (new_text, n_replacements). No-op when the filing is not a stub."""
    if not text or not filing_is_xbrl_stub(sec_filing_text):
        return text, 0
    import re as _re
    n = 0
    out = text
    for pattern, repl in _NOTE_STRIP_TRANSFORMS:
        out, k = _re.subn(pattern, repl, out)
        n += k
    return out, n


def strip_note_citations_in_run(run_dir) -> dict:
    """Apply strip_fabricated_note_citations to decision.md /
    decision_executive.md in a run dir (in place) when raw/sec_filing.md is an
    XBRL stub. Returns a summary dict. Safe no-op for readable filings."""
    from pathlib import Path
    run = Path(run_dir)
    sec = run / "raw" / "sec_filing.md"
    sec_text = sec.read_text(encoding="utf-8") if sec.exists() else None
    total = 0
    files_changed = []
    # Cover every report file the validator scans AND the PDF renders — a
    # surviving "Note N" in an appendix (analyst/debate) file leaks to the
    # customer PDF and trips the validator just the same.
    _files = (
        "decision.md", "decision_executive.md",
        "debate_bull_bear.md", "debate_risk.md",
        "analyst_market.md", "analyst_news.md",
        "analyst_social.md", "analyst_fundamentals.md",
        "raw/technicals.md", "raw/technicals_v2.md",
    )
    if filing_is_xbrl_stub(sec_text):
        for fname in _files:
            fp = run / fname
            if not fp.exists():
                continue
            new, n = strip_fabricated_note_citations(fp.read_text(encoding="utf-8"), sec_text)
            if n:
                fp.write_text(new, encoding="utf-8")
                files_changed.append(fname)
                total += n
    return {"total_stripped": total, "files_changed": files_changed}


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
