"""Phase 7.5: validate `$X net debt` / `$X net cash` claims against cells.

The MSFT 2026-05-07 audit surfaced a new failure pattern: the
deterministic Phase 6.8 block fixed the *label* on the authoritative
figure (correctly stating `Authoritative Net Debt: $8.16B`), but
downstream LLM prose freelances DIFFERENT definitions:

  pm_brief.md:                "Authoritative Net Debt: $8.16B"
                               (yfinance row, includes capital leases)
  decision.md:                "$78,272M − $40,262M = $38,010M net cash"
                               (excludes capital leases)
  analyst_fundamentals.md:    "Total Debt $56.97B − Cash+STI $78.23B
                               = $21.3B net cash" (includes leases)
  decision_executive.md:      "$38.0B cash-only net cash position"
                               (mislabels — actually includes ST inv)

All three are arithmetically valid against different debt baselines —
none are fabrications. But a stakeholder reading the executive section
sees `$38.0B net cash` and the pm_brief block sees `Net Debt $8.16B`;
the figures don't reconcile.

This validator extracts every `<value> net debt` / `<value> net cash`
claim from LLM outputs and verifies it derives from raw/net_debt.json
cells via SOME defensible computation. Claims that don't match any
canonical derivation are flagged as MATERIAL `definitional_drift`.

Acceptable derivations (signs as POSITIVE magnitudes; the claim's
sign is captured by the `is_cash` flag):

  - yfinance Net Debt row (canonical)
  - Total Debt − Cash And Cash Equivalents
  - Total Debt − (Cash + Short Term Investments)
  - (Long Term Debt + Current Debt) − Cash         [excludes leases]
  - (Long Term Debt + Current Debt) − (Cash + STI) [excludes leases]
  - Cash And Cash Equivalents − Total Debt         [net-cash framing]
  - (Cash + STI) − Total Debt                      [net-cash framing]
  - (Cash + STI) − (LTD + CD)                      [excl-leases NC]

Tolerance: 5% relative or $0.5B absolute, whichever is larger.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class NetDebtClaim:
    """A claim of `$X net debt` or `$X net cash`."""

    label: str  # exact label as written ("net debt", "net cash", etc.)
    is_cash: bool  # True for net-cash framing, False for net-debt
    value_raw: str  # "$8.16B" or "$78,272M"
    value_dollars: float  # converted to raw dollars (positive magnitude)
    file: str
    line_no: int
    match_text: str


@dataclass(frozen=True)
class NetDebtViolation:
    severity: Literal["MATERIAL", "MINOR"]
    type: Literal["definitional_drift", "no_net_debt_data", "skipped_non_usd_reporter"]
    file: str
    line_no: int
    claimed_label: str
    claimed_value: str
    claimed_dollars: float
    closest_canonical: float | None  # None for no_net_debt_data / skipped_non_usd_reporter
    closest_derivation: str | None
    delta_dollars: float | None
    match_text: str


# Match `$X net debt`, `$X net cash`, `net debt of $X`, etc.
# Allow common dollar formats: $X.XB, $X.XM, $X,XXX,XXXM, $XB.
# Bridge `[^\n.|]{0,30}?` allows phrasings like "$38.0B cash-only net cash"
# or "$190B in net debt" — small bridge defends against sentence-spanning.
#
# Phase 7.5 v1.3 (RC-A1): negative lookbehind `(?<![A-Za-z])` skips `$`
# preceded by a letter — defends against non-USD prefixes like `NT$`
# (TWD), `HK$` (HKD), `C$` (CAD), `S$` (SGD). The validator's canonical
# is USD-only; foreign-currency claims are out of scope.
#
# Phase 7.5 v1.3 (RC-B): bridge excludes `|` so the regex doesn't pair
# `net debt | $27.52` across markdown table cell boundaries.
# Phase 7.5 v1.4 (RC for AAPL 2026-05-08 false positive): bridge also
# excludes `;` so a value followed by a source citation (e.g.,
# `... Cash $45.57B; source: yfinance Net Debt row`) doesn't get paired
# with "Net Debt" from the citation. Symmetric with `_PATTERN_LABEL_FIRST`.
_PATTERN_VALUE_FIRST = re.compile(
    r"(?<![A-Za-z])\$(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>[BM])?"
    r"(?P<bridge>[^\n.;|]{0,30}?)"
    r"\s+(?P<label>net\s+(?:cash|debt))",
    re.IGNORECASE,
)
_PATTERN_LABEL_FIRST = re.compile(
    r"(?P<label>net\s+(?:cash|debt))"
    # Tighter bridge (20 chars) to defend against pairings like
    # `"net cash" and stops; the data shows that $16.70B of lease
    # obligations` — that's 33 chars and pairs the wrong dollar
    # figure across a semicolon. 20 chars covers `of $X`, `position
    # of $X`, `: $X`, `at $X` legitimate forms.
    # `|` excluded for v1.3 markdown-table-cell defense.
    r"[^\n.;|]{0,20}?"
    r"(?<![A-Za-z])\$(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>[BM])?",
    re.IGNORECASE,
)


def _line_no(text: str, char_offset: int) -> int:
    return text[:char_offset].count("\n") + 1


def _to_dollars(value: str, unit: str | None) -> float | None:
    """Convert a `$X.XB` / `$X.XM` / `$XXX,XXX` string to raw dollars."""
    try:
        num = float(value.replace(",", ""))
    except ValueError:
        return None
    u = (unit or "").upper()
    if u == "B":
        return num * 1_000_000_000
    if u == "M":
        return num * 1_000_000
    # No unit suffix — treat as raw dollars (rare but possible)
    return num


def extract_net_debt_claims(text: str) -> list[NetDebtClaim]:
    """Find `$X net debt` / `$X net cash` claims in markdown."""
    if not text:
        return []

    seen: set[tuple[int, int]] = set()
    claims: list[NetDebtClaim] = []

    for pat in (_PATTERN_VALUE_FIRST, _PATTERN_LABEL_FIRST):
        for m in pat.finditer(text):
            line_no = _line_no(text, m.start())
            key = (line_no, m.start())
            if key in seen:
                continue
            seen.add(key)

            value_str = m.group("value")
            unit = m.group("unit") or ""
            label = re.sub(r"\s+", " ", m.group("label").strip().lower())
            value_raw = f"${value_str}{unit}"

            value_dollars = _to_dollars(value_str, unit)
            if value_dollars is None:
                continue
            value_dollars = abs(value_dollars)

            # Phase 7.5 v1.2: capture the FULL surrounding paragraph (back to
            # last newline) so peer-attribution detection can find ticker
            # prefixes that appear earlier in the same sentence. The prior
            # 30-char window missed cases like
            #   "FN trades at 36.6x forward with $956M in net cash"
            # where "FN" is ~37 chars before "$956M".
            paragraph_start = text.rfind("\n", 0, m.start()) + 1
            paragraph_end = text.find("\n", m.end())
            if paragraph_end == -1:
                paragraph_end = len(text)
            match_text = text[paragraph_start:paragraph_end].replace("\n", " ").strip()

            claims.append(NetDebtClaim(
                label=label,
                is_cash="cash" in label,
                value_raw=value_raw,
                value_dollars=value_dollars,
                file="",  # filled by caller
                line_no=line_no,
                match_text=match_text,
            ))

    return claims


def _build_canonical_derivations(net_debt: dict) -> list[tuple[str, float]]:
    """Return all defensible net-debt/net-cash positive magnitudes from
    raw/net_debt.json cells. Each entry is (derivation_label, magnitude).
    """
    if not isinstance(net_debt, dict) or net_debt.get("unavailable"):
        return []

    def _f(key: str) -> float:
        v = net_debt.get(key)
        return float(v) if v is not None else 0.0

    td = _f("total_debt")
    ltd = _f("long_term_debt")
    cd = _f("current_debt")
    cl = _f("capital_lease_obligations")
    cash = _f("cash_and_equivalents")
    cash_sti = _f("cash_plus_short_term_investments") or cash
    nd_yf = _f("net_debt")

    candidates: list[tuple[str, float]] = []

    # yfinance Net Debt row (the canonical authoritative figure)
    if nd_yf:
        candidates.append(("yfinance Net Debt row", abs(nd_yf)))
    # Total Debt − Cash
    if td and cash:
        candidates.append(("Total Debt − Cash", abs(td - cash)))
    # Total Debt − (Cash + STI)
    if td and cash_sti and cash_sti != cash:
        candidates.append(("Total Debt − (Cash + STI)", abs(td - cash_sti)))
    # (LTD + CD) − Cash    [excludes capital leases]
    if (ltd or cd) and cash:
        candidates.append(("(LTD + CD) − Cash [excl leases]", abs((ltd + cd) - cash)))
    # (LTD + CD) − (Cash + STI)
    if (ltd or cd) and cash_sti and cash_sti != cash:
        candidates.append(("(LTD + CD) − (Cash + STI) [excl leases]", abs((ltd + cd) - cash_sti)))
    # Total Debt alone (sometimes cited as gross leverage, mistakenly framed as net)
    if td:
        candidates.append(("Total Debt", td))
    # Cash + STI alone (ditto, framed as net-cash position)
    if cash_sti:
        candidates.append(("Cash + STI", cash_sti))

    return candidates


def _within_tolerance(claimed: float, canonical: float) -> bool:
    """5% relative OR $0.5B absolute — whichever is larger."""
    rel = 0.05 * max(abs(claimed), abs(canonical))
    abs_tol = 5e8  # $0.5B
    tolerance = max(rel, abs_tol)
    return abs(claimed - canonical) <= tolerance


# RMBS 2026-05-08 false positive: peer bullet line "MRVL: net debt of $1.83B"
# was not recognized as peer-attributed because `:` wasn't a delimiter.
# v1.5 adds `:` so colon-delimited table-row forms (`MRVL: net debt ...`,
# `MU: net debt ...`) bind correctly to the peer ticker.
_PEER_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}(?:'s|\s|:)")


def _claim_attributed_to_other_ticker(match_text: str, main_ticker: str | None) -> bool:
    """Heuristic: is the claim's surrounding prose attributing it to a
    DIFFERENT ticker than the main one (e.g., "ORCL's net debt $96.15B"
    in a MSFT report)? If so, skip — peer net-debt should be validated
    against peer_ratios.json (Phase 7.3), not the main ticker's cells.

    Conservative scan: looks for an uppercase 2-5-letter token followed
    by `'s` or whitespace in the match_text. If the only ticker found is
    the main ticker, allow. If a different ticker appears, skip.
    """
    main_upper = (main_ticker or "").upper()
    found_tickers = set()
    for m in _PEER_TICKER_PATTERN.finditer(match_text):
        # The `\b[A-Z]{2,5}` part is what we want; rebuild without trailing
        token = re.match(r"[A-Z]{2,5}", m.group(0)).group(0)
        # Filter out common non-ticker uppercase words
        if token in {"USD", "GAAP", "ARR", "CC", "EBITDA", "FCF", "AI",
                     "PM", "TA", "CEO", "CFO", "API", "FY", "OCF",
                     "NTM", "TTM", "SOTP", "NDR", "RPO", "SBC", "MA",
                     "II", "QC", "RM", "SEC", "USA", "NYC"}:
            continue
        found_tickers.add(token)
    if not found_tickers:
        return False  # no ticker context — assume main-ticker claim
    if found_tickers == {main_upper}:
        return False  # only the main ticker mentioned — validate
    # At least one OTHER ticker present — peer-attributed claim, skip
    return True


def validate_net_debt_claims(
    claims: list[NetDebtClaim],
    net_debt_json_path: Path,
    main_ticker: str | None = None,
) -> list[NetDebtViolation]:
    """Verify each net-debt/net-cash claim derives from raw/net_debt.json
    cells via some defensible computation.

    `main_ticker`: when provided, skips claims whose surrounding prose
    attributes the figure to a DIFFERENT ticker (those are peer claims
    and should be validated against peer_ratios.json by Phase 7.3, not
    against the main ticker's net_debt.json by this validator).
    """
    if not claims:
        return []

    if not net_debt_json_path.exists():
        return [NetDebtViolation(
            severity="MINOR",
            type="no_net_debt_data",
            file=claims[0].file,
            line_no=0,
            claimed_label="",
            claimed_value="",
            claimed_dollars=0.0,
            closest_canonical=None,
            closest_derivation=None,
            delta_dollars=None,
            match_text=f"{net_debt_json_path} missing",
        )]

    try:
        net_debt = json.loads(net_debt_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    # Phase 7.5 v1.3 (RC-A2): yfinance returns balance-sheet cells in the
    # company's reporting currency (TWD for Taiwan-domiciled, JPY for
    # Japan-domiciled, etc.). The validator's canonical comparison is
    # USD-only; comparing TWD-denominated cells to USD claims produces
    # noise (ASE Tech: yfinance Net Debt $169B "USD" is actually TWD
    # 169B, ~$5.3B USD). Skip validation entirely with a single MINOR
    # notice when the reporter is non-USD. Backwards compatibility:
    # missing `financial_currency` field is treated as USD (legacy runs).
    reporting_currency = net_debt.get("financial_currency")
    if reporting_currency and str(reporting_currency).upper() != "USD":
        return [NetDebtViolation(
            severity="MINOR",
            type="skipped_non_usd_reporter",
            file=claims[0].file,
            line_no=0,
            claimed_label="",
            claimed_value="",
            claimed_dollars=0.0,
            closest_canonical=None,
            closest_derivation=None,
            delta_dollars=None,
            match_text=(
                f"reporting currency {reporting_currency!r} ≠ USD; "
                f"net-debt validation requires USD-denominated canonical. "
                f"{len(claims)} claim(s) not validated."
            ),
        )]

    canonicals = _build_canonical_derivations(net_debt)
    if not canonicals:
        return []  # net_debt cells unavailable; can't validate

    violations: list[NetDebtViolation] = []
    for claim in claims:
        # Skip claims attributed to peer tickers (handled by Phase 7.3)
        if _claim_attributed_to_other_ticker(claim.match_text, main_ticker):
            continue
        # Find closest canonical derivation
        best_label, best_val = min(
            canonicals,
            key=lambda c: abs(claim.value_dollars - c[1]),
        )
        if _within_tolerance(claim.value_dollars, best_val):
            continue  # claim matches a defensible derivation
        # Drift: doesn't match any canonical derivation within tolerance
        violations.append(NetDebtViolation(
            severity="MATERIAL",
            type="definitional_drift",
            file=claim.file,
            line_no=claim.line_no,
            claimed_label=claim.label,
            claimed_value=claim.value_raw,
            claimed_dollars=claim.value_dollars,
            closest_canonical=best_val,
            closest_derivation=best_label,
            delta_dollars=abs(claim.value_dollars - best_val),
            match_text=claim.match_text,
        ))
    return violations


def render_net_debt_violations_text(violations: list[NetDebtViolation]) -> str:
    if not violations:
        return "NET-DEBT VALIDATION PASS: 0 violations"
    # When the only entry is the non-USD skip notice, render as a notice,
    # not a FAIL — it's an explicit out-of-scope, not a validation failure.
    if (
        len(violations) == 1
        and violations[0].type == "skipped_non_usd_reporter"
    ):
        return (
            f"NET-DEBT VALIDATION SKIPPED (non-USD reporter): "
            f"{violations[0].match_text}"
        )
    lines = [f"NET-DEBT VALIDATION FAIL: {len(violations)} violation(s)"]
    for v in violations:
        loc = f"{v.file or '?'}:{v.line_no}"
        lines.append(f"  [{v.severity}] {loc}  {v.type}")
        if v.type == "definitional_drift":
            lines.append(
                f"    claimed: {v.claimed_value} {v.claimed_label}"
            )
            lines.append(
                f"    closest canonical: ~${v.closest_canonical / 1e9:.2f}B "
                f"({v.closest_derivation})"
            )
            lines.append(
                f"    delta: ${v.delta_dollars / 1e9:.2f}B (>5% relative / $0.5B abs)"
            )
        lines.append(f"    text: {v.match_text[:120]}")
    return "\n".join(lines)
