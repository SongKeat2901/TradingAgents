"""Phase 7.11: extract convertible-note specifics from inline-XBRL 10-Q HTML.

The HTML stripper used in `sec_edgar.fetch_latest_filing` discards the
inline-XBRL `<ix:nonFraction>` tags and their dimension context — leaving
only the bare numeric values without the tranche attribution. For
companies with material outstanding convertibles (MARA, MSTR, RIOT,
COIN, etc.), this means the PM cannot see conversion prices,
face amounts, or coupon rates per tranche.

This module parses the raw inline-XBRL HTML directly (before stripping)
using regex on the inline-XBRL namespace tags. For each
`us-gaap:DebtInstrument*` fact whose context references a tranche-
specific Member (e.g. `mara:ConvertibleSeniorNotesDueDecember2026Member`),
we extract the value + scale + dimension and group by tranche.

Output is a list of tranche dicts suitable for serializing to
raw/convertibles.json and rendering into a markdown block appended to
pm_brief.md.

This is regex-based on purpose: no `lxml` / `bs4` dependency (which
would also lose namespace prefixes through HTML parsing). The inline-
XBRL tag shape is stable enough that regex is reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ConvertibleTranche:
    """One convertible-note tranche, populated from inline-XBRL facts."""

    tranche: str                            # e.g. "December2026"
    face_amount: float | None               # original principal (USD)
    conversion_price: float | None          # USD/share
    conversion_ratio: float | None          # shares per $1 principal
    interest_rate_stated: float | None      # decimal (e.g. 0.01 = 1%)
    interest_rate_effective: float | None   # decimal


# Inline-XBRL context block — `<xbrli:context id="...">...</xbrli:context>`
_CONTEXT_RE = re.compile(
    r"<xbrli:context\s+[^>]*id=\"([^\"]+)\"[^>]*>(.*?)</xbrli:context>",
    re.DOTALL,
)
# explicitMember inside a context defines a dimension value
_MEMBER_RE = re.compile(
    r"<xbrldi:explicitMember[^>]*>\s*([^<]+)\s*</xbrldi:explicitMember>",
)
# Inline fact: `<ix:nonFraction attrs>VALUE</ix:nonFraction>`
_FACT_RE = re.compile(
    r"<ix:nonFraction\s+([^>]+)>([^<]*)</ix:nonFraction>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r"(\w+)=\"([^\"]*)\"")

# Targets — facts we care about for convertible analysis
_TARGET_NAME_RE = re.compile(
    r"DebtInstrument(Face|Convertible|InterestRate)",
)
# Member token suffix identifying a convertible tranche
_TRANCHE_RE = re.compile(r"ConvertibleSeniorNotesDue([A-Za-z0-9]+?)Member")


def _parse_contexts(html: str) -> dict[str, list[str]]:
    """Build {context_id: [member_values]} for each `<xbrli:context>` block."""
    contexts: dict[str, list[str]] = {}
    for cid, body in _CONTEXT_RE.findall(html):
        members = [m.strip() for m in _MEMBER_RE.findall(body)]
        contexts[cid] = members
    return contexts


def _scaled_value(raw_text: str, scale_attr: str | None) -> float | None:
    """Apply the `scale` attribute (an integer power-of-ten) to the raw value.

    XBRL `scale="6"` on a value `1025.00` means $1,025,000,000 (=1025 × 10^6).
    Returns None when parsing fails.
    """
    try:
        scale = int(scale_attr) if scale_attr is not None else 0
    except (ValueError, TypeError):
        scale = 0
    try:
        cleaned = raw_text.strip().replace(",", "").replace("$", "")
        if not cleaned:
            return None
        return float(cleaned) * (10 ** scale)
    except ValueError:
        return None


def _fact_tranche(members: list[str]) -> str | None:
    """Return the tranche name (e.g. 'December2026') from a context's
    dimension members, or None if the context is not tranche-specific."""
    for mem in members:
        m = _TRANCHE_RE.search(mem)
        if m:
            return m.group(1)
    return None


def extract_convertibles_from_html(html: str) -> list[dict]:
    """Return a list of tranche dicts extracted from inline-XBRL `html`.

    Each dict has the shape of `ConvertibleTranche.__annotations__`. Empty
    list when no convertible facts are present (most 10-Qs of companies
    without convertibles).
    """
    if not html or "<ix:nonFraction" not in html:
        return []

    contexts = _parse_contexts(html)
    if not contexts:
        return []

    by_tranche: dict[str, dict[str, float | None]] = {}
    for attrs_text, val_text in _FACT_RE.findall(html):
        attrs = dict(_ATTR_RE.findall(attrs_text))
        name = attrs.get("name", "")
        if not _TARGET_NAME_RE.search(name):
            continue
        ctx_id = attrs.get("contextRef") or attrs.get("contextref")
        if not ctx_id:
            continue
        tranche = _fact_tranche(contexts.get(ctx_id, []))
        if not tranche:
            continue

        scaled = _scaled_value(val_text, attrs.get("scale"))
        short_name = name.split(":", 1)[-1]
        bucket = by_tranche.setdefault(tranche, {
            "face_amount": None,
            "conversion_price": None,
            "conversion_ratio": None,
            "interest_rate_stated": None,
            "interest_rate_effective": None,
        })
        # Map XBRL name → bucket key. Multiple facts can share name (across
        # periods); we keep the LARGER value for face (latest period tends to
        # use higher scale) and the FIRST seen for ratios/prices (stable).
        if "FaceAmount" in short_name:
            if bucket["face_amount"] is None or (scaled and scaled > bucket["face_amount"]):
                bucket["face_amount"] = scaled
        elif "ConversionPrice" in short_name:
            if bucket["conversion_price"] is None and scaled is not None:
                bucket["conversion_price"] = scaled
        elif "ConversionRatio" in short_name:
            if bucket["conversion_ratio"] is None and scaled is not None:
                bucket["conversion_ratio"] = scaled
        elif "InterestRateStatedPercentage" in short_name:
            if bucket["interest_rate_stated"] is None and scaled is not None:
                bucket["interest_rate_stated"] = scaled
        elif "InterestRateEffectivePercentage" in short_name:
            if bucket["interest_rate_effective"] is None and scaled is not None:
                bucket["interest_rate_effective"] = scaled

    # Sort by face amount descending (largest tranche first), tie-break alpha
    result = [
        ConvertibleTranche(tranche=t, **vals)
        for t, vals in by_tranche.items()
    ]
    result.sort(key=lambda x: (-(x.face_amount or 0), x.tranche))
    return [asdict(c) for c in result]


def _fmt_dollars(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    return f"${v:,.0f}"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.3f}%"


def format_convertibles_block(
    convertibles: list[dict],
    spot: float | None = None,
    ticker: str = "?",
) -> str:
    """Render the convertibles list as a markdown block for pm_brief.md.

    When `spot` is provided, also includes a "rally needed to reach ITM"
    column so downstream agents can quickly assess dilution risk per
    tranche. Returns empty string when `convertibles` is empty.
    """
    if not convertibles:
        return ""

    total_face = sum((t.get("face_amount") or 0) for t in convertibles)

    header = (
        f"\n\n## Convertible debt structure ({ticker}, extracted from inline-XBRL 10-Q via Phase 7.11)\n\n"
        "**This block carries authoritative per-tranche convertible specifics. "
        "Use these values verbatim for dilution math. Conversion prices "
        "are the strikes; only tranches with spot price ≥ conversion price "
        "will dilute via conversion at maturity (otherwise they require "
        "cash repayment or refinancing).**\n\n"
    )

    columns = ["Tranche", "Face amount", "Conversion price", "Coupon", "Effective rate"]
    if spot is not None:
        columns.append("Rally to ITM")
    sep = "|" + "|".join("---" for _ in columns) + "|"
    header_row = "| " + " | ".join(columns) + " |"

    rows = []
    for t in convertibles:
        cp = t.get("conversion_price")
        if spot is not None and cp:
            rally_pct = (cp - spot) / spot * 100
            rally_str = f"{rally_pct:+.1f}%" if rally_pct > 0 else "ITM"
        else:
            rally_str = ""
        row = [
            t.get("tranche") or "?",
            _fmt_dollars(t.get("face_amount")),
            f"${cp:.2f}" if cp else "n/a",
            _fmt_pct(t.get("interest_rate_stated")),
            _fmt_pct(t.get("interest_rate_effective")),
        ]
        if spot is not None:
            row.append(rally_str)
        rows.append("| " + " | ".join(row) + " |")

    summary = (
        f"\n**Total face amount (sum across tranches):** "
        f"{_fmt_dollars(total_face)}. "
        f"This is the **issuance** total — compare to balance-sheet "
        f"`Notes payable LT + current` to derive cumulative redeemed.\n"
    )

    return header + header_row + "\n" + sep + "\n" + "\n".join(rows) + "\n" + summary
