"""Validate peer-metric claims against raw/peer_ratios.json.

The 2026-05-06 MARA decision.md fabricated peer leverage and valuation
metrics with false attribution::

    "Per raw/peers.json: RIOT EV/EBITDA ~12×, ND/EBITDA <1×; CIFR
     ND/EBITDA ~1.5×, op margin low single-digit; CLSK op margin ~5%"

Actual peer_ratios.json: CIFR op margin = -383.42%, CLSK op margin =
-37.83%. The "~5%" claim was a sign-flip; the EV/EBITDA / ND/EBITDA
numbers don't even appear in any raw file. Phase 6.4 v2 (commit
bc7f289) added Net Debt / TTM EBITDA / ND/EBITDA cells — but the LLM
can still cite OTHER metrics not in the deterministic block.

This validator scans LLM outputs for `<TICKER> <metric> <value>`
patterns where TICKER is in raw/peers.json. For each:

- If the metric IS a peer_ratios.json column (capex/revenue, op margin,
  TTM PE, Forward PE, Net Debt, TTM EBITDA, ND/EBITDA): verify the
  cited value matches the cell within tolerance.
- If the metric is NOT in peer_ratios.json (EV/EBITDA, P/S, $/boe,
  etc.) AND the surrounding prose attributes it to peer_ratios.json
  or peers.json: flag as fabricated source attribution.

Verification tolerance: ±5% relative for ratios, ±$0.10B absolute for
$-figures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Metrics that ARE in peer_ratios.json columns (Phase 6.4 v2 schema).
# Map common LLM phrasings to the canonical field name in peer_ratios.json.
_VERIFIABLE_METRICS: dict[str, str] = {
    "ttm pe": "ttm_pe",
    "ttm p/e": "ttm_pe",
    "p/e (ttm)": "ttm_pe",
    "trailing p/e": "ttm_pe",
    "forward pe": "forward_pe",
    "forward p/e": "forward_pe",
    "fwd pe": "forward_pe",
    "fwd p/e": "forward_pe",
    "op margin": "latest_quarter_op_margin",
    "operating margin": "latest_quarter_op_margin",
    "capex/revenue": "latest_quarter_capex_to_revenue",
    "capex/rev": "latest_quarter_capex_to_revenue",
    "capex intensity": "latest_quarter_capex_to_revenue",
    "net debt": "net_debt",
    "ttm ebitda": "ttm_ebitda",
    "ebitda (ttm)": "ttm_ebitda",
    "nd/ebitda": "nd_ebitda",
    "net debt/ebitda": "nd_ebitda",
    "leverage": "nd_ebitda",
}

# Metrics that are NOT in peer_ratios.json — citing them as if they were
# is a fabricated source attribution.
_NON_VERIFIABLE_METRICS = {
    "ev/ebitda",
    "ev / ebitda",
    "enterprise value/ebitda",
    "p/s",
    "price/sales",
    "price-to-sales",
    "$/boe",
    "$ per boe",
    "fcf/share",
    "fcf yield",
    "dividend yield",
}


@dataclass(frozen=True)
class PeerMetricViolation:
    severity: Literal["MATERIAL", "MINOR"]
    type: Literal["wrong_peer_metric", "fabricated_metric_attribution"]
    file: str
    line_no: int
    ticker: str
    metric: str
    claimed_value: str
    actual_value: str | None  # None for fabricated_metric_attribution
    match_text: str


def _line_no(text: str, char_offset: int) -> int:
    return text[:char_offset].count("\n") + 1


def _load_peer_ratios(peer_ratios_path: Path) -> dict:
    if not peer_ratios_path.exists():
        return {}
    try:
        return json.loads(peer_ratios_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_peer_tickers(peers_path: Path) -> set[str]:
    if not peers_path.exists():
        return set()
    try:
        d = json.loads(peers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(d, dict):
        return set()
    return {k for k in d.keys() if isinstance(k, str) and k.isupper()}


def _parse_value(s: str) -> tuple[float | None, str]:
    """Parse `~12×`, `0.5x`, `38.5%`, `$1.5B`, `−$2.1B` etc. to (value, kind).

    `kind` ∈ {"ratio", "pct", "billions", "millions", "raw"}.
    Returns (None, "raw") if unparseable.
    """
    s = s.strip().lstrip("~≈").lstrip("approximately ")
    s = s.replace("−", "-").replace("—", "-")
    # Pct: "38.5%"
    m = re.fullmatch(r"(-?[\d.]+)\s*%", s)
    if m:
        try:
            return float(m.group(1)), "pct"
        except ValueError:
            return None, "raw"
    # Ratio: "12x" or "12×" or "12.5x"
    m = re.fullmatch(r"(-?[\d.]+)\s*[x×]", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "ratio"
        except ValueError:
            return None, "raw"
    # Dollars in billions: "$1.5B" or "−$2.1B"
    m = re.fullmatch(r"-?\$?(-?[\d.,]+)\s*B", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", "")), "billions"
        except ValueError:
            return None, "raw"
    # Dollars in millions
    m = re.fullmatch(r"-?\$?(-?[\d.,]+)\s*M", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", "")), "millions"
        except ValueError:
            return None, "raw"
    return None, "raw"


def _values_match(claimed: float, actual_raw: float, kind: str, tolerance: float = 0.05) -> bool:
    """Compare a claimed value to an actual value from peer_ratios.json.

    peer_ratios.json stores:
      - ratios (capex/rev, op margin) as percentages already (e.g., -37.83
        for −37.83%) → kind "pct" or "ratio"
      - PE multiples as numbers (e.g., 24.62) → kind "ratio"
      - net_debt / ttm_ebitda as RAW DOLLARS (e.g., 5_888_685_000 for $5.89B)
        → kind "billions" or "millions" requires unit conversion before
        comparison

    Tolerance is 5% relative for non-degenerate cases.
    """
    # Convert actual to the same unit as claimed when claimed is dollar-scaled
    if kind == "billions":
        actual = actual_raw / 1_000_000_000
        if abs(actual) < 0.01:
            return abs(claimed - actual) < 0.5
        return abs(claimed - actual) / abs(actual) <= tolerance
    if kind == "millions":
        actual = actual_raw / 1_000_000
        if abs(actual) < 1:
            return abs(claimed - actual) < 50
        return abs(claimed - actual) / abs(actual) <= tolerance
    # Ratios and percentages: peer_ratios.json stores them in the same unit
    # as the claimed value (e.g., 38.5 for 38.5%, 12.33 for 12.33x).
    if kind in ("ratio", "pct"):
        if actual_raw == 0:
            return abs(claimed) < 0.5
        return abs(claimed - actual_raw) / abs(actual_raw) <= tolerance
    return False


def _normalise_metric(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_peer_metric_claims(
    text: str,
    peer_tickers: set[str],
) -> list[tuple[str, str, str, int, str]]:
    """Find `<METRIC> <value>` patterns and bind to nearest peer ticker.

    Returns list of (ticker, metric_raw, value_raw, line_no, match_text).

    v2 (RC for AAOI 2026-05-08): the prior v1 regex required ticker
    immediately adjacent to the metric (`[\\s,—–-]+` bridge), which
    missed PM-style prose like

        "**FN (Fabrinet)** — closest comp on optical-transceiver mix
         and hyperscaler exposure; per `raw/peers.json` Fwd P/E = 36.6x,
         TTM operating margin ≈ 11.4%, ND/EBITDA ≈ −2.1x"

    where 50+ chars of descriptive prose separate the ticker from the
    metric, and the metric uses `=` instead of whitespace. v2 reverses
    the lookup: scan for known metric phrases followed by a value, then
    look back to the nearest peer ticker within the same paragraph
    (bounded by `\\n\\n`). Falls back to the prior tight pattern only
    when v2 finds no candidates — keeps backwards compat for tickers
    that are part of the metric label (e.g. "RIOT op margin").

    Patterns covered:
      - "RIOT capex/revenue 78.7%" (immediate-adjacent, v1 form)
      - "FN — ... per peers.json Fwd P/E = 36.6x" (long bridge + `=`)
      - "CIFR EV/EBITDA ~9×" (non-verifiable metric, prose-form value)
    """
    if not text or not peer_tickers:
        return []

    results: list[tuple[str, str, str, int, str]] = []
    tickers_alt = "|".join(re.escape(t) for t in sorted(peer_tickers, key=len, reverse=True))
    ticker_re = re.compile(rf"\b(?P<t>{tickers_alt})\b")

    # Build metric alternation from known phrases (verifiable + non-
    # verifiable). Sort by length descending so longer matches win
    # (e.g. "net debt/ebitda" before "net debt").
    all_phrases = list(_VERIFIABLE_METRICS.keys()) + list(_NON_VERIFIABLE_METRICS)
    metric_alt = "|".join(re.escape(p) for p in sorted(all_phrases, key=len, reverse=True))

    # Metric-value: known metric phrase, optional separator (`=`/`:`/`~`/`≈`),
    # then a value. Handles `−` (en-dash minus, U+2212) in addition to `-`.
    metric_value_re = re.compile(
        rf"\b(?P<metric>{metric_alt})\b"
        rf"\s*[=:≈~]?\s*"
        rf"(?P<value>(?:approximately\s+)?[~≈]?[-−]?\$?[\d.,]+\s*[%x×BM]?"
        rf"|low\s+single-digit)",
        re.IGNORECASE,
    )

    seen_offsets: set[int] = set()
    for mv in metric_value_re.finditer(text):
        if mv.start() in seen_offsets:
            continue
        seen_offsets.add(mv.start())

        # Look back up to 300 chars (within same paragraph) for nearest ticker
        lookback_start = max(0, mv.start() - 300)
        lookback_text = text[lookback_start:mv.start()]
        # Cap at last paragraph break so we don't bind across `\n\n` boundaries
        para_break = lookback_text.rfind("\n\n")
        if para_break != -1:
            lookback_text = lookback_text[para_break + 2:]

        ticker_matches = list(ticker_re.finditer(lookback_text))
        if not ticker_matches:
            continue
        nearest = ticker_matches[-1].group("t").upper()

        line_no = _line_no(text, mv.start())
        line_start = text.rfind("\n", 0, mv.start()) + 1
        line_end = text.find("\n", mv.end())
        if line_end == -1:
            line_end = len(text)
        match_text = text[line_start:line_end].strip()
        results.append((nearest, mv.group("metric"), mv.group("value"), line_no, match_text))

    return results


def validate_peer_metrics(
    text: str,
    file_label: str,
    peer_ratios_path: Path,
    peers_path: Path,
) -> list[PeerMetricViolation]:
    """Scan markdown text for peer-metric claims and verify against
    peer_ratios.json. Returns structured violations."""
    peer_tickers = _load_peer_tickers(peers_path)
    if not peer_tickers:
        return []  # no peer data; can't validate
    peer_ratios = _load_peer_ratios(peer_ratios_path)

    violations: list[PeerMetricViolation] = []
    seen_keys: set[tuple[str, str, int]] = set()  # dedupe overlapping matches

    for ticker, metric_raw, value_raw, line_no, match_text in (
        extract_peer_metric_claims(text, peer_tickers)
    ):
        metric_norm = _normalise_metric(metric_raw)
        key = (ticker, metric_norm, line_no)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Skip if the metric name doesn't roughly match a known schema —
        # avoids flagging benign prose like "RIOT spent capex ~12% of revenue"
        # when the metric phrasing is too vague to bind reliably.
        verifiable_field = _VERIFIABLE_METRICS.get(metric_norm)
        is_non_verifiable = any(
            nv == metric_norm or nv in metric_norm
            for nv in _NON_VERIFIABLE_METRICS
        )

        # Case 1: metric IS in peer_ratios.json — verify the value
        if verifiable_field:
            ticker_data = peer_ratios.get(ticker, {})
            if not isinstance(ticker_data, dict) or ticker_data.get("unavailable"):
                continue  # peer marked unavailable; can't verify
            actual = ticker_data.get(verifiable_field)
            if actual is None:
                continue  # field not populated
            claimed_val, kind = _parse_value(value_raw)
            if claimed_val is None or kind == "raw":
                continue  # value didn't parse; skip
            if not _values_match(claimed_val, float(actual), kind):
                violations.append(PeerMetricViolation(
                    severity="MATERIAL",
                    type="wrong_peer_metric",
                    file=file_label,
                    line_no=line_no,
                    ticker=ticker,
                    metric=metric_raw,
                    claimed_value=value_raw,
                    actual_value=str(actual),
                    match_text=match_text,
                ))
            continue

        # Case 2: metric is NOT verifiable AND the prose attributes it to
        # raw/peers.json or raw/peer_ratios.json — fabricated source.
        if is_non_verifiable:
            # Look around match_text for explicit source attribution
            ctx_window = match_text + text[
                max(0, _char_offset_of_line(text, line_no)):
                _char_offset_of_line(text, line_no + 3)
            ]
            attribution_present = any(
                kw in ctx_window.lower()
                for kw in ("raw/peers.json", "raw/peer_ratios.json",
                           "per peers.json", "per peer_ratios", "peer_ratios.json")
            )
            if attribution_present:
                violations.append(PeerMetricViolation(
                    severity="MATERIAL",
                    type="fabricated_metric_attribution",
                    file=file_label,
                    line_no=line_no,
                    ticker=ticker,
                    metric=metric_raw,
                    claimed_value=value_raw,
                    actual_value=None,
                    match_text=match_text,
                ))

    return violations


def _char_offset_of_line(text: str, line_no: int) -> int:
    """1-indexed line → character offset; clamps at end of text."""
    if line_no <= 1:
        return 0
    lines = text.split("\n")
    if line_no - 1 >= len(lines):
        return len(text)
    return sum(len(l) + 1 for l in lines[: line_no - 1])


def render_peer_violations_text(violations: list[PeerMetricViolation]) -> str:
    if not violations:
        return "PEER VALIDATION PASS: 0 violations"
    lines = [f"PEER VALIDATION FAIL: {len(violations)} violation(s)"]
    for v in violations:
        loc = f"{v.file or '?'}:{v.line_no}"
        lines.append(f"  [{v.severity}] {loc}  {v.type}  ({v.ticker})")
        if v.type == "wrong_peer_metric":
            lines.append(
                f"    {v.metric}: claimed {v.claimed_value} → "
                f"actual {v.actual_value} (peer_ratios.json)"
            )
        elif v.type == "fabricated_metric_attribution":
            lines.append(
                f"    {v.metric} ({v.claimed_value}) attributed to peer_ratios.json "
                f"but that metric is NOT a peer_ratios.json column"
            )
        lines.append(f"    text: {v.match_text[:120]}")
    return "\n".join(lines)
