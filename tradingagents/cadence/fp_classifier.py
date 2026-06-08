from __future__ import annotations

import re
from tradingagents.cadence.models import FlagDisposition, FlagVerdict, RunData

# Words that, when adjacent to the flagged dollar figure, prove it is NOT a
# net-debt/net-cash *balance* (so phase_7_5 mis-grabbed it).
_NON_NETDEBT_NEARBY = (
    "buyback", "repurchase", "authoriz",            # share-buyback authorization
    "operating activities", "operating cash flow",  # cash-flow statement line
    "fcf", "free cash flow",                        # free cash flow
    "capex", "capital expenditure",
)


def _phase_violations(validation: dict):
    for phase, block in validation.items():
        if isinstance(block, dict):
            for v in block.get("violations", []) or []:
                yield phase, v


def _dollars_to_tokens(d: float | None) -> list[str]:
    """Plausible textual renderings of a dollar figure for window matching."""
    if not d:
        return []
    toks = []
    billions = d / 1e9
    millions = d / 1e6
    toks.append(f"{billions:.2f}".rstrip("0").rstrip("."))   # 163 / 39.14
    toks.append(f"{int(round(millions)):,}")                  # 2,540
    return [t for t in toks if t]


def _is_netdebt_dollar_grab(v: dict) -> bool:
    mt = (v.get("match_text") or "").lower()
    for tok in _dollars_to_tokens(v.get("claimed_dollars")):
        idx = mt.find(tok.lower())
        if idx == -1:
            continue
        window = mt[max(0, idx - 40): idx + 40]
        if any(w in window for w in _NON_NETDEBT_NEARBY):
            return True
    return False


def _fmt_price(p) -> str | None:
    if p is None:
        return None
    s = f"{float(p):.2f}".rstrip("0").rstrip(".")
    return s


def _is_from_to_miswire(v: dict) -> bool:
    """True when match_text reads 'from $<actual_close> ... to $<claimed_price>',
    i.e. the validator paired the date with the 'to' endpoint, not the 'from'."""
    mt = (v.get("match_text") or "").lower()
    frm = _fmt_price(v.get("actual_close"))
    to = _fmt_price(v.get("claimed_price"))
    if not frm or not to:
        return False
    pat = r"from\b[^.]*?\$?" + re.escape(frm) + r"[^.]*?\bto\b[^.]*?\$?" + re.escape(to)
    return re.search(pat, mt) is not None


_METRIC_TO_KEY = {
    "nd/ebitda": "nd_ebitda", "net debt/ebitda": "nd_ebitda", "leverage": "nd_ebitda",
    "ttm pe": "ttm_pe", "ttm p/e": "ttm_pe", "forward pe": "forward_pe",
    "forward p/e": "forward_pe", "ttm ebitda": "ttm_ebitda", "net debt": "net_debt",
    "op margin": "op_margin", "operating margin": "op_margin",
    "capex/revenue": "capex_revenue",
}


def _parse_metric_value(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    return float(m.group(0).replace(",", "")) if m else None


def _is_respectively_mismap(v: dict, run: RunData) -> bool:
    if "respectively" not in (v.get("match_text") or "").lower():
        return False
    key = _METRIC_TO_KEY.get((v.get("metric") or "").strip().lower())
    claimed = _parse_metric_value(v.get("claimed_value"))
    blamed = (v.get("ticker") or "").upper()
    if not key or claimed is None:
        return False
    for peer, cells in (run.peer_ratios or {}).items():
        if peer.upper() == blamed or not isinstance(cells, dict):
            continue
        cell = cells.get(key)
        if isinstance(cell, (int, float)) and abs(cell - claimed) <= 0.01:
            return True   # claimed value belongs to a DIFFERENT peer -> mis-map
    return False


def classify_violation(phase: str, v: dict, run: RunData) -> FlagVerdict:
    vtype = v.get("type")
    detail = {k: v.get(k) for k in ("severity", "file", "line_no", "claimed_value",
                                    "claimed_dollars", "match_text", "ticker",
                                    "metric", "actual_close", "claimed_price")
              if v.get(k) is not None}

    if vtype == "skipped_non_usd_reporter":
        return FlagVerdict(phase, FlagDisposition.CORRECT_BY_DESIGN,
                           "non-USD reporter: net-debt check correctly skipped", detail)

    if vtype == "definitional_drift" and _is_netdebt_dollar_grab(v):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "flagged $ is a non-net-debt quantity (buyback / operating "
                           "cash flow / FCF) sitting near 'net debt/cash' wording", detail)

    if vtype == "wrong_close" and _is_from_to_miswire(v):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "price-date 'from $A (date) to $B' cross-wire: validator "
                           "paired the date with the 'to' endpoint, not the 'from' value",
                           detail)

    if vtype == "wrong_peer_metric" and _is_respectively_mismap(v, run):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "peer 'X and Y respectively': claimed value matches the OTHER "
                           "peer's cell; validator mapped the metric to the wrong ticker",
                           detail)

    return FlagVerdict(phase, FlagDisposition.NEEDS_ADJUDICATION,
                       f"{vtype or 'unknown'}: no known false-positive pattern matched", detail)


def classify_run_flags(run: RunData) -> list[FlagVerdict]:
    return [classify_violation(phase, v, run)
            for phase, v in _phase_violations(run.validation)]
