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

from tradingagents.validators._helpers import (
    claim_attributed_to_other_ticker as _claim_attributed_to_other_ticker,
    line_no as _line_no,
)

# ---------------------------------------------------------------------------
# Phase 10 — wk25 context guards (off-balance-sheet / peer / pro-forma /
# historical).  Applied in validate_net_debt_claims AFTER extraction so that
# the extractor is untouched and all existing delta-bridge / tail guards keep
# working.  The window used for these checks is ±80 chars around the claimed
# value string in the claim's match_text.  This is narrower than the whole
# paragraph so incidental mentions elsewhere in the line don't over-skip.
# ---------------------------------------------------------------------------

_GUARD_WINDOW = 80  # chars on each side of the value string

# Off-balance-sheet commitment — NOT a balance-sheet debt figure.
# Guard 1 is now positional: only suppresses the claim when the OBS phrase
# appears AT OR AFTER the value's position in the context window (i.e., the
# figure is described AS an OBS item, e.g. "$261B in off-balance-sheet …").
# When OBS precedes the value ("net debt of $50B includes $20B OBS leases"),
# the $50B is the subject position and must still be flagged.
# Implementation: instead of searching the full ±80-char window, we only
# search a forward-only slice (value position → value position + 60 chars).
_OBS_RE = re.compile(r"off[- ]balance[- ]sheet", re.IGNORECASE)

# Forward / pro-forma estimates — NOT the current net-debt position.
# Narrowed per adversarial review:
#   - REMOVE bare 'forward' (ubiquitous analyst prose like "Looking forward…")
#   - REMOVE bare 'estimate[sd]?'; keep only compound 'forward estimate' /
#     'pro[- ]forma estimate'
#   - REMOVE bare 'rises? to'; keep only conditional 'would rise'
#   - KEEP 'pro[- ]forma', 'post-close', 'post-acquisition' (TXN FP)
_PROFORMA_RE = re.compile(
    r"\b(?:pro[- ]forma(?:\s+estimate)?|forward\s+estimate|post[- ]close"
    r"|post[- ]acquisition|would\s+rise)\b",
    re.IGNORECASE,
)

# Historical / pre-event FROM side of "from $X to $Y".
# We require the word "from" to appear IMMEDIATELY before the dollar figure
# in the context window so we only skip the FROM side, not the TO side.
_HIST_FROM_TO_RE = re.compile(
    r"\bfrom\s+\$[\d,]+(?:\.\d+)?\s*[BM]?\s+to\b",
    re.IGNORECASE,
)
# Other strong historical qualifiers in the window.
# Narrowed per adversarial review:
#   - DROP 'was\s+\$' entirely (suppresses "Net debt was $80B per yfinance"
#     — current reporting phrasing; the GOOGL FP doesn't need it because
#     _HIST_FROM_TO_RE already catches "from $X to" patterns)
#   - Restrict bare 'before\s+' to event-nouns only (a specific list of
#     transaction/corporate-event words); "before interest", "before taxes"
#     are NOT event qualifiers
#   - KEEP 'prior to', 'pre-<event>' (unchanged)
_HIST_OTHER_RE = re.compile(
    r"\b(?:before\s+(?:the\s+)?(?:wiz|deal|acquisition|merger|transaction"
    r"|close|closing|offer|ipo|spin[- ]off|divestiture|buyout)\b"
    r"|prior\s+to\b"
    r"|pre[-\s](?:deal|acquisition|merger|transaction|close|closing|wiz|offer))\b",
    re.IGNORECASE,
)


# wk29 whose-number guard: a dollar figure whose IMMEDIATE antecedent is a
# DIFFERENT financial metric (market cap / enterprise value) belongs to THAT
# metric, not net debt — even when "net debt" appears later in the same
# sentence via a bridge ("+ net debt", "and net debt of").  Real cases:
#   TXN 2026-07-17: "(market cap $258.53B + net debt $10.50B)"
#   MARA 2026-07-17: "market cap of only $4.45B and net debt of $1.90B"
# The [^$\d]{0,20} tail keeps it positional (metric label within ~20 chars of
# the value, no intervening dollar/number) so the real net-debt value that
# follows ("net debt of $1.90B") is still extracted and validated.
_COMPETING_METRIC_PREFIX_RE = re.compile(
    r"(?:market\s+cap(?:italization)?|mkt\s+cap|market\s+value"
    r"|enterprise\s+value|\bEV\b)[^$\d]{0,20}$",
    re.IGNORECASE,
)

# wk29 financing-flow guard: "net debt-issuance proceeds" is a Q1 financing
# cash flow, not a net-debt position (GOOGL 2026-07-17: "$31.4B of net
# debt-issuance proceeds"). The bridge/tail delta guards miss it because
# 'issuance' sits AFTER the label ("net debt-issuance"), not in the bridge.
_FINANCING_FLOW_RE = re.compile(
    r"debt[-\s]issuance|issuance\s+proceeds|proceeds\s+from\s+(?:the\s+)?issuance"
    r"|debt[-\s]rais(?:e|ing)",
    re.IGNORECASE,
)


def _is_competing_metric_prefixed(match_text: str, value_raw: str) -> bool:
    """True when the value's immediate antecedent is market cap / EV, so the
    figure belongs to that metric, not net debt."""
    pos = match_text.find(value_raw)
    if pos <= 0:
        return False
    prefix = match_text[max(0, pos - 40): pos]
    return bool(_COMPETING_METRIC_PREFIX_RE.search(prefix))


# wk29 not-a-position descriptors, checked positionally around the value:
#  - TAIL: "$24B annual funding gap" (ORCL) — the value IS the funding gap.
#  - PREFIX: "contractual-obligations figure (...) of $13.21B" (ECHO) — the
#    value is a maturity/obligations total, not the net-debt position.
_FUNDING_GAP_TAIL_RE = re.compile(
    r"^[^$\d]{0,15}(?:annual\s+|)funding\s+(?:gap|shortfall|need)"
    r"|^[^$\d]{0,15}cash\s+shortfall",
    re.IGNORECASE,
)
# Descriptor ... "of" immediately before the value. Requires an UNAMBIGUOUS
# obligations-total phrase ("contractual obligations", "obligations figure/
# total/table/schedule") — NOT bare "maturing"/"maturity", which appears near
# legitimate net-debt positions ("$5B maturing in 2026 … net debt of $99B").
# The phrase need only appear in the prefix window; a parenthetical restatement
# ("figure (...) of $13,213,574 thousand ($13.21B)") separates it from the
# value by another $ figure, so an "of $"-anchored form would miss it.
_OBLIGATIONS_PHRASE_RE = re.compile(
    r"contractual[- ]obligation"
    r"|obligations?\s+(?:figure|total|table|schedule)",
    re.IGNORECASE,
)
# Protects a genuine net-debt position that happens to sit after an
# obligations phrase: when "net debt/cash" is the value's IMMEDIATE antecedent
# ("… net debt of $20B"), the value is a position and must still be validated.
_NET_POSITION_IMMEDIATE_RE = re.compile(
    r"net\s+(?:debt|cash)\b[^$]{0,18}$",
    re.IGNORECASE,
)


def _is_non_position_descriptor(match_text: str, value_raw: str) -> bool:
    """True when the value is a funding gap (tail) or a maturity/contractual-
    obligations total (prefix) — neither is a net-debt position."""
    pos = match_text.find(value_raw)
    if pos < 0:
        return False
    tail = match_text[pos + len(value_raw):]
    if _FUNDING_GAP_TAIL_RE.search(tail):
        return True
    prefix = match_text[max(0, pos - 160): pos]
    return bool(
        _OBLIGATIONS_PHRASE_RE.search(prefix)
        and not _NET_POSITION_IMMEDIATE_RE.search(prefix)
    )


def _context_window(match_text: str, value_raw: str) -> str:
    """Return the ±GUARD_WINDOW-char window around value_raw in match_text.

    If value_raw appears multiple times (rare), uses the first occurrence.
    Falls back to the whole match_text if not found.
    """
    pos = match_text.find(value_raw)
    if pos == -1:
        return match_text  # fallback: whole line
    start = max(0, pos - _GUARD_WINDOW)
    end = min(len(match_text), pos + len(value_raw) + _GUARD_WINDOW)
    return match_text[start:end]


_DOLLAR_RE = re.compile(r"(?<![A-Za-z])\$[\d,]+(?:\.\d+)?\s*[BM]?")


def _is_off_balance_sheet_context(match_text: str, value_raw: str) -> bool:
    """True when the OBS phrase directly follows value_raw with no intervening
    dollar figure.

    The figure is characterized AS an off-balance-sheet item (e.g. ORCL
    '$261B in off-balance-sheet commitments') when OBS follows the value
    and there is no other dollar figure between the value and OBS in the text.

    When a different dollar figure intervenes ('net debt of $50B includes $20B
    in off-balance-sheet leases'), the OBS phrase qualifies $20B (the intervening
    figure), not $50B; $50B is the subject and must still be validated.

    Implementation: search a forward-only slice of 60 chars after value_raw.
    If OBS is found, additionally check that no other dollar figure appears
    between value_raw's end and the OBS match's start.
    """
    pos = match_text.find(value_raw)
    if pos == -1:
        # Fallback: use full text — conservative (may over-skip in edge cases)
        return bool(_OBS_RE.search(match_text))
    value_end = pos + len(value_raw)
    forward_slice = match_text[value_end: value_end + 60]
    obs_match = _OBS_RE.search(forward_slice)
    if not obs_match:
        return False
    # Check for any intervening dollar figure between value_end and OBS start
    between = forward_slice[: obs_match.start()]
    if _DOLLAR_RE.search(between):
        # Another dollar figure sits between the value and OBS → OBS qualifies
        # that intervening figure, not the claimed value.
        return False
    return True


def _is_proforma_or_forward_context(window: str) -> bool:
    """True when the context window explicitly marks the figure as a
    forward, pro-forma, or post-close estimate — not a current position."""
    return bool(_PROFORMA_RE.search(window))


def _is_historical_from_side(match_text: str, value_raw: str) -> bool:
    """True when value_raw is the FROM side of 'from $X to $Y' in the text.

    We look for the 'from $X to' pattern where $X is value_raw (with some
    tolerance for whitespace/unit variations).  We also check for other
    explicit pre-event qualifiers in the ±80-char window.
    """
    window = _context_window(match_text, value_raw)
    if _HIST_OTHER_RE.search(window):
        return True
    # Check the "from $X to" form: find the pattern and confirm value_raw
    # appears as the X part.
    for m in _HIST_FROM_TO_RE.finditer(match_text):
        # The segment from m.start() should contain value_raw right after "from "
        segment = match_text[m.start(): m.start() + len(value_raw) + 20]
        if value_raw in segment:
            return True
    return False


def _load_all_peer_tickers(raw_dir: Path) -> set[str]:
    """Load peer tickers from raw/peers.json (any-length uppercase keys)
    and from raw/peer_ratios.json (same format used by peer_metric_validator).

    Returns a set of uppercase strings.  Missing / malformed files are
    silently ignored (the guard degrades to the existing 2-5 char regex path).
    """
    tickers: set[str] = set()
    for fname in ("peers.json", "peer_ratios.json"):
        p = raw_dir / fname
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for k in d:
            if isinstance(k, str) and k.upper() == k and len(k) >= 1:
                tickers.add(k)
    return tickers


# Short-prefix context window for peer detection: how many chars before the
# dollar figure we scan for a peer ticker symbol (e.g. "vs T (" is ~6 chars).
_PEER_PREFIX_WINDOW = 30


def _is_peer_attributed_with_full_ticker_list(
    match_text: str,
    value_raw: str,
    main_ticker: str | None,
    peer_tickers: set[str],
) -> bool:
    """True when a PEER ticker (including 1-letter ones) appears within
    _PEER_PREFIX_WINDOW chars BEFORE the dollar figure in match_text, AND that
    ticker is different from main_ticker and is in the known peer list.

    This supplements the existing _claim_attributed_to_other_ticker() which
    only recognises 2-5 letter tickers via the full paragraph scan.
    """
    if not peer_tickers:
        return False
    main_upper = (main_ticker or "").upper()
    pos = match_text.find(value_raw)
    if pos == -1:
        return False
    prefix = match_text[max(0, pos - _PEER_PREFIX_WINDOW): pos]
    # Look for any peer ticker (any length, including 1-letter) in the prefix.
    # We require a word boundary before the ticker and a non-letter delimiter
    # after (space, '(', ':', '\'s', etc.) to avoid matching substrings.
    for ticker in peer_tickers:
        if ticker == main_upper:
            continue
        # Build a pattern: \bTICKER(?:'s|\s|[(]:,)
        pat = re.compile(
            r"\b" + re.escape(ticker) + r"(?:'s|\s|[\(:\,]|$)",
            re.IGNORECASE,
        )
        if pat.search(prefix):
            return True
    return False


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
#
# Phase 9.2 (ORCL 2026-07-01 fix): `(?!\s+outlay)` after the label — the
# 8-K supplemental table "Net Cash Outlay for Capital Expenditures" is a
# capex-funding term, not a net-cash position; quoting it verbatim
# produced 5 spurious MATERIAL definitional_drift blockers ("$4,592 =
# Net Cash Outlay ...", "net cash outlay $47.7B", "$15.7B of net cash
# outlay"). Same class of guard as the "net cash from operations"
# lookahead in `_PATTERN_LABEL_FIRST`.
_PATTERN_VALUE_FIRST = re.compile(
    r"(?<![A-Za-z])\$(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>[BM])?"
    r"(?P<bridge>[^\n.;|/÷×]{0,30}?)"
    r"\s+(?P<label>net\s+(?:cash|debt))(?!\s+(?:\*\*)?out(?:lay|flow))",
    re.IGNORECASE,
)
_PATTERN_LABEL_FIRST = re.compile(
    # Phase 7.12 v3 (GOOGL 2026-05-26 fix): negative lookahead to exclude
    # "net cash from operations" / "net cash from operating activities" /
    # "net cash from ops" — those are OCF (operating cash flow), a totally
    # different metric than the balance-sheet "net cash position" the
    # validator is designed to check. False positive observed on GOOGL:
    # "Net cash from operations $45,790M" got matched as a net-cash claim
    # vs canonical $49.34B; the LLM meant OCF, not net-cash position.
    r"(?P<label>net\s+(?:cash|debt))"
    # Phase 9.2 (ORCL 2026-07-01): also exclude "net cash outlay" — the
    # 8-K's supplemental capex-funding term, not a net-cash position —
    # plus the sibling flow term "outflow" and word-level bold
    # ("net cash **outlay**"), per adversarial review.
    r"(?!\s+(?:from\s+(?:operations|operating|ops)|(?:\*\*)?out(?:lay|flow)))"
    # Tighter bridge (20 chars) to defend against pairings like
    # `"net cash" and stops; the data shows that $16.70B of lease
    # obligations` — that's 33 chars and pairs the wrong dollar
    # figure across a semicolon. 20 chars covers `of $X`, `position
    # of $X`, `: $X`, `at $X` legitimate forms.
    # `|` excluded for v1.3 markdown-table-cell defense.
    # `/` excluded for v1.6 ratio-operator defense (NVDA 2026-05-08:
    # `(-$51.52B net cash) / $133.2B TTM EBITDA` was paired as
    # `$133.2B net cash` because `/` was inside the bridge — the value
    # AFTER `/` is the ratio denominator, not a continuation of the
    # 'net cash' label).
    #
    # Phase 8.1 (AVGO 2026-05-07 fix): bridge captured (was non-capturing)
    # so we can check it for delta-indicator words (increase/decrease/
    # change/swing) post-match. "net debt *increase* of $2.92B" matches
    # the regex; the bridge capture lets us recognize $2.92B as a delta
    # amount, not a position magnitude.
    r"(?P<bridge>[^\n.;|/÷×]{0,20}?)"
    # Phase 7.12 v2 (MSFT 2026-05-21 fix): optionally consume an inline
    # subtraction prefix `$A − $B =` so the captured value is the
    # derivation RESULT, not the minuend. Same root cause as the
    # peer_metric inline-equation FP — the LLM writes
    #   "Net debt = $40,262M − $32,105M = $8,157M"
    # which is correct math, but without this prefix-eater the regex
    # binds `Net debt` to $40,262M (the minuend) and flags a spurious
    # definitional drift. Non-capturing; only fires when the
    # `$A − $B =` shape is present, so single-value forms are
    # unaffected. Handles both U+2212 (−) and ASCII `-` as the minus.
    r"(?:[-−]?\$[\d,]+(?:\.\d+)?\s*[BM]?\s*[−-]\s*"
    r"[-−]?\$[\d,]+(?:\.\d+)?\s*[BM]?\s*=\s*)?"
    # Phase 8.1: allow optional `**` markdown bold around the value, so
    # the inline-subtraction prefix-eater above can correctly hand off
    # to `**$7,975M**` (the bolded result in ON's net-debt math). Same
    # fix as the peer_metric regex.
    r"(?<![A-Za-z])(?:\*\*)?\$(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>[BM])?(?:\*\*)?",
    re.IGNORECASE,
)


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


_DELTA_COMPARATORS_RE = re.compile(
    # Comparator words/patterns that, when they immediately follow the $X
    # value, mean the value is a DELTA, period-over-period change, or a
    # range endpoint — NOT a magnitude claim about net debt itself.
    #
    # Phase 7.12 (META 2026-05-21 fix): "yields $5.59B net debt,
    #   approximately $30B lower." — `lower` after the value.
    # Phase 8.1 (AVGO 2026-05-07): extend with period-over-period
    #   markers (`sequentially`, `YoY`, `QoQ`, `year-over-year`).
    # Phase 8.1 (ORCL 2026-05-07): a `[\-–]\s*\d` tail means the value
    #   is the LOW end of a range like `$5–6B` — skip.
    r"^\s*(?:"
    r"lower|higher|less|more|below|above|different|apart"
    r"|shy(?:\s+of)?|short(?:\s+of)?|away(?:\s+from)?"
    r"|over|under"
    r"|sequentially|year[-\s]?over[-\s]?year|quarter[-\s]?over[-\s]?quarter"
    r"|YoY|QoQ"
    # Phase 9: debt-flow words trailing the label ("net debt raise/issuance")
    # mean the value is a financing cash flow, not a net-debt position.
    r"|raise|raised|issuance|issued|repaid|repayment|drawdown|drawn|borrowing"
    r")\b"
    r"|^[\-––]\s*\d",   # range endpoint: `$5–6B`, `$5-6B`
    re.IGNORECASE,
)

# Phase 8.1 (AVGO 2026-05-07 fix): when the BRIDGE between the label and
# the value contains a delta-indicator word (increase, decrease, change,
# swing, etc.), the value is the amount-of-change, not a position.
#
# Phase 8.2 (MSTR 2026-05-29 fix): MSTR re-run had three FPs that needed
# wider bridge guards:
#   - "$0.06B higher than yfinance Net Debt"  → "higher" in bridge
#   - "net debt + $10.0B preferred ..."        → "+" in bridge (additive
#     component, not the net-debt magnitude itself)
# Extend to catch positional comparators (higher/lower/above/below/more/
# less/over/under/plus) and a bare `+` separator. Conservative on common
# English words (e.g., "and", "with") — keep them OUT to avoid skipping
# legitimate claims like "net debt and EBITDA are both stable at $X".
_DELTA_BRIDGE_RE = re.compile(
    # Stems (no closing \b — match "increase/increased/increasing" etc.)
    # Phase 9 (GOOGL 2026-05-26 fix): debt-flow words (raise/raised, issuance/
    # issued, repaid/repayment, drawn) mean the value is a financing CASH FLOW
    # (e.g. "$29.9B net debt raise" = Q1 debt issuance), not a net-debt position.
    # wk29 (ORCL 2026-07-17): "net debt grew $16.47B in one year" — 'grew' is a
    # YoY-change verb; $16.47B is the delta, not the position. Add grow/grew/grown.
    r"\b(?:increas|decreas|chang|swing|delta|rose|risen|fell|fallen"
    r"|rais|issu|repaid|repay|repaym|drawn|drew|borrow|grew|grow|grown)"
    # Full words (require closing \b)
    r"|\b(?:higher|lower|above|below|more|less|over|under|plus)\b"
    # Bare additive operator (with whitespace either side)
    r"|\s\+\s",
    re.IGNORECASE,
)


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

            # Phase 7.12 delta-phrase guard (tail check): if the value is
            # followed by a comparator word (lower/higher/sequentially/YoY)
            # or a range endpoint marker (`-6B`), it's not a magnitude.
            tail = text[m.end():m.end() + 20]
            if _DELTA_COMPARATORS_RE.match(tail):
                continue

            # Phase 8.1 delta-bridge guard: if the captured bridge contains
            # `increase`/`decrease`/`change`/`swing`/etc., the value is a
            # delta amount (e.g. "net debt *increase* of $2.92B"), not the
            # net-debt position itself. Both LABEL_FIRST and VALUE_FIRST
            # capture the bridge; missing group → empty string (no match).
            bridge = m.groupdict().get("bridge", "") or ""
            if _DELTA_BRIDGE_RE.search(bridge):
                continue

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

    # Component cells: a value matching a raw balance-sheet component (Cash,
    # STI, LTD, CD, capital leases) is a TRACEABLE figure cited inside a
    # reconciliation near a "net debt"/"net cash" label — not a fabricated
    # net-position claim. Accepting them kills the recurring false positives
    # where the validator paired a component cell with a nearby label (INTC
    # 2026-05-29: Cash $17.25B / STI $15.54B beside "Net Debt of $27.78B").
    sti = (cash_sti - cash) if (cash_sti and cash and cash_sti != cash) else 0.0
    for lbl, val in (("Cash component", cash), ("STI component", sti),
                     ("Long-Term Debt component", ltd), ("Current Debt component", cd),
                     ("Capital-lease component", cl)):
        if val:
            candidates.append((lbl, abs(val)))

    return candidates


def _within_tolerance(claimed: float, canonical: float) -> bool:
    """5% relative OR $0.5B absolute — whichever is larger."""
    rel = 0.05 * max(abs(claimed), abs(canonical))
    abs_tol = 5e8  # $0.5B
    tolerance = max(rel, abs_tol)
    return abs(claimed - canonical) <= tolerance


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

    # Phase 10: load peer tickers from raw/ directory (siblings to net_debt.json)
    # so the 1-letter ticker guard can use the actual peer list (e.g. "T" for AT&T).
    raw_dir = net_debt_json_path.parent
    _peer_tickers = _load_all_peer_tickers(raw_dir)

    violations: list[NetDebtViolation] = []
    for claim in claims:
        # Skip claims attributed to peer tickers (handled by Phase 7.3).
        # This covers 2-5 letter tickers via the existing paragraph-scan path.
        if _claim_attributed_to_other_ticker(claim.match_text, main_ticker):
            continue

        # Phase 10 — context guards: skip claims that are explicitly marked as
        # NOT the subject's current net debt.  We use a ±80-char window around
        # the claimed dollar figure so incidental mentions elsewhere in the same
        # paragraph don't over-skip.  When uncertain, we fall through and flag.
        window = _context_window(claim.match_text, claim.value_raw)

        # Guard 1: off-balance-sheet commitment — not on the balance sheet.
        # Positional: OBS must appear AT OR AFTER the value in match_text.
        if _is_off_balance_sheet_context(claim.match_text, claim.value_raw):
            continue

        # Guard 2: forward / pro-forma estimate — not the current position.
        if _is_proforma_or_forward_context(window):
            continue

        # Guard 3: historical from-side — "from $X to $Y"; also catches
        # "before the …", "prior to", "was $X" qualifiers.
        if _is_historical_from_side(claim.match_text, claim.value_raw):
            continue

        # Guard 4: peer-attributed via 1-letter (or any-length) peer ticker
        # appearing immediately before the dollar figure in the match_text.
        # Supplements _claim_attributed_to_other_ticker which only covers
        # 2-5 letter tickers.
        if _is_peer_attributed_with_full_ticker_list(
            claim.match_text, claim.value_raw, main_ticker, _peer_tickers
        ):
            continue

        # Guard 5 (wk29 whose-number): the value's immediate antecedent is a
        # competing metric (market cap / enterprise value), so it is not net
        # debt even though "net debt" appears later in the sentence.
        if _is_competing_metric_prefixed(claim.match_text, claim.value_raw):
            continue

        # Guard 6 (wk29 financing-flow): "net debt-issuance proceeds" is a
        # financing cash flow, not a net-debt position.
        if _FINANCING_FLOW_RE.search(window):
            continue

        # Guard 7 (wk29 not-a-position): funding gap (tail) or a maturity /
        # contractual-obligations total (prefix) — neither is net debt.
        if _is_non_position_descriptor(claim.match_text, claim.value_raw):
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
