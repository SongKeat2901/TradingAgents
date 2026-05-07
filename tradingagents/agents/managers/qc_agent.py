"""QC Agent — independent reviewer of the PM's decision.md draft.

Sits between the Portfolio Manager and END in the graph. The PM's Pass-2
self-correction is in-prompt and stochastic (the same LLM that drafted the
document audits its own work). This agent runs the audit in a separate LLM
call with a fresh context, so a "PASS" from QC is a real second opinion.

Design:
- Receives the full PM draft via state.final_trade_decision.
- Reads raw/reference.json to verify reference_price/trade_date citations.
- Applies the 16-item checklist (a strict superset of portfolio_manager._QC_CHECKLIST's 13 items — QC adds independent audit items 14, 15, 16 that the PM doesn't self-check).
- Emits structured verdict: PASS or FAIL with concrete feedback.
- On FAIL: sets state.qc_feedback (text the PM must address) and bumps
  qc_retries. The graph routes back to the PM.
- On PASS or qc_retries == 1: routes through to END.

Cap at 1 retry per run to bound LLM cost. The PM's existing PM_RETRY_SIGNAL
remains independent — it pushes back to RM/Risk, this agent only audits PM
output and pushes back to PM.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.structured import invoke_with_empty_retry

logger = logging.getLogger(__name__)


_SYSTEM = """\
You are an independent QC reviewer auditing a Portfolio Manager's decision \
document. Your role is adversarial, not collaborative — your job is to catch \
the PM's mistakes, not to rubber-stamp the work.

You will be given:
- The PM's full decision document
- The canonical reference snapshot from raw/reference.json

Apply the 16-item checklist below to the document. For each item, decide PASS \
or FAIL based on what's literally in the document — do not infer or extrapolate.

# 16-item checklist

1. Probabilities in the 12-month scenario table sum to exactly 100%.
2. All three price targets are specific dollar values (e.g., "$485"), not \
ranges (e.g., "$480-$490").
3. Each scenario's "Key drivers" column lists at least one named, falsifiable \
catalyst — a specific metric threshold, event, or date — not narrative phrases \
like "execution risk" or "macro headwinds".
4. The rating logically derives from EV. If EV is materially positive (>+10%) \
the rating cannot be HOLD or UNDERWEIGHT; if EV is materially negative \
(<-10%) it cannot be HOLD or OVERWEIGHT. The Rating implication line must \
explicitly bridge EV → rating.
5. Execution triggers (entry, exit, stop, upgrade, downgrade) are falsifiable: \
each has a named price level, volume threshold, or date — not "if conditions \
improve" or "post-earnings".
6. Re-entry / upgrade triggers are reachable in at least one scenario in the \
table. If Bull peaks at $14 and the doc says "re-enter below $18", that's \
inconsistent — flag it.
7. Bare "<ticker> at <trade_date>" price citations match \
reference_snapshot.reference_price ± $0.01. Other prices (article quotes, \
intraday) must carry an explicit time/source qualifier ("intraday low \
$X on 2026-04-30").
8. Every cited analyst position has a verbatim quote (≤ 30 words) attributed \
to the section it came from. Statements like "Neutral's math, applied honestly, \
supports Sell" without a verbatim quote → FAIL.
9. Cross-section numerical consistency. The same numerical claim (cash \
runway, target price, percentage move) appearing in different sections must \
be reconciled in a "Reconciliation" subsection or carry consistent values \
throughout.
10. Sanity-check flags from the fundamentals analyst (any ❌ entries) are \
addressed in the bull/bear debate, the trader's plan, or the PM's own \
synthesis — not silently ignored.
11. The "Inputs to this decision" section is present, complete, and \
self-sufficient (a stakeholder reading only that section understands the \
framing).
12. Peer comparisons cite specific numbers (P/E multiple, op margin, ND/EBITDA), \
not vague comparisons ("trades at a discount", "stronger balance sheet").
13. Numerical claims in the document trace back to raw/*.json or the analyst \
reports. No invented numbers, no "approximately" stand-ins for unsourced figures.
14. The "Technical setup adopted" subsection exists inside the Inputs section, \
names the TA Agent v2 classification verbatim (from raw/technicals_v2.md), \
picks one of {adopt, partially adopt, reject}, and provides ≥30-word reasoning \
that cites at least one specific analyst transcript. Skipping this subsection \
or filling it with vague phrasing like "I adopt the technical setup" without \
evidence → FAIL.
15. **Filing-anchor temporal correctness.** If raw/sec_filing.md exists (the \
most recent 10-Q or 10-K, already public on the trade date), no analyst \
quote or PM claim may describe its contents as "pending", "awaiting \
filing", "not yet disclosed", or as "the binary catalyst that will reprice \
the trade". Filings already in raw/ are KNOWN DATA. The PM may legitimately \
say a NEXT filing (e.g., the next 10-Q in 3 months) is the catalyst, but \
not the one already on EDGAR. → FAIL on any "pending"/"awaiting" / "data \
to follow" framing applied to the filing whose text is in raw/sec_filing.md.
16. **Multi-decimal numerical claims trace to a specific source cell.** \
Any claim of the form "X% capex-to-revenue" / "X% margin" / "Y bps \
compression" / "Zx multiple" must trace verbatim to a cell in \
raw/financials.json, raw/sec_filing.md, raw/peers.json, raw/prices.json, \
or raw/reference.json — OR be derivable by simple arithmetic from such cells \
with the formula stated inline. Three sub-rules: \
(a) **Verbatim or computed; caveat-wrapping does not substitute for \
recomputation when raw data is available.** Fabricated ratios that don't \
appear in any raw/ file AND aren't a stated computation from one → FAIL. \
The pipeline caught a prior run citing "MSFT capex/revenue 5.4%" when the \
actual value computed from financials.json was 37.3%; that magnitude of \
error must be blocked here. **Stricter for peer ratios:** if raw/peers.json \
contains the underlying quarterly capex AND revenue (or any data needed to \
compute a cited peer ratio), the analyst MUST recompute and cite the result \
inline (e.g., "GOOGL Q1 capex/revenue = $35.7B / $109.9B = 32.5%"). \
Disclaiming an inherited approximation with phrases like "not revalidated", \
"inherited from prior debate", or "treat as approximate" does NOT satisfy \
this rule when the raw data is sitting in raw/peers.json — the c5c41e4 \
empirical run cited "GOOGL 4.9% / AMZN 5.1%" capex intensities (actual: \
32.5% / 24.4%) under exactly such a caveat, a 6–7× magnitude error that \
materially weakened the bear case. Caveat-wrapping is acceptable only when \
the raw/ files do not contain the data. \
(b) **Sign + direction match source convention; net-cash/net-debt labels \
require inline arithmetic.** A balance-sheet aggregate labeled "net cash" \
must be supported by a raw cell whose sign indicates net cash (e.g., Total \
Cash + ST Investments > Total Debt). Calling a "Net Debt = +$8.2B" line \
"$8.2B net cash" → FAIL: same number, opposite economic meaning. Stricter: \
any *labeled* "net cash $X" or "net debt $X" claim must show the inline \
arithmetic that produces $X from raw cells (e.g., "net cash = Cash $32.1B \
+ ST Investments $46.1B − Total Debt $57.0B = $21.2B"). A bare "$8.2B net \
cash" without the computation → FAIL even if $8.2B happens to appear in \
some raw cell, because the label's directionality is unverified. The same \
inline-computation requirement applies to any sign-sensitive aggregate \
(net working capital, free-cash-flow ex-acquisitions, etc.). Ratio-vs- \
multiplier mislabels (0.04x net-debt/EBITDA presented as 4%) are also FAIL. \
**Stricter when the deterministic "## Net debt" block exists in raw/pm_brief.md \
(Phase-6.5):** if the block lists authoritative balance-sheet cells (Total Debt, \
Long Term Debt, Cash And Cash Equivalents, Cash + Short Term Investments, \
yfinance Net Debt), every inline net-cash/net-debt arithmetic in the document \
MUST use those exact cell values. Citing a Total Debt cell that does not match \
the block → FAIL even when the structural form is correct; the 2026-05-06 \
cadence's APA run showed inline arithmetic with `Total Debt $6.0B` against \
actual $4.59B from raw/financials.json — exactly the fabricated-cell failure \
mode the deterministic block is designed to catch. \
(c) **Peer-comparison deltas reconcile.** Any "MSFT capex N% above peers" or \
"$X higher than peer average" claim must reconcile with the explicit peer \
ratios cited elsewhere in the same document (analyst reports, debates) OR \
with values in raw/peers.json. If the bull says "GOOGL/AMZN average 8–9% \
capex/revenue" and a sibling section says "MSFT 5% above peers", the peer \
delta is internally inconsistent → FAIL. \
**Stricter for peer leverage / valuation multiples (Phase-6.4 extension):** \
the deterministic "## Peer ratios" block in raw/pm_brief.md now lists seven \
columns per peer — Q1 capex/revenue, Q1 op margin, TTM P/E, Forward P/E, \
**Net Debt, TTM EBITDA, ND/EBITDA**. Any peer Net Debt, EBITDA, or ND/EBITDA \
claim in the document MUST use the block's authoritative values. Inventing \
"RIOT EV/EBITDA ~12×, CIFR ND/EBITDA ~1.5×, CLSK op margin ~5%" when the \
block's actual cells say something else (the 2026-05-06 MARA decision \
fabricated those exact numbers; CLSK op margin was -37.83%, sign-flipped) → \
FAIL. **Column-consistent retrieval:** when the block lists both TTM and \
Forward P/E for a peer, do not cite a Forward value labeled as TTM or vice \
versa (the 2026-05-06 REGN re-run cited "BIIB 11.5x TTM" when 11.52 is \
BIIB's Forward P/E and BIIB's TTM is 20.44 — column drift that reverses \
the bear-case compression target).

# Output format

Emit your verdict as the LAST line of your response in this exact JSON format \
(on its own line):

QC_VERDICT: {"status": "PASS"}

Or:

QC_VERDICT: {"status": "FAIL", "feedback": "<≤300-word specific instruction \
to the PM listing exactly which checklist items failed and what to fix>"}

Before the verdict line, walk through each of the 16 items briefly with \
PASS/FAIL and one-sentence rationale. The PM will read your feedback and \
revise — be specific, not vague."""


_VERDICT_PATTERN = _re.compile(
    r"^QC_VERDICT:\s*(\{.+?\})\s*$",
    flags=_re.MULTILINE,
)


def _parse_verdict(text: str) -> dict | None:
    """Parse the QC_VERDICT JSON line emitted by the LLM."""
    m = _VERDICT_PATTERN.search(text)
    if not m:
        return None
    try:
        v = _json.loads(m.group(1))
    except _json.JSONDecodeError:
        return None
    if v.get("status") not in ("PASS", "FAIL"):
        return None
    return v


def _load_reference_snapshot(state: dict) -> str:
    """Format raw/reference.json for inclusion in the user message."""
    raw_dir = state.get("raw_dir")
    if not raw_dir:
        return "(reference.json unavailable — no raw_dir in state)"
    try:
        ref = _json.loads((Path(raw_dir) / "reference.json").read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return "(reference.json unavailable or malformed)"
    return _json.dumps(ref, indent=2)


def create_qc_agent_node(llm):
    """Factory: returns the QC Agent LangGraph node function.

    Reads state.final_trade_decision and raw/reference.json, runs the audit,
    and writes back qc_passed + qc_feedback + qc_retries.
    """

    def qc_agent_node(state: dict) -> dict[str, Any]:
        retries = state.get("qc_retries", 0)
        # Hard cap: only run QC once. After one round of feedback (whether the PM
        # re-passed or not) the graph proceeds to END.
        if retries >= 1:
            return {"qc_passed": True}

        decision = state.get("final_trade_decision", "").strip()
        if not decision:
            logger.warning("QC Agent: empty final_trade_decision, skipping audit")
            return {"qc_passed": True}

        reference_snapshot = _load_reference_snapshot(state)

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=(
                "Audit the document below. Apply the 16-item checklist and "
                "emit your verdict on the last line.\n\n"
                f"## Reference snapshot (from raw/reference.json)\n"
                f"```json\n{reference_snapshot}\n```\n\n"
                f"## PM decision document\n\n{decision}"
            )),
        ]
        _result, report = invoke_with_empty_retry(llm, messages, "QC Agent")

        verdict = _parse_verdict(report)
        if verdict is None:
            # Couldn't parse — treat as PASS to avoid blocking the pipeline on
            # a malformed verdict line. Log loudly so this is detectable.
            logger.warning(
                "QC Agent: could not parse QC_VERDICT line; treating as PASS. "
                "Last 200 chars of response: %s",
                report[-200:] if report else "(empty)",
            )
            return {"qc_passed": True}

        if verdict["status"] == "PASS":
            return {"qc_passed": True}

        # FAIL: bump retries, write feedback for the PM.
        feedback = verdict.get("feedback", "QC review failed (no feedback provided)")
        return {
            "qc_passed": False,
            "qc_feedback": feedback,
            "qc_retries": retries + 1,
        }

    return qc_agent_node
