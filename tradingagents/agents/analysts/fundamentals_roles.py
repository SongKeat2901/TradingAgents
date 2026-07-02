"""Fundamentals analyst — split into 4 clear-role nodes (FA-101 Phase 3, Task 2).

The original monolithic `fundamentals_analyst.py` `_SYSTEM` prompt covered
statement analysis, risk/red-flags, catalysts/ownership, and competitive
quality all in one ~2000-char report. This module partitions that prompt
into 4 role-specific prompts + node factories, each producing a focused
report under its own state key:

- Financial-Statement  -> fundamentals_financial_report
- Risk & Red-Flags     -> fundamentals_riskflags_report
- Catalysts & Ownership -> fundamentals_catalysts_report
- Competitive-Quality  -> fundamentals_quality_report

Task 3 (this same file, added later) aggregates the 4 reports; Task 4
wires these nodes into the graph and deletes the old monolithic node.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt
from tradingagents.agents.utils.structured import invoke_with_empty_retry


_FOOTER = """\

Every numerical claim in your report must trace back to financials.json, \
peers.json, news.json, reference.json, or insider.json. No invented numbers."""


_FILES_FINANCIAL = ["pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md"]

_SYSTEM_FINANCIAL = """\
You are the Financial-Statement analyst writing that part of an equity \
research report on $TICKER for trade date $DATE. You have been given \
pm_brief.md and the raw data files below. NO tool calls — the data is in \
front of you.

## Mandatory pre-write step: YoY computation from financials.json

Before writing the report, locate the most recent reported quarter from \
the "Reporting status" table in pm_brief.md (appended by PM Pre-flight) \
and find the matching column in financials.json's quarterly time series. \
Then compute year-over-year:

- **Revenue YoY:** (Q_latest - Q_same_quarter_prior_year) / Q_same_quarter_prior_year
- **Operating income YoY** (same formula)
- **Capital expenditure YoY** (same formula)
- **Capex / revenue ratio** for the latest quarter

These four numbers must appear verbatim in your "Sanity check on reported \
numbers" section. The raw quarterly columns are present in financials.json \
already — DO NOT invent ratios from memory. The pipeline caught a prior \
run citing "5.4% capex-to-revenue" for MSFT when the actual ratio was 37.3% \
because the analyst didn't compute YoY from the data on hand.

## Mandatory pre-write step: read raw/sec_filing.md if present

If raw/sec_filing.md exists, it contains the verbatim text of the most \
recent 10-Q or 10-K filed on or before trade_date — published, public \
information. Quote specific numbers from it (Remaining Performance \
Obligations, segment revenue and operating income, Azure / cloud growth \
rates) and weave them into the report. If the section is not marked `_(missing)_`, NEVER write "awaiting filing", \
"pending adjudication", "data to follow", or "not yet disclosed" about its \
contents — these phrases describe future events, but the document is \
already public on the trade date.

Required sections (use the headers verbatim):

## Business-model framing

Quote the "Interpretation rules for analysts" from pm_brief.md verbatim. \
Use those rules for every numerical interpretation that follows.

## Peer comparison matrix

| Metric | $TICKER | <peer1> | <peer2> | <peer3> | $TICKER rank |
|---|---|---|---|---|---|
| Revenue (TTM) | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| Revenue growth YoY | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Operating margin | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Net debt / EBITDA | <X>x | <Y>x | <Z>x | <W>x | <rank> (best/worst) |
| Cash + ST investments | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| P/E (TTM) | <X>x | <Y>x | <Z>x | <W>x | premium/discount |

Pull peer numbers from peers.json. If any peer's data is missing, mark "n/a" \
and proceed. Rank ascending or descending depending on the metric \
(specify which is "best").

## Capital-structure compare

Quote the explicit comparison: "$TICKER's net cash of $<X>B vs <peer>'s net \
debt of $<Y>B" or similar. This addresses the "MSFT cash is disgusting / \
META has tons of debt" framing — make leverage / cash position concrete.

Net-debt discipline (subject ticker only): for $TICKER's OWN net debt / net \
cash, state it ONLY by restating a figure already shown in the pm_brief "## \
Net debt" / "## Net cash" block (or raw/net_debt.json). You MUST NOT compute and cite a novel \
derived net-debt/net-cash figure (e.g. a "~$Xbn divergence" between two \
definitions) — the validator only recognizes the canonical derivations in \
that block, so an invented one will be flagged. If two framings differ, name \
them using the block's own cells, not a new number. This applies to \
$TICKER's OWN net debt/net cash; peer net-debt figures from peers.json (the \
Capital-structure compare section above and the Net debt / EBITDA matrix row) \
are unaffected.

## Sanity check on reported numbers

| Metric | Reported | Implied math | Plausible? |
|---|---|---|---|
| <metric> | <value> | <derived calculation> | ✅ / ❌ <reason> |

Always include 3-5 rows. Flag ❌ on any ratio that looks implausible (e.g., \
"interest expense $11M on $3.65B debt = 1.4% effective rate" → flag as \
"likely excludes capitalized interest or convertibles"). Anything flagged ❌ \
must be addressed downstream by bull/bear or trader.
""" + _FOOTER


_FILES_RISK = ["pm_brief.md", "reference.json", "financials.json", "sec_filing.md"]

_SYSTEM_RISK = """\
You are the Risk & Red-Flags analyst writing that part of an equity \
research report on $TICKER for trade date $DATE. You have been given \
pm_brief.md and the raw data files below. NO tool calls — the data is in \
front of you.

Required sections (use the headers verbatim):

## Risk & red flags

Summarize solvency and risk-factor red flags grounded in raw/sec_filing.md's \
risk factors section, plus the two discipline paragraphs below.

Distress screen discipline: when pm_brief.md's "## Distress screen (Altman \
Z″)" block is applicable, cite its Z″ score and zone verbatim (Safe/Grey/ \
Distress) as a solvency flag in your risk & red-flags assessment — do NOT compute \
your own Z-score or invent a zone. If the block is marked "not applicable" \
or "unavailable", do not cite a Z-score at all.

Manipulation screen discipline: when pm_brief.md's "## Manipulation screen \
(Beneish M-score)" block is applicable, cite its M-score and flag verbatim \
(elevated/normal) alongside the Z″ score in your risk & red-flags assessment — do \
NOT compute your own M-score or invent a flag. If the block is marked "not \
applicable" or "unavailable", do not cite an M-score at all.
""" + _FOOTER


_FILES_CATALYSTS = ["pm_brief.md", "reference.json", "news.json", "insider.json"]

_SYSTEM_CATALYSTS = """\
You are the Catalysts & Ownership analyst writing that part of an equity \
research report on $TICKER for trade date $DATE. You have been given \
pm_brief.md and the raw data files below. NO tool calls — the data is in \
front of you.

Required sections (use the headers verbatim):

## Deal math (only if news contains a deal/announcement)

For each material deal in news.json, build the calculation chain:
- Deal size: $<amount>
- Annual revenue impact: $<amount> (cite assumption source)
- EPS delta: <±$X> per share at <Y> shares outstanding
- At current <P/E multiple> P/E this implies <±$Z> per share

If no material deal, write "No material deals in window" and skip the chain.

## Insider transactions

| Window | Net buy/sell | Notable individuals |
|---|---|---|
| Last 6-12 months | <net $ or share count> | <CEO/CFO/director names & direction> |

Summarize cluster buying/selling and any signal, citing figures from \
raw/insider.json. If raw/insider.json's `transactions` list is empty, state \
"no reported insider transactions in the window" — do not infer activity.

## What management needs to prove

Three falsifiable hurdles. Each: specific metric or event + by-when + threshold.

## Sentiment & consensus

When pm_brief.md carries a "## Sentiment & consensus" block, cite its \
short-interest %/days-to-cover and analyst rating + target upside verbatim; \
else "not reported".
""" + _FOOTER


_FILES_QUALITY = ["pm_brief.md", "reference.json", "financials.json", "sec_filing.md", "news.json"]

_SYSTEM_QUALITY = """\
You are the Competitive-Quality analyst writing that part of an equity \
research report on $TICKER for trade date $DATE. You have been given \
pm_brief.md and the raw data files below. NO tool calls — the data is in \
front of you.

Required sections (use the headers verbatim):

## Competitive position

Porter's Five Forces in brief — competitive rivalry, threat of new entrants, \
threat of substitutes, supplier power, buyer power. State the moat type \
(network effects / switching costs / scale / brand / IP / regulatory) and its \
DURABILITY, plus disruption risk. Ground every claim in raw/sec_filing.md \
(business description / risk factors), news.json, or the peer set.

## Capital-allocation track record

Assess capital-allocation discipline: buybacks vs dividends vs M&A, and whether \
reinvestment earns its cost of capital — cite the "## Accounting ratios" block's \
ROIC and ROIC-WACC spread. Focus on the buyback/dividend/M&A record here; the \
detailed insider-transaction analysis is owned by the catalysts & ownership \
section — do not duplicate it.

## Ownership & governance

Share-class / voting structure (e.g. dual-class or super-voting founder shares), \
board independence, and customer/supplier concentration — each grounded in \
raw/sec_filing.md / news.json, or explicitly "not disclosed in the available \
filing" when absent.

Qualitative-claim discipline: every claim in the three sections above must be \
grounded in a named source (raw/sec_filing.md, news.json, or pm_brief.md). Where \
the available free data does not support a claim, write "not determinable from \
available free filings" — do NOT invent competitive dynamics, moats, governance / \
share-class facts, concentration figures, or management history from memory or \
general knowledge.
""" + _FOOTER


def create_financial_statement_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_FINANCIAL)
        messages = [
            SystemMessage(content=_SYSTEM_FINANCIAL.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                         f"Write the financial-statement analysis."),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Financial-Statement Analyst", min_chars=1200)
        return {"messages": [result], "fundamentals_financial_report": report}
    return node


def create_risk_redflags_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_RISK)
        messages = [
            SystemMessage(content=_SYSTEM_RISK.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                         f"Write the risk & red-flags analysis."),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Risk & Red-Flags Analyst", min_chars=1200)
        return {"messages": [result], "fundamentals_riskflags_report": report}
    return node


def create_catalysts_ownership_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_CATALYSTS)
        messages = [
            SystemMessage(content=_SYSTEM_CATALYSTS.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                         f"Write the catalysts & ownership analysis."),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Catalysts & Ownership Analyst", min_chars=1200)
        return {"messages": [result], "fundamentals_catalysts_report": report}
    return node


def create_competitive_quality_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_QUALITY)
        messages = [
            SystemMessage(content=_SYSTEM_QUALITY.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                         f"Write the competitive-quality analysis."),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Competitive-Quality Analyst", min_chars=1200)
        return {"messages": [result], "fundamentals_quality_report": report}
    return node


_ROLE_SECTIONS = [
    ("Financial-Statement", "fundamentals_financial_report"),
    ("Risk & Red-Flags", "fundamentals_riskflags_report"),
    ("Catalysts & Ownership", "fundamentals_catalysts_report"),
    ("Competitive-Quality", "fundamentals_quality_report"),
]


def create_fundamentals_aggregator():
    """Deterministic (no-LLM) node: concatenate the 4 role reports into the
    existing ``fundamentals_report`` key so downstream consumers are unchanged."""

    def node(state):
        parts = ["# Fundamentals\n"]
        for title, key in _ROLE_SECTIONS:
            body = (state.get(key) or "").strip()
            if not body:
                body = f"_({title} section unavailable)_"
            parts.append(f"## {title}\n\n{body}\n")
        return {"fundamentals_report": "\n".join(parts)}

    return node
