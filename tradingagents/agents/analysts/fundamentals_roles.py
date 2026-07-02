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

Price-target-grid discipline: when pm_brief.md carries a "## Forward-EPS \
price-target grid" block, cite its projected-EPS path, the exit-multiple price \
scenarios, and especially the "Implied P/E if price stays flat" compression row \
VERBATIM in your valuation reasoning — the compression row answers "is the \
growth already priced in?". Do NOT recompute, extend, or re-anchor the grid, \
and do NOT invent price targets beyond it. If the block is marked unavailable, \
cite no forward price targets at all.

## Segments & model-specific KPIs

If raw/sec_filing.md reports business segments, give a brief sum-of-the-parts \
sketch: each segment's revenue and operating margin FROM THE FILING, which \
segment drives value, and — for a multi-segment conglomerate — whether the \
parts would plausibly be worth more separately. Then surface any model-specific \
KPIs the filing discloses: net revenue retention (NRR), ARPU, same-store / \
comparable sales, backlog / remaining performance obligations, LTV/CAC. Quote \
the filing's figures. For any segment split or KPI NOT in the filing, write \
"not disclosed in the available filing" — do NOT invent segment economics or \
KPIs from memory.
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

Goodwill screen discipline: when pm_brief.md's "## Goodwill / impairment screen" \
block reports goodwill, cite its goodwill/equity and goodwill/assets ratios and \
the flag (elevated/normal) verbatim as an impairment-risk note — do NOT compute \
your own goodwill ratio. If the block says "no goodwill reported", state that the \
name carries no goodwill (not a red flag) and cite no ratio.

Refinancing discipline: when pm_brief.md's "## Refinancing / maturity-wall proxy" \
block is applicable, cite its current-debt %, cash coverage, and the flag \
(elevated/moderate/low) verbatim as a near-term rollover-risk note — do NOT invent \
a maturity schedule. Note it is a proxy (current-vs-long-term split), not the full \
10-K maturity ladder.

Commodity exposure discipline: cite pm_brief.md's "## Commodity input exposure" \
block's exposure level (high/moderate/low) and primary inputs verbatim as an \
input-cost / margin-risk note alongside cyclicality — do NOT invent a commodity \
sensitivity the block does not state. A "low" exposure is a valid finding.
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

## Institutional & insider ownership

When pm_brief.md carries a "## Institutional & insider ownership" block, cite \
its institutional ownership %, insider ownership %, and the notable top \
institutional holders (with quarter-over-quarter stake change) verbatim as an \
ownership-structure signal; else "not reported". Do NOT invent holders or \
percentages the block does not state.

## Recent SEC filings

When pm_brief.md carries a "## SEC filing surface" block, cite its recent 8-K \
material-event dates + item categories (a cluster of 8-Ks can flag a live \
catalyst) and the latest DEF 14A proxy date verbatim; else "not reported". \
These are filing dates/categories only — do NOT infer filing contents you \
were not given.

## Activist & large-stake ownership

When pm_brief.md carries a "## Activist & large-stake filings (13D/13G)" block, \
cite its filings verbatim, distinguishing 13D (activist intent) from 13G \
(passive >5% holder) and naming the filer + date; a recent 13D is a potential \
catalyst. Else "not reported". Do NOT infer the activist thesis or current \
stake size — you have only the filing dates, types, and filers.
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
ROIC, ROIC-WACC spread, and Incremental ROIC (ΔNOPAT/ΔIC — reinvestment quality) \
verbatim when present. Focus on the buyback/dividend/M&A record here; the \
detailed insider-transaction analysis is owned by the catalysts & ownership \
section — do not duplicate it.

## Ownership & governance

Share-class / voting structure (e.g. dual-class or super-voting founder shares), \
board independence, and customer/supplier concentration — each grounded in \
raw/sec_filing.md / news.json, or explicitly "not disclosed in the available \
filing" when absent.

## Material ESG risks

From raw/sec_filing.md's risk-factors section, flag any FINANCIALLY MATERIAL \
environmental, social, or governance risk the company itself discloses \
(climate/regulatory exposure, environmental liabilities, labor/safety, \
data-privacy, product/consumer). Quote the filing. Free structured ESG scores \
are unavailable (no reliable free feed), so assess ONLY what the filing \
discloses — write "no material ESG risk disclosed in the available filing" if \
none, and do NOT import an ESG rating, score, or controversy from memory.

Qualitative-claim discipline: every claim in the four sections above must be \
grounded in a named source (raw/sec_filing.md, news.json, or pm_brief.md). Where \
the available free data does not support a claim, write "not determinable from \
available free filings" — do NOT invent competitive dynamics, moats, governance / \
share-class facts, concentration figures, or management history from memory or \
general knowledge.
""" + _FOOTER


ROLE_RETRY_CAP = 2

_REQUIRED_FINANCIAL = ["## Business-model framing", "## Peer comparison matrix",
                       "## Capital-structure compare", "## Sanity check on reported numbers"]
_REQUIRED_RISK = ["## Risk & red flags"]
_REQUIRED_CATALYSTS = ["## Insider transactions", "## What management needs to prove",
                       "## Sentiment & consensus"]
_REQUIRED_QUALITY = ["## Competitive position", "## Capital-allocation track record",
                     "## Ownership & governance"]


def check_role_output(required_headers, report, min_chars=600):
    """Deterministic structural check: every required header present + a length
    floor. Returns human-readable issues; empty list == passed. No LLM, no file
    materialization — cheap enough to run on every role invocation."""
    text = report or ""
    issues = [f"missing required section: {h}" for h in required_headers if h not in text]
    if len(text.strip()) < min_chars:
        issues.append(f"report too short ({len(text.strip())} chars < {min_chars})")
    return issues


def format_role_feedback(issues):
    lines = "\n".join(f"- {i}" for i in issues)
    return ("Your previous draft was incomplete. Fix these before rewriting the "
            "full section:\n" + lines)


def _make_role_node(llm, *, system, files, name, write_verb, required_headers,
                    report_key, passed_key, feedback_key, retries_key):
    """Shared role-node body: build the role prompt (injecting any prior retry
    feedback), invoke, then run the deterministic self-check. On a failed check
    with retries left, returns passed=False + feedback + incremented retries so
    the graph's per-role conditional edge can self-loop this node (Phase 4)."""

    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=files)
        human = (f"For your reference: {instrument_context}\n\n{context}\n\n"
                 f"Write the {write_verb}.")
        prior_fb = state.get(feedback_key, "")
        if prior_fb:
            human += f"\n\n{prior_fb}"
        messages = [
            SystemMessage(content=system.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=human),
        ]
        result, report = invoke_with_empty_retry(llm, messages, name, min_chars=1200)
        issues = check_role_output(required_headers, report)
        passed = not issues
        prior = state.get(retries_key, 0)
        return {
            "messages": [result],
            report_key: report,
            passed_key: passed,
            feedback_key: "" if passed else format_role_feedback(issues),
            retries_key: prior if passed else prior + 1,
        }

    return node


def create_financial_statement_analyst(llm):
    return _make_role_node(
        llm, system=_SYSTEM_FINANCIAL, files=_FILES_FINANCIAL,
        name="Financial-Statement Analyst", write_verb="financial-statement analysis",
        required_headers=_REQUIRED_FINANCIAL,
        report_key="fundamentals_financial_report", passed_key="fundamentals_financial_passed",
        feedback_key="fundamentals_financial_feedback", retries_key="fundamentals_financial_retries")


def create_risk_redflags_analyst(llm):
    return _make_role_node(
        llm, system=_SYSTEM_RISK, files=_FILES_RISK,
        name="Risk & Red-Flags Analyst", write_verb="risk & red-flags analysis",
        required_headers=_REQUIRED_RISK,
        report_key="fundamentals_riskflags_report", passed_key="fundamentals_riskflags_passed",
        feedback_key="fundamentals_riskflags_feedback", retries_key="fundamentals_riskflags_retries")


def create_catalysts_ownership_analyst(llm):
    return _make_role_node(
        llm, system=_SYSTEM_CATALYSTS, files=_FILES_CATALYSTS,
        name="Catalysts & Ownership Analyst", write_verb="catalysts & ownership analysis",
        required_headers=_REQUIRED_CATALYSTS,
        report_key="fundamentals_catalysts_report", passed_key="fundamentals_catalysts_passed",
        feedback_key="fundamentals_catalysts_feedback", retries_key="fundamentals_catalysts_retries")


def create_competitive_quality_analyst(llm):
    return _make_role_node(
        llm, system=_SYSTEM_QUALITY, files=_FILES_QUALITY,
        name="Competitive-Quality Analyst", write_verb="competitive-quality analysis",
        required_headers=_REQUIRED_QUALITY,
        report_key="fundamentals_quality_report", passed_key="fundamentals_quality_passed",
        feedback_key="fundamentals_quality_feedback", retries_key="fundamentals_quality_retries")


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
