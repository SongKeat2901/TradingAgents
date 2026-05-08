"""Build a styled PDF report from the per-run markdown output files.

Combines decision.md + the analyst reports + debate transcripts into one
document with a cover page, section breaks, and CSS-styled typography.
Used by the Telegram notifier to deliver a readable artifact in chat
instead of the truncated 4096-char inline message.

Requires `markdown` and `weasyprint`. WeasyPrint also needs cairo and
pango at the OS level; on macOS install via `brew install cairo pango`.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import markdown
from weasyprint import CSS, HTML


_CSS = """
/* Phase 6.6 — TrueKnot brand-pack tokens applied throughout.
   Source: docs/brand/letterhead.html + docs/brand/brand-pack.html in
   the TrueKnot Web Setup project. */

@page {
    size: A4;
    margin: 2.4cm 1.5cm 2cm 1.5cm;

    @top-left {
        content: "TRUEKNOT";
        font-size: 8.5pt;
        color: #0a0f1e;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-weight: 800;
        letter-spacing: 0.14em;
    }

    @top-right {
        content: string(running-header);
        font-size: 8.5pt;
        color: #6c6f78;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    @bottom-left {
        content: "TrueKnot Pte. Ltd. · trueknot.sg";
        font-size: 7.5pt;
        color: #a9abb0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 7.5pt;
        color: #a9abb0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
}

@page :first {
    @top-left { content: ""; }
    @top-right { content: ""; }
    @bottom-left { content: ""; }
    @bottom-right { content: ""; }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #0a0f1e;
}

/* ── Cover page ── */
.cover {
    page-break-after: always;
    text-align: center;
    padding-top: 3cm;
}

.cover-brand {
    margin-bottom: 2.8cm;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5em;
}

.cover-brand-lockup {
    display: inline-flex;
    align-items: center;
    gap: 0.6em;
}

.cover-knot {
    width: 14mm;
    height: 14mm;
}

.cover-wordmark {
    font-size: 18pt;
    font-weight: 800;
    letter-spacing: 0.14em;
    color: #0a0f1e;
}

.cover-tagline {
    font-size: 10pt;
    color: #6c6f78;
    font-style: italic;
    letter-spacing: 0.04em;
}

.cover .eyebrow {
    font-size: 10pt;
    letter-spacing: 0.32em;
    color: #6c6f78;
    text-transform: uppercase;
    margin-bottom: 0.5em;
}

.cover .title {
    font-size: 56pt;
    font-weight: 800;
    color: #0a0f1e;
    line-height: 1.0;
    letter-spacing: -0.01em;
    margin: 0.2em 0 0.4em 0;
}

.cover .subtitle {
    font-size: 16pt;
    color: #4f525d;
    margin-bottom: 3em;
    font-weight: 400;
}

.cover .decision-badge {
    display: inline-block;
    padding: 0.65em 1.8em;
    background: #0a0f1e;
    color: #ffffff;
    font-size: 15pt;
    font-weight: 700;
    border-radius: 4px;
    letter-spacing: 0.08em;
    margin-bottom: 3.5em;
}

.cover .meta {
    color: #6c6f78;
    font-size: 9pt;
    line-height: 1.8;
    letter-spacing: 0.02em;
}

.cover .disclaimer {
    margin-top: 4em;
    color: #a9abb0;
    font-size: 8pt;
    font-style: italic;
    line-height: 1.55;
    max-width: 13cm;
    margin-left: auto;
    margin-right: auto;
}

.cover-corporate-footer {
    position: absolute;
    bottom: 1.2cm;
    left: 1.5cm;
    right: 1.5cm;
    border-top: 0.3mm solid #e7e7e9;
    padding-top: 0.5em;
    display: flex;
    justify-content: space-between;
    font-size: 7.5pt;
    color: #a9abb0;
    letter-spacing: 0.02em;
}

/* ── Section headers ── */
h1 {
    font-size: 22pt;
    font-weight: 800;
    color: #0a0f1e;
    border-bottom: 0.5mm solid #0a0f1e;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 0.7em;
    page-break-before: always;
    string-set: running-header content();
    letter-spacing: -0.005em;
}

h2 {
    font-size: 14pt;
    font-weight: 700;
    color: #0a0f1e;
    margin-top: 1.6em;
    margin-bottom: 0.4em;
    break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: 700;
    color: #4f525d;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
    break-after: avoid;
    letter-spacing: 0.01em;
}

h4, h5, h6 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #4f525d;
    break-after: avoid;
}

p {
    margin: 0 0 0.7em 0;
    text-align: left;
    color: #4f525d;
}

strong {
    font-weight: 700;
    color: #0a0f1e;
}

em {
    font-style: italic;
    color: #0a0f1e;
}

code, pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 9pt;
}

code {
    background: #f8f9fb;
    padding: 0.1em 0.3em;
    border-radius: 2px;
    color: #0a0f1e;
    border: 0.2mm solid #e7e7e9;
}

pre {
    background: #f8f9fb;
    padding: 0.9em 1.1em;
    border-radius: 4px;
    border-left: 0.5mm solid #0a0f1e;
    overflow-x: auto;
    margin: 1em 0;
    line-height: 1.4;
    color: #0a0f1e;
}

pre code {
    background: transparent;
    padding: 0;
    border: none;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 9.5pt;
}

thead {
    /* Repeat the header row when a table splits across pages. */
    display: table-header-group;
}

th {
    background: #f8f9fb;
    font-weight: 700;
    text-align: left;
    padding: 0.6em 0.8em;
    border-bottom: 0.4mm solid #0a0f1e;
    color: #0a0f1e;
    letter-spacing: 0.02em;
}

td {
    padding: 0.55em 0.8em;
    border-bottom: 0.2mm solid #e7e7e9;
    vertical-align: top;
    color: #4f525d;
}

td strong, th strong {
    color: #0a0f1e;
}

tr:last-child td {
    border-bottom: none;
}

ul, ol {
    margin: 0.4em 0 0.9em 1.4em;
    padding: 0;
}

li {
    margin-bottom: 0.3em;
    color: #4f525d;
}

li strong {
    color: #0a0f1e;
}

blockquote {
    border-left: 0.5mm solid #e7e7e9;
    margin: 1em 0;
    padding-left: 1em;
    color: #6c6f78;
    font-style: italic;
}

hr {
    border: none;
    border-top: 0.2mm solid #e7e7e9;
    margin: 2em 0;
}

.section-pretitle {
    font-size: 8.5pt;
    color: #a9abb0;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.3em;
    margin-top: 0.5em;
    font-weight: 600;
}

.appendix-divider {
    page-break-before: always;
    text-align: center;
    border: none;
    color: #0a0f1e;
    font-size: 22pt;
    font-weight: 800;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    padding: 6cm 0 0 0;
    margin: 0;
}

.appendix-divider + .section-pretitle {
    text-align: center;
    font-size: 9pt;
    color: #6c6f78;
    margin-top: 0.6em;
    margin-bottom: 4cm;
}

.exec-summary-banner {
    background: #f8f9fb;
    border-left: 0.6mm solid #0a0f1e;
    padding: 1em 1.2em;
    margin: 0 0 1.5em 0;
    font-size: 9.5pt;
    color: #4f525d;
    font-style: italic;
    line-height: 1.6;
}
"""


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{ticker} Research — {date}</title>
</head>
<body>

<div class="cover">
    <div class="cover-brand">
        <div class="cover-brand-lockup">
            <svg class="cover-knot" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <mask id="cv-knot-hm" maskUnits="userSpaceOnUse" x="0" y="0" width="56" height="56">
                        <rect width="56" height="56" fill="white"/>
                        <rect x="14.64" y="14.64" width="10" height="10" fill="black"/>
                        <rect x="31.36" y="31.36" width="10" height="10" fill="black"/>
                    </mask>
                    <mask id="cv-knot-vm" maskUnits="userSpaceOnUse" x="0" y="0" width="56" height="56">
                        <rect width="56" height="56" fill="white"/>
                        <rect x="31.36" y="14.64" width="10" height="10" fill="black"/>
                        <rect x="14.64" y="31.36" width="10" height="10" fill="black"/>
                    </mask>
                </defs>
                <ellipse cx="28" cy="28" rx="21" ry="8" fill="none" stroke="#0a0f1e" stroke-width="4.5" mask="url(#cv-knot-hm)"/>
                <ellipse cx="28" cy="28" rx="8" ry="21" fill="none" stroke="#0a0f1e" stroke-width="4.5" mask="url(#cv-knot-vm)"/>
            </svg>
            <span class="cover-wordmark">TRUEKNOT</span>
        </div>
        <div class="cover-tagline">Untangle what matters.</div>
    </div>

    <div class="eyebrow">Trading Research</div>
    <div class="title">{ticker}</div>
    <div class="subtitle">{date_human}</div>
    <div class="decision-badge">{decision_short}</div>
    <div class="meta">
        Generated {generated_at} SGT<br>
        Multi-agent research pipeline · {model_label}
    </div>
    <div class="disclaimer">
        This document is research output from a simulated multi-agent decision pipeline.
        It is for educational and research purposes only and does not constitute financial,
        investment, or trading advice. Past data does not guarantee future results.
    </div>
    <div class="cover-corporate-footer">
        <span>TrueKnot Pte. Ltd. &nbsp;·&nbsp; UEN 202608241M</span>
        <span>1 Bukit Batok Cres, #05-15, Singapore 658064 &nbsp;·&nbsp; trueknot.sg</span>
    </div>
</div>

<h1>Executive Summary</h1>
<div class="section-pretitle">Rating, scenario analysis, trading plan.</div>
<div class="exec-summary-banner">Distilled from the full investment recommendation. See pages that follow for setup, technical context, and complete reasoning. Operational source material (analyst reports, debate transcripts) appears in the appendix at the back.</div>
{executive_summary_html}

<h1>Investment Thesis</h1>
<div class="section-pretitle">Setup, business-model framing, peer set, what this run must answer.</div>
{pm_brief_html}

<h1>Technical Setup</h1>
<div class="section-pretitle">Major historical levels, volume zones, trading playbook.</div>
{technicals_html}

<h1>Investment Recommendation</h1>
<div class="section-pretitle">Stakeholder-voice executive read of the final actionable verdict.</div>
{decision_html}

<h1 class="appendix-divider">Appendix</h1>
<div class="section-pretitle">Source material — kept verbatim for review and troubleshooting.</div>

<h1>Appendix A — Risk Team Debate</h1>
<div class="section-pretitle">Aggressive vs Neutral vs Conservative — three voices stress-test the proposal.</div>
{debate_risk_html}

<h1>Appendix B — Bull vs Bear Debate</h1>
<div class="section-pretitle">Adversarial researchers argue the case for and against.</div>
{debate_bull_bear_html}

<h1>Appendix C — Market Analyst Notes</h1>
<div class="section-pretitle">Price action, technicals, indicators.</div>
{analyst_market_html}

<h1>Appendix D — Fundamentals Analyst Notes</h1>
<div class="section-pretitle">Earnings, valuation, balance sheet.</div>
{analyst_fundamentals_html}

<h1>Appendix E — News Analyst Notes</h1>
<div class="section-pretitle">Macroeconomic and event flow.</div>
{analyst_news_html}

<h1>Appendix F — Social Sentiment Analyst Notes</h1>
<div class="section-pretitle">Public sentiment and social-media mood.</div>
{analyst_social_html}

<h1>Appendix G — PM Working Notes</h1>
<div class="section-pretitle">Unfiltered PM decision document — multi-agent process, reconciliation tables, "what I am rejecting" — kept verbatim for audit. The Investment Recommendation above is the stakeholder-voice translation of this material.</div>
{decision_working_notes_html}

</body>
</html>
"""


_DECISION_PATTERNS = (
    "FINAL TRANSACTION PROPOSAL:",
    "Final Rating:",
    "**Decision:**",
    "Decision:",
)


def _summarize_decision(decision_text: str, fallback: str = "See full report") -> str:
    """Pull a short headline from the decision.md or raw decision string."""
    if not decision_text:
        return fallback
    for line in decision_text.splitlines():
        for pat in _DECISION_PATTERNS:
            if pat in line:
                cleaned = re.sub(r"[*#`]", "", line).strip()
                # Strip the pattern prefix to get just the verdict
                idx = cleaned.find(pat.replace("*", "").replace("#", "").strip())
                if idx >= 0:
                    cleaned = cleaned[idx + len(pat) - 2 * pat.count("*"):].strip(": ").strip()
                if cleaned:
                    return cleaned[:80]
    # Fallback: first non-empty non-heading line
    for line in decision_text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return re.sub(r"[*`]", "", s)[:80]
    return fallback


def _format_date_human(date: str) -> str:
    """`2024-05-10` → `May 10, 2024`."""
    try:
        return _dt.datetime.strptime(date, "%Y-%m-%d").strftime("%B %-d, %Y")
    except ValueError:
        return date


def _demote_h1_to_h2(html: str) -> str:
    """Demote nested <h1> to <h2> so only the template's section dividers
    trigger CSS `page-break-before: always`. Without this, every `# heading`
    inside an analyst report forces a page break, leaving the outer section
    title orphaned on a near-empty page.
    """
    return html.replace("<h1>", "<h2>").replace("</h1>", "</h2>")


def render_md_from_path(path: Path) -> str:
    """Render markdown from an arbitrary path; returns '(missing)' if absent."""
    if not path.exists():
        return "<em>(missing)</em>"
    text = path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    return _demote_h1_to_h2(html)


# Phase 6.5 executive-format cleanup. Replace internal multi-agent vocabulary
# in front-of-document sections with executive-friendly prose. Appendix
# sections are left verbatim — the user wants the operational detail
# preserved there for review/troubleshooting.
_AGENTIC_VOCAB_REPLACEMENTS: list[tuple[str, str]] = [
    # Order matters: most-specific patterns first so generic catch-alls
    # don't shadow them.

    # Citation-style references to debate paragraph locations — these are
    # internal traceability metadata (e.g. "(Aggressive, opening paragraph;
    # ≤30 words paraphrased)"). Strip them entirely; the appendix has the
    # original text if anyone wants to verify.
    (r"\s*\((?:Aggressive|Conservative|Neutral|Bull|Bear)[,;]\s*[^)]*paraphrased\s*\)", ""),
    (r"\s*\((?:Aggressive|Conservative|Neutral|Bull|Bear)[,;]\s*[Pp]oint\s*\d+[^)]*\)", ""),
    (r"\s*\((?:Aggressive|Conservative|Neutral|Bull|Bear)[,;]\s*[Pp]aragraph\s*\d+[^)]*\)", ""),

    # Multi-word phrases referencing specific agent stances:
    (r"\bAggressive's strongest punch\b", "aggressive case's strongest argument"),
    (r"\bConservative's strongest punch\b", "conservative case's strongest argument"),
    (r"\bConservative's structural punch\b", "conservative case's structural argument"),
    (r"\bConservative's load-bearing punch\b", "conservative case's load-bearing argument"),
    (r"\bAggressive's load-bearing punch\b", "aggressive case's load-bearing argument"),
    (r"\bAggressive overreaches\b", "The aggressive case overreaches"),
    (r"\bAggressive omitted\b", "The aggressive case omitted"),
    (r"\bConservative correctly notes\b", "The conservative case correctly notes"),
    (r"\bNeutral lands the sequencing verdict\b", "The neutral case lands the sequencing verdict"),
    (r"\bNeutral lands\b", "The neutral case lands"),
    (r"\bAggressive lands\b", "The aggressive case lands"),
    (r"\bConservative lands\b", "The conservative case lands"),
    (r"\bThe Trader's HOLD proposal\b", "The trader proposal (HOLD)"),
    (r"\bThe Trader's SELL proposal\b", "The trader proposal (SELL)"),
    (r"\bTrader's HOLD proposal\b", "trader proposal (HOLD)"),
    (r"\bTrader's SELL proposal\b", "trader proposal (SELL)"),
    (r"\bThe Research Manager's verdict\b", "The research synthesis"),
    (r"\bResearch Manager's verdict\b", "research synthesis"),
    (r"\bThe Aggressive Risk Analyst's\b", "The aggressive case's"),
    (r"\bThe Conservative Risk Analyst's\b", "The conservative case's"),
    (r"\bThe Neutral Risk Analyst's\b", "The neutral case's"),
    (r"\bAll three risk analysts\b", "All three risk perspectives"),
    (r"\ball three risk analysts\b", "all three risk perspectives"),
    # Generic agent-role names (after the specific phrases):
    (r"\bThe Trader's\b", "The trader's"),
    (r"\bThe Trader\b", "The trader"),
    (r"\bThe Research Manager's\b", "The research synthesis's"),
    (r"\bResearch Manager's\b", "research synthesis's"),
    (r"\bResearch Manager\b", "Research synthesis"),
    (r"\bAggressive Risk Analyst\b", "aggressive case"),
    (r"\bConservative Risk Analyst\b", "conservative case"),
    (r"\bNeutral Risk Analyst\b", "neutral case"),
    (r"\bAggressive's\b", "the aggressive case's"),
    (r"\bConservative's\b", "the conservative case's"),
    (r"\bNeutral's\b", "the neutral case's"),
    (r"\bthe risk debate\b", "the risk analysis"),
    (r"\bRisk Debate\b", "Risk Analysis"),
    # File-path leaks.
    (r"raw/peer_ratios\.json", "the peer-ratios dataset"),
    (r"raw/peers\.json", "the peer dataset"),
    (r"raw/sec_filing\.md", "the 10-Q text"),
    (r"raw/financials\.json", "the financials dataset"),
    (r"raw/calendar\.json", "the earnings calendar"),
    (r"raw/reference\.json", "the reference snapshot"),
    (r"raw/classification\.json", "the technical classifier output"),
    (r"raw/pm_brief\.md", "the setup brief"),
    (r"\bpm_brief\.md\b", "the setup brief"),
    (r"\bpeer_ratios\.json\b", "the peer-ratios dataset"),
    # Internal v1/v2 pass labels.
    (r"\bTA Agent v2\b", "Technical analysis"),
    (r"\bTA v2\b", "Technical analysis"),
    (r"\bTA v1\b", "Technical analysis (first pass)"),
    (r" — v2 Report\b", ""),
    (r" — v2\b", ""),
    (r"\(v2\)", ""),
    # PM Pre-flight terminology.
    (r"\bPM Pre-flight Brief\b", "Setup brief"),
    (r"\bPM Pre-flight\b", "Setup"),
    # QC framework refs that occasionally leak.
    (r"\bItem 16[a-c]?\b", "the numerical-trace check"),
    (r"\bItem 15\b", "the filing-anchor check"),
    (r"\bQC checklist\b", "quality check"),
    (r"\bQC verdict\b", "quality verdict"),
    # Inherited-debate-transcript references (these belong to debug history).
    (r"\bprior PM transcripts\b", "prior decisions"),
    (r"\bprior debate transcripts\b", "prior decisions"),
    (r"\binherited from prior debate\b", "carried over from prior decisions"),
]


def _clean_agentic_vocabulary(text: str) -> str:
    """Replace internal multi-agent vocabulary with executive-friendly equivalents.
    Applied to front-of-document Markdown only (NOT appendix); the user wants
    the operational language preserved in the appendix for troubleshooting."""
    for pattern, replacement in _AGENTIC_VOCAB_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


# Phase 6.5: deterministic blocks (Phase 6.2 calendar, 6.3 SEC filing footer,
# 6.4 peer ratios) carry instruction tails directed at the analyst-tier LLM
# ("Use these dates verbatim. Do not write 'data to follow'...", "Treat as
# known data, never as 'pending adjudication'...", etc.). Those instructions
# do their job in pm_brief.md (where the LLM reads them) but they leak into
# the executive-facing PDF. Strip them from front-of-document rendering only.
_LLM_DIRECTIVE_PATTERNS: list[str] = [
    # Calendar block instruction tail (italic standalone paragraph).
    r"\*Use these dates verbatim\..*?catalyst windows\.\*",
    # Peer ratios block instruction tail. The Phase 6.4 v1 footer ended at
    # "from memory.*"; the v2 footer (commit bc7f289) extended past that with
    # the ND/EBITDA = (n/m) explanation, ending at "substitute multiple).*".
    # The original `from memory\.\*` regex broke once v2 shipped because the
    # closing `*` is no longer adjacent to "from memory." — the new pattern
    # captures through the ND/EBITDA paragraph.
    r"\*Use these values verbatim\..*?(?:substitute multiple\)|from memory)\.\*",
    # Phase 6.5 net-debt block footer. v1 ended at "Do not introduce cells
    # not in this table.***"; v2 (commit 499f4bd) extended with the AMD
    # 10-Q crosscheck reference, ending at "...not present in either source.***".
    # The non-greedy regex captures both shapes.
    r"\*Use the cells above verbatim\..*?(?:in either source|in this table)\.\*\*\*",
    # Phase 6.9 latest-session block footer. The text starts with
    # `*Authoritative spot for this report:` and runs through the closing
    # asterisk after "...never as a realized close.*". Non-greedy match
    # captures the whole instructional paragraph (incl. the COIN
    # 2026-05-08 incident reference) so it doesn't leak into the PDF.
    r"\*Authoritative spot for this report:.*?never as a realized close\.\*",
    # Phase 6.9 unavailable warning when raw/prices.json has no rows.
    r"\*\*Price-history unavailable\*\*[\s\S]*?from memory\.\n",
    # Phase 6.9 trade-date-after-latest-session note (italic standalone).
    r"\*\*Note: trade_date .*? has NOT yet indexed that session's close\.\*\*",
    # Phase 6.4 v2 + Phase 6.5 v2 unavailable warnings. These appear when
    # peer_ratios.json or net_debt cells couldn't be computed; the warning
    # itself is an LLM imperative ("Do not cite peer ratios in this report")
    # that an executive reader doesn't need to see.
    r"\*\*All peers unavailable\*\*[\s\S]*?from memory\.\n",
    r"\*\*Balance-sheet net-debt cells unavailable\*\*[\s\S]*?from memory\.\n",
    r"\*\*Peer-ratios table could not be rendered\*\*[\s\S]*?\.\n",
    # SEC filing footer suffix: "; full text in raw/sec_filing.md. Treat as
    # **known data**, never as "pending adjudication" or "awaiting filing"."
    # The whole tail is operational metadata + LLM imperative; an executive
    # reader doesn't need either piece.
    r";\s*full text in raw/sec_filing\.md\.\s*Treat as \*\*known data\*\*, never as \"pending adjudication\" or \"awaiting filing\"\.",
    # Bare imperative tails (in case the prefix already stripped or
    # the pm_brief.md suffix differs).
    r"Treat as \*\*known data\*\*, never as \"pending adjudication\" or \"awaiting filing\"\.",
    r"Treat them as known data, NEVER as 'pending adjudication' or 'awaiting filing'\.",
]


def _strip_llm_directives(text: str) -> str:
    """Remove LLM-targeting instruction tails from deterministic blocks.

    Calendar / peer-ratios / SEC-filing blocks carry trailing imperatives
    aimed at the analyst-tier LLM. Those should NOT appear in the
    executive-facing PDF surface. This function strips them; pm_brief.md
    on disk is unchanged (the LLM still reads the directives at runtime)."""
    for pattern in _LLM_DIRECTIVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


def _extract_section(md_text: str, section_pattern: str) -> str | None:
    """Pull a single section (header + body) out of a Markdown document.
    `section_pattern` is a regex that should match the header line (e.g.
    r'^## 12-Month Scenario Analysis\\s*$'). Returns the section content
    starting with the header up to the next same-or-higher-level header,
    or None if not found."""
    lines = md_text.split("\n")
    start = None
    header_level = None
    for i, line in enumerate(lines):
        if re.match(section_pattern, line):
            start = i
            # Count leading # characters
            m = re.match(r"^(#+)\s", line)
            header_level = len(m.group(1)) if m else 2
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#+)\s", lines[j])
        if m and len(m.group(1)) <= header_level:
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()


def _build_executive_summary_md(decision_md: str) -> str:
    """Distil decision_executive.md (Phase 6.7) or decision.md into a 1-page
    executive summary for page 2 of the PDF.

    Phase 6.7 path (preferred): when the input has the "## Executive Summary"
    header, pull that section + "## Rating and Trading Plan" verbatim. The
    Executive PM has already produced stakeholder-voiced content; no
    additional regex extraction needed beyond picking the load-bearing
    sections.

    Pre-Phase-6.7 fallback: input is the old PM working-notes format (Bottom
    Line / 12-Month Scenario Analysis / Immediate action). Used when the
    Executive PM output is missing — e.g., older runs, or runs where the
    working notes were empty so the Executive PM short-circuited.
    """
    if not decision_md:
        return "_(no decision document available)_"

    parts: list[str] = []

    # Phase 6.7 path: input is decision_executive.md with the new template.
    if "## Executive Summary" in decision_md:
        exec_sum = _extract_section(decision_md, r"^## Executive Summary\s*$")
        if exec_sum:
            parts.append(exec_sum)
        rating_plan = _extract_section(decision_md, r"^## Rating and Trading Plan\s*$")
        if rating_plan:
            parts.append(rating_plan)
        if parts:
            return "\n\n".join(parts)
        # If the new sections somehow weren't found despite the header,
        # fall through to the pre-Phase-6.7 path (defensive).

    # Pre-Phase-6.7 path: input is decision.md with old section names.
    # 1. Pull final rating from "## Bottom Line".
    bottom = _extract_section(decision_md, r"^## Bottom Line\s*$")
    if bottom:
        # Promote h2 → h2 (kept) and replace "Bottom Line" with "Verdict"
        bottom = re.sub(r"^## Bottom Line\s*$", "## Verdict", bottom, flags=re.MULTILINE)
        parts.append(bottom)

    # 2. Pull the scenario table (probabilities + targets + drivers).
    scenarios = _extract_section(decision_md, r"^## 12-Month Scenario Analysis\s*$")
    if scenarios:
        # Trim the trailing "Rating implication" block since we already
        # have the rating in the Bottom Line above.
        scenarios = re.sub(
            r"\n\*\*Rating implication.*$",
            "",
            scenarios,
            flags=re.DOTALL,
        )
        parts.append(scenarios)

    # 3. Pull the trading plan's immediate-action sub-table.
    plan = _extract_section(decision_md, r"^### Immediate action.*$")
    if plan:
        parts.append("## Trading Plan\n\n" + plan.replace("### Immediate action", "### Immediate action").lstrip("# ").strip())

    if not parts:
        # Fallback: if we couldn't locate the named sections, emit a brief
        # placeholder rather than a misleading blank page.
        return "_(executive summary unavailable — see Investment Recommendation)_"

    return "\n\n".join(parts)


def _humanize_model_id(model_id: str | None) -> str:
    """Render a model id like 'claude-opus-4-6' as 'Opus 4.6' for the cover.
    Returns the raw id if the pattern doesn't match. None → '(unknown)'."""
    if not model_id:
        return "(unknown)"
    m = re.match(r"claude-(opus|sonnet|haiku)-(\d+)-(\d+)", model_id)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}.{m.group(3)}"
    return model_id


def _resolve_model_label(out: Path) -> str:
    """Read state.json's `_meta` block (if present) and render a human-readable
    'Opus 4.6 judges · Haiku 4.5 analysts' label. Falls back to '(model not
    recorded)' when the run pre-dates _meta or the state file is missing."""
    state_path = out / "state.json"
    if not state_path.exists():
        return "(model not recorded)"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "(model not recorded)"
    meta = state.get("_meta") or {}
    deep = _humanize_model_id(meta.get("deep_think_llm"))
    quick = _humanize_model_id(meta.get("quick_think_llm"))
    if deep == "(unknown)" and quick == "(unknown)":
        return "(model not recorded)"
    return f"{deep} judges · {quick} analysts"


def build_research_pdf(
    output_dir: str, ticker: str, date: str, decision: str
) -> Path:
    """Combine the per-run markdown reports into one styled PDF.

    Returns the Path of the generated PDF.
    """
    out = Path(output_dir)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "nl2br"])

    def render_md(filename: str) -> str:
        """Render a Markdown file verbatim. Used for appendix sections that
        the user wants kept intact for troubleshooting reference."""
        path = out / filename
        if not path.exists():
            return f"<p><em>(missing: {filename})</em></p>"
        text = path.read_text(encoding="utf-8")
        return _demote_h1_to_h2(md.reset().convert(text))

    def render_md_polished(filename: str) -> str:
        """Render a Markdown file with executive-format polish: strip
        LLM-targeting directives + replace agent vocabulary. Used for
        front-of-document sections (Investment Thesis, Technical Setup,
        Investment Recommendation, Executive Summary)."""
        path = out / filename
        if not path.exists():
            return f"<p><em>(missing: {filename})</em></p>"
        text = path.read_text(encoding="utf-8")
        text = _strip_llm_directives(text)
        text = _clean_agentic_vocabulary(text)
        return _demote_h1_to_h2(md.reset().convert(text))

    def render_md_polished_from_path(path: Path) -> str:
        if not path.exists():
            return "<em>(missing)</em>"
        text = path.read_text(encoding="utf-8")
        text = _strip_llm_directives(text)
        text = _clean_agentic_vocabulary(text)
        return _demote_h1_to_h2(md.reset().convert(text))

    decision_md_text = ""
    decision_md_path = out / "decision.md"
    if decision_md_path.exists():
        decision_md_text = decision_md_path.read_text(encoding="utf-8")

    # Phase 6.7: prefer decision_executive.md as the prominent Investment
    # Recommendation source. The original decision.md (PM working notes) is
    # preserved as Appendix F for audit. Falls back to decision.md when the
    # executive translation is absent (older runs pre-dating Phase 6.7, or
    # runs where the working notes were empty).
    decision_executive_md_text = ""
    decision_executive_md_path = out / "decision_executive.md"
    if decision_executive_md_path.exists():
        decision_executive_md_text = decision_executive_md_path.read_text(encoding="utf-8")

    decision_short = _summarize_decision(
        decision_executive_md_text or decision_md_text or decision
    )

    # Executive Summary section (page 2) — extracted from the executive
    # document when present (cleaner section names + already stakeholder-
    # voiced); falls back to regex extraction from decision.md for older runs.
    executive_summary_md = _build_executive_summary_md(
        decision_executive_md_text or decision_md_text
    )
    executive_summary_md = _strip_llm_directives(executive_summary_md)
    executive_summary_md = _clean_agentic_vocabulary(executive_summary_md)
    executive_summary_html = _demote_h1_to_h2(md.reset().convert(executive_summary_md))

    # Front-of-document sections get the polish pass; appendix sections do not.
    pm_brief_html = render_md_polished_from_path(out / "raw" / "pm_brief.md")
    technicals_v2 = out / "raw" / "technicals_v2.md"
    technicals_v1 = out / "raw" / "technicals.md"
    technicals_html = render_md_polished_from_path(
        technicals_v2 if technicals_v2.exists() else technicals_v1
    )

    # Phase 6.7: Investment Recommendation = decision_executive.md when
    # present (stakeholder voice); fall back to decision.md for runs that
    # pre-date Phase 6.7. Working notes always rendered as Appendix G.
    if decision_executive_md_path.exists():
        decision_html = render_md_polished("decision_executive.md")
        decision_working_notes_html = render_md("decision.md")
    else:
        decision_html = render_md_polished("decision.md")
        decision_working_notes_html = (
            "<em>(working notes are the same document as the Investment "
            "Recommendation above — this run pre-dates Phase 6.7 stakeholder-"
            "voice translation)</em>"
        )

    html = _HTML_TEMPLATE.format(
        ticker=ticker,
        date=date,
        date_human=_format_date_human(date),
        decision_short=decision_short,
        generated_at=_dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"),
        model_label=_resolve_model_label(out),
        executive_summary_html=executive_summary_html,
        pm_brief_html=pm_brief_html,
        technicals_html=technicals_html,
        decision_html=decision_html,
        decision_working_notes_html=decision_working_notes_html,
        debate_risk_html=render_md("debate_risk.md"),
        debate_bull_bear_html=render_md("debate_bull_bear.md"),
        analyst_market_html=render_md("analyst_market.md"),
        analyst_fundamentals_html=render_md("analyst_fundamentals.md"),
        analyst_news_html=render_md("analyst_news.md"),
        analyst_social_html=render_md("analyst_social.md"),
    )

    pdf_path = out / f"research-{date}-{ticker}.pdf"
    HTML(string=html).write_pdf(str(pdf_path), stylesheets=[CSS(string=_CSS)])
    return pdf_path
