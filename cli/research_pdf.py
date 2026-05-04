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
import re
from pathlib import Path

import markdown
from weasyprint import CSS, HTML


_CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm 2cm 1.5cm;

    @top-right {
        content: string(running-header);
        font-size: 9pt;
        color: #888;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #888;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
}

@page :first {
    @top-right { content: ""; }
    @bottom-center { content: ""; }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1d1d1f;
}

.cover {
    page-break-after: always;
    text-align: center;
    padding-top: 4cm;
}

.cover .eyebrow {
    font-size: 11pt;
    letter-spacing: 0.3em;
    color: #6e6e73;
    text-transform: uppercase;
    margin-bottom: 0.5em;
}

.cover .title {
    font-size: 48pt;
    font-weight: 700;
    color: #000;
    line-height: 1.05;
    margin: 0.2em 0 0.4em 0;
}

.cover .subtitle {
    font-size: 18pt;
    color: #424245;
    margin-bottom: 3em;
}

.cover .decision-badge {
    display: inline-block;
    padding: 0.6em 1.6em;
    background: #1d1d1f;
    color: #fff;
    font-size: 16pt;
    font-weight: 600;
    border-radius: 6px;
    letter-spacing: 0.04em;
    margin-bottom: 4em;
}

.cover .meta {
    color: #6e6e73;
    font-size: 10pt;
    line-height: 1.8;
}

.cover .disclaimer {
    margin-top: 6em;
    color: #86868b;
    font-size: 8.5pt;
    font-style: italic;
    max-width: 14cm;
    margin-left: auto;
    margin-right: auto;
}

h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #000;
    border-bottom: 2px solid #1d1d1f;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 0.8em;
    page-break-before: always;
    string-set: running-header content();
}

h2 {
    font-size: 15pt;
    font-weight: 600;
    color: #1d1d1f;
    margin-top: 1.6em;
    margin-bottom: 0.4em;
    break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #424245;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
    break-after: avoid;
}

h4, h5, h6 {
    font-size: 11pt;
    font-weight: 600;
    color: #424245;
    break-after: avoid;
}

p {
    margin: 0 0 0.7em 0;
    text-align: left;
}

strong {
    font-weight: 600;
    color: #000;
}

em {
    font-style: italic;
    color: #1d1d1f;
}

code, pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 9pt;
}

code {
    background: #f5f5f7;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    color: #1d1d1f;
}

pre {
    background: #f5f5f7;
    padding: 0.9em 1.1em;
    border-radius: 6px;
    border-left: 3px solid #1d1d1f;
    overflow-x: auto;
    margin: 1em 0;
    line-height: 1.4;
}

pre code {
    background: transparent;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 10pt;
}

thead {
    /* Repeat the header row when a table splits across pages. */
    display: table-header-group;
}

th {
    background: #f5f5f7;
    font-weight: 600;
    text-align: left;
    padding: 0.6em 0.8em;
    border-bottom: 2px solid #1d1d1f;
    color: #1d1d1f;
}

td {
    padding: 0.5em 0.8em;
    border-bottom: 1px solid #d2d2d7;
    vertical-align: top;
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
}

blockquote {
    border-left: 3px solid #d2d2d7;
    margin: 1em 0;
    padding-left: 1em;
    color: #424245;
    font-style: italic;
}

hr {
    border: none;
    border-top: 1px solid #d2d2d7;
    margin: 2em 0;
}

.section-pretitle {
    font-size: 9pt;
    color: #86868b;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.3em;
    margin-top: 0.5em;
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
    <div class="eyebrow">Trading Research</div>
    <div class="title">{ticker}</div>
    <div class="subtitle">{date_human}</div>
    <div class="decision-badge">{decision_short}</div>
    <div class="meta">
        Generated {generated_at} UTC<br>
        TradingAgents multi-agent pipeline · Opus 4.6 judges · Haiku 4.5 analysts
    </div>
    <div class="disclaimer">
        This document is research output from a simulated multi-agent decision pipeline.
        It is for educational and research purposes only and does not constitute financial,
        investment, or trading advice. Past data does not guarantee future results.
    </div>
</div>

<h1>PM Pre-flight Brief</h1>
<div class="section-pretitle">Run mandate, business-model classification, peer set, framing.</div>
{pm_brief_html}

<h1>Technical Setup</h1>
<div class="section-pretitle">Major historical levels, volume zones, trading playbook.</div>
{technicals_html}

<h1>Portfolio Manager — Final Decision</h1>
<div class="section-pretitle">Synthesizes all preceding analysis into a single actionable verdict.</div>
{decision_html}

<h1>Risk Team Debate</h1>
<div class="section-pretitle">Aggressive vs Neutral vs Conservative — three voices stress-test the trader's proposal.</div>
{debate_risk_html}

<h1>Bull vs Bear Debate</h1>
<div class="section-pretitle">Adversarial researchers argue the case for and against, judged by the Research Manager.</div>
{debate_bull_bear_html}

<h1>Market Analyst</h1>
<div class="section-pretitle">Price action, technicals, indicators.</div>
{analyst_market_html}

<h1>Fundamentals Analyst</h1>
<div class="section-pretitle">Earnings, valuation, balance sheet.</div>
{analyst_fundamentals_html}

<h1>News Analyst</h1>
<div class="section-pretitle">Macroeconomic and event flow.</div>
{analyst_news_html}

<h1>Social Sentiment Analyst</h1>
<div class="section-pretitle">Public sentiment and social-media mood.</div>
{analyst_social_html}

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


def build_research_pdf(
    output_dir: str, ticker: str, date: str, decision: str
) -> Path:
    """Combine the per-run markdown reports into one styled PDF.

    Returns the Path of the generated PDF.
    """
    out = Path(output_dir)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "nl2br"])

    def render_md(filename: str) -> str:
        path = out / filename
        if not path.exists():
            return f"<p><em>(missing: {filename})</em></p>"
        text = path.read_text(encoding="utf-8")
        return _demote_h1_to_h2(md.reset().convert(text))

    decision_md_text = ""
    decision_md_path = out / "decision.md"
    if decision_md_path.exists():
        decision_md_text = decision_md_path.read_text(encoding="utf-8")

    decision_short = _summarize_decision(decision_md_text or decision)

    pm_brief_html = render_md_from_path(out / "raw" / "pm_brief.md")
    technicals_html = render_md_from_path(out / "raw" / "technicals.md")

    html = _HTML_TEMPLATE.format(
        ticker=ticker,
        date=date,
        date_human=_format_date_human(date),
        decision_short=decision_short,
        generated_at=_dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        pm_brief_html=pm_brief_html,
        technicals_html=technicals_html,
        decision_html=render_md("decision.md"),
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
