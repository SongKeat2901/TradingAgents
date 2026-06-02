"""Customer-facing PDF hardening (2026-05-31 A+ goal, phase P1).

The cover page must not leak the internal agent architecture, and every
rendered section (front-of-document AND appendix) must be scrubbed of
internal scaffolding — raw/ file paths, QC rule numbers, internal pass
labels — before it reaches the customer-facing PDF. The raw .md files on
disk remain verbatim for operator audit; only the PDF surface is polished.
"""

import pytest

pytestmark = pytest.mark.unit


def test_cover_template_has_no_architecture_leak():
    from cli.research_pdf import _HTML_TEMPLATE

    low = _HTML_TEMPLATE.lower()
    assert "simulated multi-agent" not in low, "cover discloses simulated pipeline"
    assert "multi-agent research pipeline" not in low, "cover discloses agent architecture"
    # The model-architecture label ("Opus judges · Sonnet analysts") must not be
    # a customer-facing cover element.
    assert "{model_label}" not in _HTML_TEMPLATE, "cover still renders the model label"
    assert "judges" not in low


def test_cover_keeps_a_not_advice_disclaimer():
    """Dropping the architecture leak must NOT drop the not-investment-advice
    disclaimer — that has to stay for a customer-facing document."""
    from cli.research_pdf import _HTML_TEMPLATE

    low = _HTML_TEMPLATE.lower()
    assert "does not constitute financial" in low or "not constitute" in low
    assert "advice" in low


def test_clean_agentic_vocabulary_scrubs_raw_file_paths():
    from cli.research_pdf import _clean_agentic_vocabulary as scrub

    # Specific raw/ paths get friendly names; the generic catch-all handles
    # any raw/<file> not individually mapped.
    assert "raw/" not in scrub("see raw/forward_probabilities.json for the bands")
    assert "raw/" not in scrub("derived in raw/net_debt.json and raw/volume_profile.json")
    assert "raw/" not in scrub("per raw/peer_ratios.json the peer trades at 20x")


def test_clean_agentic_vocabulary_scrubs_bare_internal_filenames():
    from cli.research_pdf import _clean_agentic_vocabulary as scrub

    assert "net_debt.json" not in scrub("computed in net_debt.json")
    assert "peers.json" not in scrub("the peers.json cell shows 24x")
    assert "forward_probabilities.json" not in scrub("from forward_probabilities.json")
    # Bare pm_brief (no .md) and any other <name>.json must also be scrubbed.
    assert "pm_brief" not in scrub("per pm_brief peer table the margin is 40.6%")
    assert "pm_brief" not in scrub("reinforcing the pm_brief's framing")
    assert "news.json" not in scrub("sourced from news.json items")
    assert ".json" not in scrub("see insider.json and social.json")
    # intrinsic-value artifact (both raw/ and bare forms) → friendly name, no leak
    assert "intrinsic_value" not in scrub("per raw/intrinsic_value.json the IV base is $X")
    assert "intrinsic_value" not in scrub("the intrinsic_value.json fair value")


def test_clean_agentic_vocabulary_scrubs_internal_qc_rule_refs():
    from cli.research_pdf import _clean_agentic_vocabulary as scrub

    out = scrub("This violates Rule 16(c) and QC checklist Item 16a per the deterministic peer-ratios block.")
    assert "Rule 16" not in out
    assert "Item 16" not in out
    assert "deterministic peer-ratios block" not in out


def test_appendix_sections_are_polished_not_verbatim(tmp_path, monkeypatch):
    """Appendix renders must run the same leak-scrub as front-of-document.
    We assert this at the render-helper level: the build wires appendix
    sections through the polished renderer."""
    import cli.research_pdf as pdf
    import inspect

    src = inspect.getsource(pdf.build_research_pdf)
    # Appendix sections must NOT use the raw (verbatim) render_md — they must
    # use the polished renderer that strips directives + scrubs vocabulary.
    for section in (
        "debate_risk.md", "debate_bull_bear.md", "analyst_market.md",
        "analyst_fundamentals.md", "analyst_news.md", "analyst_social.md",
    ):
        assert f'render_md_polished("{section}")' in src, (
            f"appendix section {section} still rendered verbatim (leaks scaffolding)"
        )
    # decision.md remains available as the Investment Recommendation fallback
    # source (older runs without decision_executive.md); always polished.
    assert 'render_md_polished("decision.md")' in src


def test_pm_brief_and_working_notes_dumps_removed_from_customer_pdf():
    """2026-06-02 full-scrub policy: the raw PM setup-brief ('Investment
    Thesis') and the unfiltered 'PM Working Notes' appendix are the two
    largest process-narration / instruction leak surfaces (the MSFT
    2026-05-29 audit found 'Interpretation rules for analysts', 'cited
    verbatim from raw cells' there). Troubleshooting uses the raw/ run-dir
    files, not the PDF, so these dumps are dropped from the customer PDF
    wholesale rather than phrase-scrubbed."""
    import cli.research_pdf as pdf
    import inspect

    template = pdf._HTML_TEMPLATE
    assert "Investment Thesis" not in template, "raw pm_brief dump still in PDF"
    assert "{pm_brief_html}" not in template
    assert "PM Working Notes" not in template, "unfiltered working-notes dump still in PDF"
    assert "{decision_working_notes_html}" not in template

    src = inspect.getsource(pdf.build_research_pdf)
    assert "pm_brief_html" not in src, "build still assembles the pm_brief dump"
    assert "decision_working_notes_html" not in src


def test_strip_llm_directives_removes_interpretation_rules_leadin():
    """MSFT 2026-05-29 p31 leak: analyst notes restate the setup brief's
    'authoritative interpretation rules for this report' with a verbatim
    blockquote of the agent-facing rules. Strip the lead-in + blockquote;
    keep the substantive analysis that follows."""
    from cli.research_pdf import _strip_llm_directives as strip

    text = (
        "## Business-model framing\n\n"
        "Per `pm_brief.md`, the authoritative interpretation rules for this report are:\n\n"
        "> *\"Treat Azure constant-currency growth as the single most important KPI; "
        "capex is a margin headwind.\"*\n\n"
        "The three-segment structure is the analytical frame: Intelligent Cloud is the "
        "growth engine. Every interpretation that follows applies these rules.\n"
    )
    out = strip(text)
    assert "interpretation rules for this report" not in out
    assert "Treat Azure constant-currency growth" not in out, "agent-rule blockquote leaked"
    assert "Every interpretation that follows applies these rules" not in out
    # Substantive analysis survives.
    assert "Intelligent Cloud is the" in out
    assert "## Business-model framing" in out


def test_clean_agentic_vocabulary_neutralises_provenance_narration():
    """Process-narration phrases ('cited verbatim from raw cells', 'quoted
    verbatim from the peer dataset') must not reach the customer PDF. They
    are neutralised to plain, grammatical provenance language rather than
    deleted mid-sentence."""
    from cli.research_pdf import _clean_agentic_vocabulary as scrub

    out = scrub("Both numbers are cited verbatim from raw cells; they agree.")
    assert "verbatim from raw cells" not in out
    assert "cited verbatim" not in out

    out2 = scrub("Peer ratios quoted verbatim from the peer dataset: GOOGL 28x.")
    assert "verbatim" not in out2
    assert "GOOGL 28x" in out2  # the data survives


def test_full_scrub_v2_kills_all_interpretation_rule_leadins():
    """2026-06-02 batch: six distinct lead-in phrasings restate the setup
    brief's interpretation rules with a verbatim quoted block. All must be
    stripped (lead-in + the quoted rules), regardless of phrasing or whether
    the rules follow as a blockquote, bullets, or numbered list."""
    from cli.research_pdf import _strip_llm_directives as strip

    leadins = [
        'The following rules are quoted verbatim from the pm_brief.md "Interpretation rules for analysts" and govern every numerical interpretation in this report:',
        "The PM Pre-flight Brief specifies the following interpretation rules, quoted verbatim:",
        "Per pm_brief.md interpretation rules for analysts (quoted verbatim):",
        "From pm_brief.md *Interpretation rules for analysts* (verbatim):",
        "Quoting the interpretation rules for analysts verbatim from pm_brief.md:",
        "Quoting the Interpretation rules for analysts from pm_brief.md verbatim:",
    ]
    blocks = [
        '\n\n> *"Rule one. Rule two."*\n\nKeep this analysis.',
        '\n- Treat capex as a demand signal\n- Net cash is the floor\n\nKeep this analysis.',
        '\n1. Azure is the KPI\n2. Capex is a headwind\n\nKeep this analysis.',
    ]
    for li in leadins:
        for blk in blocks:
            out = strip(li + blk)
            assert "interpretation rules" not in out.lower(), f"leaked: {li!r}"
            assert "Rule one" not in out and "Azure is the KPI" not in out and "Treat capex" not in out, f"rules leaked under {li!r}"
            assert "Keep this analysis." in out


def test_full_scrub_v2_strips_provenance_footnote_and_neutralises_pm_brief():
    """The ONDS-style trailing italic sourcing footnote (anti-hallucination
    narration + raw-file list) must be removed; scattered pm_brief / PM
    Pre-flight Brief references must read as neutral methodology language,
    never as an internal document."""
    from cli.research_pdf import _strip_llm_directives, _clean_agentic_vocabulary

    footnote = (
        "Headline analysis here.\n\n"
        "*All numerical claims in this report trace to the following sources: "
        "financials.json (income statement); raw/pm_brief.md (peer ratios table, "
        "interpretation rules); raw/reference.json (reference price $9.06). "
        "No figures were invented or sourced from memory.*\n"
    )
    out = _clean_agentic_vocabulary(_strip_llm_directives(footnote))
    assert "trace to the following sources" not in out
    assert "No figures were invented" not in out
    assert "interpretation rules" not in out.lower()
    assert "Headline analysis here." in out

    for src in (
        "Per pm_brief.md, NII on float is a key driver.",
        "per the pm_brief framing, margin compresses.",
        "The PM Pre-flight Brief specifies the peer set.",
    ):
        scrubbed = _clean_agentic_vocabulary(_strip_llm_directives(src))
        assert "pm_brief" not in scrubbed
        assert "the setup brief" not in scrubbed
        assert "PM Pre-flight" not in scrubbed


def test_residual_raw_paths_fully_scrubbed():
    from cli.research_pdf import _clean_agentic_vocabulary as scrub
    # no-extension, unmapped-name, and trailing forms all leave no "raw/"
    assert "raw/" not in scrub("intrinsic value model (raw/intrinsic_value DCF)")
    assert "raw/" not in scrub("see raw/something_unmapped and raw/peers.json")
