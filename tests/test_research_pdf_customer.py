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
    # PM working notes (decision.md as appendix) must also be polished.
    assert 'render_md_polished("decision.md")' in src
