"""Targeted keyword excerpts from the FULL filing text (pro-deck technique B).

The 2026-07-02 ORCL benchmark caught `fetch_latest_filing`'s 60K-char
truncation silently dropping ALL MD&A prose on large inline-XBRL 10-Ks (the
header metadata alone exceeds 60K chars), so RPO / capex / segment paragraphs
never reached sec_filing.md. The fix: search the full stripped text for a
fixed keyword set and carry windowed excerpts past the truncation.
"""
import pytest

from tradingagents.agents.utils.sec_edgar import (
    DEFAULT_EXCERPT_KEYWORDS,
    extract_keyword_excerpts,
)

pytestmark = pytest.mark.unit


def test_default_keywords_cover_pro_deck_topics():
    joined = " ".join(DEFAULT_EXCERPT_KEYWORDS)
    assert "remaining performance obligation" in joined
    assert "capital expenditure" in joined
    assert "prepayment" in joined
    assert "segment" in joined


def test_extracts_window_around_match_case_insensitive():
    text = "x" * 50_000 + " Remaining Performance Obligations were $638 billion. " + "y" * 50_000
    hits = extract_keyword_excerpts(text, keywords=("remaining performance obligation",),
                                    before=30, after=80)
    assert len(hits) == 1
    assert "638 billion" in hits[0]["text"]
    assert hits[0]["keyword"] == "remaining performance obligation"
    # window respects bounds
    assert len(hits[0]["text"]) <= 30 + len("remaining performance obligation") + 80


def test_no_match_returns_empty():
    assert extract_keyword_excerpts("nothing to see here", keywords=("backlog",)) == []
    assert extract_keyword_excerpts("", keywords=("backlog",)) == []


def test_close_matches_collapse_into_one_window():
    # two mentions 50 chars apart -> one excerpt, not two overlapping ones
    text = ("A" * 10_000 + "capital expenditure plans... more capital expenditure talk"
            + "B" * 10_000)
    hits = extract_keyword_excerpts(text, keywords=("capital expenditure",),
                                    before=100, after=200, min_gap=500)
    assert len(hits) == 1


def test_max_per_keyword_and_total_budget():
    blob = ("Z" * 3000).join(["reportable segment info %d" % i for i in range(10)])
    hits = extract_keyword_excerpts(blob, keywords=("reportable segment",),
                                    before=50, after=100, min_gap=200, max_per_keyword=3)
    assert len(hits) == 3
    tiny = extract_keyword_excerpts(blob, keywords=("reportable segment",),
                                    before=50, after=100, min_gap=200,
                                    max_per_keyword=10, max_total_chars=400)
    assert sum(len(h["text"]) for h in tiny) <= 400 + 200  # one window may straddle


def test_fetch_latest_filing_attaches_excerpts_when_truncated(monkeypatch):
    """Truncated filing -> excerpts computed from the FULL text (content that
    was cut off is still searchable)."""
    from tradingagents.agents.utils import sec_edgar
    from tests.test_sec_edgar import _stub_submissions_json

    submissions = _stub_submissions_json([
        {"form": "10-K", "filing_date": "2026-06-22", "accession": "0001-26-1",
         "primary_doc": "orcl.htm"},
    ])
    filler = "<p>" + "filler " * 30_000 + "</p>"
    tail = "<p>Remaining performance obligations were $638 billion, of which 12% next twelve months.</p>"
    html = ("<html><body>" + filler + tail + "</body></html>").encode("utf-8")

    def fake_get(url, timeout=30):
        return submissions if "submissions" in url else html

    monkeypatch.setattr(sec_edgar, "_http_get", fake_get)
    monkeypatch.setattr(sec_edgar, "_resolve_cik", lambda t: 1341439)
    filing = sec_edgar.fetch_latest_filing("ORCL", "2026-07-01", max_text_chars=5_000)
    assert filing["content_truncated"] is True
    assert "638 billion" not in filing["content"]
    kws = [e["keyword"] for e in filing["excerpts"]]
    assert "remaining performance obligation" in kws


def test_format_for_prompt_renders_excerpts_section():
    from tradingagents.agents.utils import sec_edgar
    filing = {
        "ticker": "ORCL", "form": "10-K", "filing_date": "2026-06-22",
        "accession_number": "0001-26-1", "primary_document": "x.htm",
        "url": "https://example.com/x.htm",
        "content": "short content", "content_truncated": True,
        "excerpts": [{"keyword": "remaining performance obligation",
                      "text": "RPO were $638 billion, 12% next twelve months."}],
        "source": "sec.gov",
    }
    block = sec_edgar.format_for_prompt(filing)
    assert "Targeted excerpts" in block
    assert "$638 billion" in block
    assert "remaining performance obligation" in block


def test_format_for_prompt_no_excerpts_section_when_absent():
    from tradingagents.agents.utils import sec_edgar
    filing = {
        "ticker": "ORCL", "form": "10-K", "filing_date": "2026-06-22",
        "accession_number": "0001-26-1", "primary_document": "x.htm",
        "url": "https://example.com/x.htm",
        "content": "short content", "content_truncated": False,
        "source": "sec.gov",
    }
    assert "Targeted excerpts" not in sec_edgar.format_for_prompt(filing)
