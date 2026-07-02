"""Full debt maturity ladder from the 10-K debt note (FREE_FINISH_GOAL Phase 1).

Today the pipeline ships only a refinancing *proxy* (current-vs-long-term debt
split). The real year-by-year maturity ladder is free — it's in the 10-K
long-term-debt note as text. These tests cover:

- debt-maturity targeted excerpts from the FULL filing text (past truncation),
- `fetch_debt_maturity_note`: 10-K-only fallback fetch for when the latest
  filing is a 10-Q (the ladder lives only in the annual debt note),
- the `## Debt maturity ladder` pm_brief block (verbatim quote or honest n/a),
- the refinancing-proxy block upgrade (points at the full ladder when present),
- PM Pre-flight wiring (raw/debt_maturity.json + block, fail-open),
- the Risk & Red-Flags role directive.
"""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


# Realistic phrasings from real 10-K debt notes (stripped-HTML style).
_ORCL_STYLE = (
    "filler " * 500
    + "Future principal payments (adjusted for the effects of the fair value "
      "hedge accounting) for all of our borrowings at May 31, 2026 were as "
      "follows: (in millions) fiscal 2027 $ 8,750 fiscal 2028 $ 10,250 fiscal "
      "2029 $ 9,000 fiscal 2030 $ 7,171 fiscal 2031 $ 6,500 thereafter $ "
      "62,600 total $ 104,271 "
    + "tail " * 500
)

_MSFT_STYLE = (
    "filler " * 500
    + "Maturities of long-term debt for each of the next five years and "
      "thereafter are as follows: (In millions) Year Ending June 30, 2027 $ "
      "2,250 2028 $ 5,250 2029 $ 3,000 2030 $ 4,250 2031 $ 1,000 thereafter "
      "$ 24,225 Total $ 39,975 "
    + "tail " * 500
)


# --- keyword coverage + pure extraction -----------------------------------

def test_debt_keywords_cover_common_10k_phrasings():
    from tradingagents.agents.utils.sec_edgar import DEBT_MATURITY_EXCERPT_KEYWORDS
    joined = " ".join(DEBT_MATURITY_EXCERPT_KEYWORDS)
    assert "maturities of long-term debt" in joined
    assert "aggregate maturities" in joined
    assert "future principal payments" in joined


def test_extract_debt_maturity_excerpts_orcl_style():
    from tradingagents.agents.utils.sec_edgar import extract_debt_maturity_excerpts
    hits = extract_debt_maturity_excerpts(_ORCL_STYLE)
    assert hits, "ORCL-style 'future principal payments' note not excerpted"
    text = " ".join(h["text"] for h in hits)
    assert "fiscal 2027 $ 8,750" in text
    assert "thereafter $ 62,600" in text


def test_extract_debt_maturity_excerpts_msft_style():
    from tradingagents.agents.utils.sec_edgar import extract_debt_maturity_excerpts
    hits = extract_debt_maturity_excerpts(_MSFT_STYLE)
    assert hits, "MSFT-style 'maturities of long-term debt' note not excerpted"
    text = " ".join(h["text"] for h in hits)
    assert "2027 $ 2,250" in text and "thereafter $ 24,225" in text


def test_extract_debt_maturity_excerpts_msft_2025_phrasing():
    """MSFT's FY25 10-K says 'maturities of OUR long-term debt' (live probe
    2026-07-02) — the possessive must match, while 'maturities of our debt
    investments' (the investment portfolio, NOT borrowings) must not be the
    only path to a hit."""
    from tradingagents.agents.utils.sec_edgar import extract_debt_maturity_excerpts
    text = (
        "filler " * 500
        + "The following table outlines maturities of our long-term debt, "
          "including the current portion, as of June 30, 2025: (In millions) "
          "Year Ending June 30, 2026 $ 3,000 2027 9,250 2028 0 2029 2,054 "
          "2030 0 Thereafter 34,902 Total $ 49,206 "
        + "tail " * 500
    )
    hits = extract_debt_maturity_excerpts(text)
    assert hits, "possessive 'maturities of our long-term debt' not excerpted"
    joined = " ".join(h["text"] for h in hits)
    assert "2027 9,250" in joined and "Thereafter 34,902" in joined


def test_extract_debt_maturity_excerpts_empty_when_no_note():
    from tradingagents.agents.utils.sec_edgar import extract_debt_maturity_excerpts
    assert extract_debt_maturity_excerpts("no debt talk here " * 100) == []
    assert extract_debt_maturity_excerpts("") == []


# --- fetch_latest_filing attaches debt excerpts (past truncation) ----------

def test_fetch_latest_filing_attaches_debt_maturity_excerpts(monkeypatch):
    """The ladder must survive the 60K truncation: computed from FULL text."""
    from tradingagents.agents.utils import sec_edgar
    from tests.test_sec_edgar import _stub_submissions_json

    submissions = _stub_submissions_json([
        {"form": "10-K", "filing_date": "2026-06-22", "accession": "0001-26-1",
         "primary_doc": "orcl.htm"},
    ])
    filler = "<p>" + "filler " * 30_000 + "</p>"
    tail = ("<p>Future principal payments for all of our borrowings at May 31, "
            "2026 were as follows: fiscal 2027 $ 8,750 fiscal 2028 $ 10,250 "
            "thereafter $ 62,600</p>")
    html = ("<html><body>" + filler + tail + "</body></html>").encode("utf-8")

    monkeypatch.setattr(sec_edgar, "_http_get",
                        lambda url, timeout=30: submissions if "submissions" in url else html)
    monkeypatch.setattr(sec_edgar, "_resolve_cik", lambda t: 1341439)
    filing = sec_edgar.fetch_latest_filing("ORCL", "2026-07-01", max_text_chars=5_000)
    assert filing["content_truncated"] is True
    assert "8,750" not in filing["content"]  # past the cut
    dm = filing["debt_maturity_excerpts"]
    assert dm and any("fiscal 2027 $ 8,750" in e["text"] for e in dm)


# --- fetch_debt_maturity_note (10-K-only fallback fetch) --------------------

def test_fetch_debt_maturity_note_happy_path(monkeypatch):
    """Latest filing is a 10-Q; the note fetch must pick the latest 10-K
    on/before trade_date and return its debt excerpts."""
    from tradingagents.agents.utils import sec_edgar
    from tests.test_sec_edgar import _stub_submissions_json

    submissions = _stub_submissions_json([
        {"form": "10-Q", "filing_date": "2026-04-29", "accession": "0001-26-2",
         "primary_doc": "msft-q3.htm"},
        {"form": "10-K", "filing_date": "2025-07-30", "accession": "0001-25-9",
         "primary_doc": "msft-10k.htm"},
    ])
    html = ("<html><body><p>" + _MSFT_STYLE + "</p></body></html>").encode("utf-8")

    fetched = []
    def fake_get(url, timeout=30):
        fetched.append(url)
        return submissions if "submissions" in url else html
    monkeypatch.setattr(sec_edgar, "_http_get", fake_get)
    monkeypatch.setattr(sec_edgar, "_resolve_cik", lambda t: 789019)

    note = sec_edgar.fetch_debt_maturity_note("MSFT", "2026-05-01")
    assert note.get("unavailable") is not True
    assert note["form"] == "10-K"
    assert note["filing_date"] == "2025-07-30"
    assert any("msft-10k.htm" in u for u in fetched)
    assert any("2027 $ 2,250" in e["text"] for e in note["excerpts"])


def test_fetch_debt_maturity_note_no_10k(monkeypatch):
    from tradingagents.agents.utils import sec_edgar
    from tests.test_sec_edgar import _stub_submissions_json

    submissions = _stub_submissions_json([
        {"form": "10-Q", "filing_date": "2026-04-29", "accession": "0001-26-2",
         "primary_doc": "q.htm"},
    ])
    monkeypatch.setattr(sec_edgar, "_http_get",
                        lambda url, timeout=30: submissions if "submissions" in url else b"<html></html>")
    monkeypatch.setattr(sec_edgar, "_resolve_cik", lambda t: 123)
    note = sec_edgar.fetch_debt_maturity_note("XYZ", "2026-05-01")
    assert note["unavailable"] is True
    assert "10-k" in note["reason"].lower()


def test_fetch_debt_maturity_note_network_failure(monkeypatch):
    from tradingagents.agents.utils import sec_edgar
    monkeypatch.setattr(sec_edgar, "_http_get", lambda url, timeout=30: None)
    note = sec_edgar.fetch_debt_maturity_note("MSFT", "2026-05-01")
    assert note["unavailable"] is True


def test_fetch_debt_maturity_note_10k_without_ladder_keywords(monkeypatch):
    """A 10-K whose text has no maturity phrasing -> excerpts [], NOT unavailable
    (the block renders the honest 'not located' n/a)."""
    from tradingagents.agents.utils import sec_edgar
    from tests.test_sec_edgar import _stub_submissions_json

    submissions = _stub_submissions_json([
        {"form": "10-K", "filing_date": "2025-07-30", "accession": "0001-25-9",
         "primary_doc": "k.htm"},
    ])
    html = b"<html><body><p>no debt note phrasing here at all</p></body></html>"
    monkeypatch.setattr(sec_edgar, "_http_get",
                        lambda url, timeout=30: submissions if "submissions" in url else html)
    monkeypatch.setattr(sec_edgar, "_resolve_cik", lambda t: 789019)
    note = sec_edgar.fetch_debt_maturity_note("MSFT", "2026-05-01")
    assert note.get("unavailable") is not True
    assert note["excerpts"] == []


# --- pm_brief block formatting ----------------------------------------------

def _happy_note():
    return {
        "ticker": "ORCL", "form": "10-K", "filing_date": "2026-06-22",
        "accession_number": "0001-26-1",
        "url": "https://www.sec.gov/Archives/edgar/data/1341439/000126/orcl.htm",
        "excerpts": [{"keyword": "future principal payments",
                      "text": ("Future principal payments for all of our borrowings at "
                               "May 31, 2026 were as follows: fiscal 2027 $ 8,750 fiscal "
                               "2028 $ 10,250 thereafter $ 62,600")}],
        "source": "sec.gov",
    }


def test_format_debt_maturity_block_happy():
    from tradingagents.agents.utils.sec_edgar import format_debt_maturity_block
    block = format_debt_maturity_block(_happy_note())
    assert "## Debt maturity ladder" in block
    assert "10-K filed 2026-06-22" in block
    assert "fiscal 2027 $ 8,750" in block          # verbatim excerpt carried
    assert "verbatim" in block.lower()             # quote-verbatim directive
    assert "never" in block.lower()                # do-not-fabricate directive


def test_format_debt_maturity_block_na_unavailable():
    from tradingagents.agents.utils.sec_edgar import format_debt_maturity_block
    block = format_debt_maturity_block(
        {"unavailable": True, "reason": "no 10-K on/before trade_date", "ticker": "XYZ"})
    assert "## Debt maturity ladder" in block and "n/a" in block
    assert "full ladder not disclosed" in block.lower()
    assert "proxy" in block.lower()  # points the reader at the refi proxy


def test_format_debt_maturity_block_na_when_no_excerpts():
    from tradingagents.agents.utils.sec_edgar import format_debt_maturity_block
    note = _happy_note()
    note["excerpts"] = []
    block = format_debt_maturity_block(note)
    assert "n/a" in block
    assert "full ladder not disclosed" in block.lower()


def test_format_debt_maturity_block_none_input():
    from tradingagents.agents.utils.sec_edgar import format_debt_maturity_block
    assert "n/a" in format_debt_maturity_block(None)


