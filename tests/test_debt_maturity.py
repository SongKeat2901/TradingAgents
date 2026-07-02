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


# --- refinancing proxy block upgrade ----------------------------------------

def test_refinancing_block_points_at_ladder_when_available():
    from tradingagents.agents.utils.distress_screens import (
        compute_refinancing_pressure, format_refinancing_block,
    )
    r = compute_refinancing_pressure(
        {"total_debt": 100, "long_term_debt": 50, "cash_and_equivalents": 20})
    block = format_refinancing_block(r, ladder_available=True)
    assert "## Debt maturity ladder" in block  # cross-reference to the full ladder
    assert "not disclosed" not in block.lower()


def test_refinancing_block_default_keeps_proxy_caveat():
    from tradingagents.agents.utils.distress_screens import (
        compute_refinancing_pressure, format_refinancing_block,
    )
    r = compute_refinancing_pressure(
        {"total_debt": 100, "long_term_debt": 50, "cash_and_equivalents": 20})
    block = format_refinancing_block(r)  # backwards-compatible signature
    assert "NOT the full year-by-year maturity ladder" in block


def test_debt_ladder_available_helper():
    from tradingagents.agents.utils.distress_screens import debt_ladder_available
    assert debt_ladder_available(_happy_note()) is True
    assert debt_ladder_available({"unavailable": True, "reason": "x"}) is False
    note = _happy_note()
    note["excerpts"] = []
    assert debt_ladder_available(note) is False
    assert debt_ladder_available(None) is False


# --- PM Pre-flight wiring ----------------------------------------------------

@pytest.fixture
def _stub_network(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        lambda d, t: {"trade_date": d, "_unavailable": []},
    )
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: {"unavailable": True, "reason": "stubbed", "ticker": t},
    )
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_debt_maturity_note",
        lambda t, d: {"unavailable": True, "reason": "stubbed", "ticker": t},
        raising=False,
    )


def _run_preflight(tmp_path):
    from tests.test_pm_preflight import _VALID_BRIEF
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)
    node = create_pm_preflight_node(fake_llm)
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-01",
             "raw_dir": str(tmp_path / "raw")}
    return node(state)


def test_pm_preflight_uses_10k_filing_excerpts_without_refetch(tmp_path, monkeypatch, _stub_network):
    """Latest filing IS a 10-K carrying debt excerpts -> block quotes them and
    fetch_debt_maturity_note is NOT called (no duplicate EDGAR fetch)."""
    filing = {
        "ticker": "MSFT", "form": "10-K", "filing_date": "2026-06-22",
        "accession_number": "0001-26-1", "primary_document": "k.htm",
        "url": "https://example.com/k.htm", "content": "text",
        "content_truncated": True, "excerpts": [],
        "debt_maturity_excerpts": _happy_note()["excerpts"],
        "source": "sec.gov",
    }
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: filing)
    called = []
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_debt_maturity_note",
        lambda t, d: called.append(t))
    _run_preflight(tmp_path)
    assert called == []
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Debt maturity ladder" in brief
    assert "fiscal 2027 $ 8,750" in brief
    dm = json.loads((tmp_path / "raw" / "debt_maturity.json").read_text(encoding="utf-8"))
    assert dm["form"] == "10-K"


def test_pm_preflight_uses_10q_inline_excerpts_when_present(tmp_path, monkeypatch, _stub_network):
    """Some issuers (MSFT) repeat the maturities table in the 10-Q's debt note.
    Inline excerpts from the LATEST filing are fresher than last year's 10-K —
    use them and skip the fallback fetch; the source line stays honest (10-Q)."""
    filing = {
        "ticker": "MSFT", "form": "10-Q", "filing_date": "2026-04-29",
        "accession_number": "0001-26-2", "primary_document": "q.htm",
        "url": "https://example.com/q.htm", "content": "text",
        "content_truncated": True, "excerpts": [],
        "debt_maturity_excerpts": [{"keyword": "maturities of our long-term debt",
                                    "text": "Year Ending June 30, 2026 $ 3,000 2027 9,250"}],
        "source": "sec.gov",
    }
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: filing)
    called = []
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_debt_maturity_note",
        lambda t, d: called.append(t))
    _run_preflight(tmp_path)
    assert called == []
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Debt maturity ladder" in brief
    assert "2027 9,250" in brief
    assert "10-Q filed 2026-04-29" in brief  # source stays honest
    dm = json.loads((tmp_path / "raw" / "debt_maturity.json").read_text(encoding="utf-8"))
    assert dm["form"] == "10-Q"


def test_pm_preflight_falls_back_to_10k_note_when_latest_is_10q(tmp_path, monkeypatch, _stub_network):
    filing = {
        "ticker": "MSFT", "form": "10-Q", "filing_date": "2026-04-29",
        "accession_number": "0001-26-2", "primary_document": "q.htm",
        "url": "https://example.com/q.htm", "content": "text",
        "content_truncated": False, "excerpts": [], "source": "sec.gov",
    }
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: filing)
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_debt_maturity_note",
        lambda t, d: _happy_note())
    _run_preflight(tmp_path)
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Debt maturity ladder" in brief
    assert "fiscal 2027 $ 8,750" in brief
    dm = json.loads((tmp_path / "raw" / "debt_maturity.json").read_text(encoding="utf-8"))
    assert dm["filing_date"] == "2026-06-22"


def test_pm_preflight_debt_note_na_block_when_unavailable(tmp_path, _stub_network):
    """Both the filing and the note fetch are stubbed unavailable -> the block
    still lands, as an honest n/a (so the Risk role's fallback directive has an
    anchor), and debt_maturity.json records the reason."""
    _run_preflight(tmp_path)
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Debt maturity ladder" in brief
    assert "full ladder not disclosed" in brief.lower()
    dm = json.loads((tmp_path / "raw" / "debt_maturity.json").read_text(encoding="utf-8"))
    assert dm["unavailable"] is True


def test_pm_preflight_debt_note_exception_is_fail_open(tmp_path, monkeypatch, _stub_network):
    def _raises(t, d):
        raise RuntimeError("simulated EDGAR crash")
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_debt_maturity_note", _raises)
    out = _run_preflight(tmp_path)  # MUST NOT raise
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Debt maturity ladder" in brief and "n/a" in brief
    assert out["pm_brief"] == brief


# --- Risk & Red-Flags role directive ----------------------------------------

def test_risk_role_has_debt_ladder_directive():
    from tradingagents.agents.analysts import fundamentals_roles as fr
    s = fr._SYSTEM_RISK.lower()
    assert "## debt maturity ladder" in s
    assert "verbatim" in s
    assert "full ladder not disclosed" in s


def test_risk_role_requires_debt_ladder_section():
    from tradingagents.agents.analysts import fundamentals_roles as fr
    assert "## Debt maturity ladder" in fr._REQUIRED_RISK
