"""Tests for tradingagents.agents.utils.sec_edgar (Phase-6.3 SEC EDGAR fetcher)."""
import json

import pytest

pytestmark = pytest.mark.unit


def _stub_submissions_json(filings: list[dict]) -> bytes:
    """Build a minimal submissions JSON in EDGAR's recent-filings format."""
    return json.dumps({
        "name": "Test Co",
        "filings": {
            "recent": {
                "form": [f["form"] for f in filings],
                "filingDate": [f["filing_date"] for f in filings],
                "accessionNumber": [f["accession"] for f in filings],
                "primaryDocument": [f["primary_doc"] for f in filings],
            },
        },
    }).encode("utf-8")


def test_fetch_latest_filing_happy_path(monkeypatch):
    """Happy path: ticker maps to CIK, EDGAR returns a 10-Q filed before trade_date,
    HTML is fetched and stripped to plain text."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "10-Q", "filing_date": "2026-04-29", "accession": "0001193125-26-191507", "primary_doc": "msft-20260331.htm"},
        {"form": "10-Q", "filing_date": "2026-01-28", "accession": "0001193125-26-027207", "primary_doc": "msft-20251231.htm"},
    ])
    filing_html = b"<html><body><h1>Q3 FY26</h1><p>Azure and other cloud services revenue increased 40%.</p><script>nope</script></body></html>"

    calls = []
    def fake_http_get(url, timeout=30):
        calls.append(url)
        if "submissions/CIK" in url:
            return submissions
        if "Archives/edgar/data" in url:
            return filing_html
        return None
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    result = sec_edgar.fetch_latest_filing("MSFT", "2026-05-01")
    assert result.get("unavailable") is not True
    assert result["ticker"] == "MSFT"
    assert result["form"] == "10-Q"
    assert result["filing_date"] == "2026-04-29"
    assert result["accession_number"] == "0001193125-26-191507"
    assert "Azure and other cloud services revenue increased 40%" in result["content"]
    assert "nope" not in result["content"]  # script tags stripped
    assert "789019" in calls[0]  # MSFT CIK
    assert "msft-20260331.htm" in calls[1]


def test_fetch_latest_filing_skips_filings_after_trade_date(monkeypatch):
    """A 10-Q filed AFTER the trade date must not be returned (look-ahead bias)."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "10-Q", "filing_date": "2026-04-29", "accession": "0001193125-26-191507", "primary_doc": "msft-20260331.htm"},
    ])

    def fake_http_get(url, timeout=30):
        if "submissions/CIK" in url:
            return submissions
        return b"<html>doc</html>"
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    # Trade date 2026-04-28 is BEFORE the filing date — should return unavailable.
    result = sec_edgar.fetch_latest_filing("MSFT", "2026-04-28")
    assert result["unavailable"] is True
    assert "no 10-q or 10-k" in result["reason"].lower()


def test_fetch_latest_filing_unknown_ticker(monkeypatch):
    """Ticker not in cache and not found in EDGAR directory → unavailable."""
    from tradingagents.agents.utils import sec_edgar

    def fake_http_get(url, timeout=30):
        if "company_tickers.json" in url:
            return json.dumps({"0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple"}}).encode("utf-8")
        return None
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    result = sec_edgar.fetch_latest_filing("NOPENOPE", "2026-05-01")
    assert result["unavailable"] is True
    assert "CIK not found" in result["reason"]


def test_fetch_latest_filing_network_failure(monkeypatch):
    """Network failure on EDGAR → graceful unavailable, never raises."""
    from tradingagents.agents.utils import sec_edgar

    monkeypatch.setattr(sec_edgar, "_http_get", lambda url, timeout=30: None)

    result = sec_edgar.fetch_latest_filing("MSFT", "2026-05-01")
    assert result["unavailable"] is True
    assert "unreachable" in result["reason"].lower()


def test_fetch_latest_filing_invalid_trade_date():
    """Invalid trade date string → unavailable, no network call."""
    from tradingagents.agents.utils import sec_edgar
    result = sec_edgar.fetch_latest_filing("MSFT", "not-a-date")
    assert result["unavailable"] is True
    assert "invalid trade_date" in result["reason"]


def test_fetch_latest_filing_truncates_long_content(monkeypatch):
    """Content longer than max_text_chars must be truncated and flagged."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "10-K", "filing_date": "2025-07-30", "accession": "0000950170-25-100235", "primary_doc": "msft-20250630.htm"},
    ])
    long_html = b"<html><body>" + (b"X" * 200_000) + b"</body></html>"

    def fake_http_get(url, timeout=30):
        return submissions if "submissions" in url else long_html
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    result = sec_edgar.fetch_latest_filing("MSFT", "2026-05-01", max_text_chars=10_000)
    assert result.get("unavailable") is not True
    assert len(result["content"]) <= 10_000
    assert result["content_truncated"] is True


def test_format_for_prompt_emits_block_with_temporal_anchor():
    """The Markdown block must contain the form/date and the 'treat as known
    data' instruction so downstream agents can never write 'pending'."""
    from tradingagents.agents.utils import sec_edgar
    filing = {
        "ticker": "MSFT",
        "form": "10-Q",
        "filing_date": "2026-04-29",
        "accession_number": "0001193125-26-191507",
        "primary_document": "msft-20260331.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm",
        "content": "Azure and other cloud services revenue increased 40%.",
        "content_truncated": False,
        "source": "sec.gov",
    }
    block = sec_edgar.format_for_prompt(filing)
    assert "MSFT 10-Q filed 2026-04-29" in block
    assert "treat them as known data" in block
    assert "NEVER as 'pending adjudication'" in block
    assert "Azure and other cloud services revenue increased 40%" in block


def test_format_for_prompt_returns_empty_for_unavailable():
    """Unavailable filing → empty string (no markdown emitted, downstream
    agents fall back to LLM judgment without ground truth)."""
    from tradingagents.agents.utils import sec_edgar
    assert sec_edgar.format_for_prompt({"unavailable": True, "reason": "x", "ticker": "MSFT"}) == ""


def test_fetch_latest_filing_skips_filing_with_missing_accession(monkeypatch):
    """If EDGAR returns a malformed filing row (e.g. accessionNumber list is
    shorter than the other arrays), that row must be skipped via IndexError,
    not raised. The fetcher's never-raises contract must hold under
    malformed-but-not-empty submissions JSON."""
    from tradingagents.agents.utils import sec_edgar

    # Two filing rows. The first row (i=0, 2026-04-29) will have a valid
    # filingDate but accessionNumber has ZERO entries, so accessionNumber[0]
    # raises IndexError → row 0 is skipped. The second row (i=1, 2026-01-28)
    # also raises IndexError on accessionNumber[1] → both rows skipped →
    # function returns None → fetch_latest_filing returns unavailable.
    # This validates the never-raises contract: no KeyError/IndexError escapes.
    submissions_no_accessions = json.dumps({
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q"],
                "filingDate": ["2026-04-29", "2026-01-28"],
                "accessionNumber": [],           # both rows will IndexError here
                "primaryDocument": ["msft-20260331.htm", "msft-20251231.htm"],
            },
        },
    }).encode("utf-8")

    def fake_http_get(url, timeout=30):
        if "submissions/CIK" in url:
            return submissions_no_accessions
        return b"<html>doc</html>"
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    # Both rows are malformed (missing accessionNumber entries) → all skipped →
    # unavailable returned, but crucially no IndexError raised to the caller.
    result = sec_edgar.fetch_latest_filing("MSFT", "2026-05-01")
    assert result["unavailable"] is True
    # The reason must reflect missing filing, not an uncaught exception
    assert "no 10-q or 10-k" in result["reason"].lower()


def test_fetch_latest_filing_skips_malformed_row_continues_to_next(monkeypatch):
    """A single malformed row (bad filingDate) must be skipped and the next
    valid row returned — the loop must not abort on first exception."""
    from tradingagents.agents.utils import sec_edgar

    submissions = json.dumps({
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q"],
                # First row has a garbled date → ValueError → skipped.
                # Second row is fully valid.
                "filingDate": ["NOT-A-DATE", "2026-01-28"],
                "accessionNumber": ["0001193125-26-191507", "0001193125-26-027207"],
                "primaryDocument": ["msft-20260331.htm", "msft-20251231.htm"],
            },
        },
    }).encode("utf-8")

    def fake_http_get(url, timeout=30):
        if "submissions/CIK" in url:
            return submissions
        return b"<html>doc</html>"
    monkeypatch.setattr(sec_edgar, "_http_get", fake_http_get)

    # First row (i=0) skipped due to ValueError on bad date.
    # Second row (i=1, 2026-01-28) is valid and before trade_date 2026-05-01.
    result = sec_edgar.fetch_latest_filing("MSFT", "2026-05-01")
    assert result.get("unavailable") is not True
    assert result["filing_date"] == "2026-01-28"
    assert result["accession_number"] == "0001193125-26-027207"
