"""Tests for the SEC 8-K earnings press release fetcher (Exhibit 99.x).

EARNINGS_RELEASE_GOAL.md: the pro-deck gap analysis marked capex-funding split
and forward guidance "call-only", but both live in the free 8-K earnings press
release (item 2.02, Ex-99.1). These tests cover the doc picker, the fetch
(happy + every fail-soft path), the release excerpt keywords, and the pm_brief
block formatter.
"""
import json

import pytest

pytestmark = pytest.mark.unit


def _stub_submissions_json(filings: list[dict]) -> bytes:
    """Minimal submissions JSON in EDGAR's recent-filings format, with 8-K items."""
    return json.dumps({
        "name": "Test Co",
        "filings": {
            "recent": {
                "form": [f["form"] for f in filings],
                "filingDate": [f["filing_date"] for f in filings],
                "accessionNumber": [f["accession"] for f in filings],
                "primaryDocument": [f["primary_doc"] for f in filings],
                "items": [f.get("items", "") for f in filings],
            },
        },
    }).encode("utf-8")


def _stub_index_json(names: list[str]) -> bytes:
    """Minimal EDGAR filing-directory index.json."""
    return json.dumps({
        "directory": {"item": [{"name": n} for n in names], "name": "/Archives/edgar/data/x"},
    }).encode("utf-8")


# Filler between key sentences so the excerpt windows (before=800/after=1400)
# don't overlap — mirroring a real release, where these topics sit thousands
# of chars apart across a 40-60K-char document.
_FILLER = b"<p>" + b"segment detail and reconciliation tables. " * 80 + b"</p>"

_EXHIBIT_HTML = (
    b"<html><body>"
    b"<p>Oracle Announces Fiscal 2026 Fourth Quarter Results</p>"
    b"<p>Total Revenues expected to grow 27% to 29% in Q1 FY27.</p>"
    + _FILLER +
    b"<p>The company announced a $40 billion debt and equity financing plan, "
    b"including a $20 billion at-the-market equity program.</p>"
    + _FILLER +
    b"<p>Remaining performance obligations grew to $638 billion.</p>"
    + _FILLER +
    b"<p>&#8220;Cloud demand continues to outstrip supply,&#8221; said Safra Catz, "
    b"Chief Executive Officer.</p>"
    b"<script>not_text</script>"
    b"</body></html>"
)


# ---------------------------------------------------------------- doc picker

def test_pick_ex99_prefers_99_1_htm():
    from tradingagents.agents.utils.sec_edgar import _pick_ex99_document
    names = ["orcl-8k_20260610.htm", "orcl-ex99_2.htm", "orcl-ex99_1.htm",
             "logo.jpg", "index.json", "R1.htm"]
    assert _pick_ex99_document(names) == "orcl-ex99_1.htm"


def test_pick_ex99_accepts_dash_and_991_variants():
    from tradingagents.agents.utils.sec_edgar import _pick_ex99_document
    assert _pick_ex99_document(["a8-k.htm", "ex-99d1.htm"]) == "ex-99d1.htm"
    assert _pick_ex99_document(["a8-k.htm", "d12345dex991.htm"]) == "d12345dex991.htm"


def test_pick_ex99_falls_back_to_txt_and_none_when_absent():
    from tradingagents.agents.utils.sec_edgar import _pick_ex99_document
    assert _pick_ex99_document(["form8k.htm", "ex99-1.txt"]) == "ex99-1.txt"
    assert _pick_ex99_document(["form8k.htm", "graph.jpg"]) is None
    assert _pick_ex99_document([]) is None


# ---------------------------------------------------------------- fetch

def _fake_http(submissions: bytes, index: bytes | None, exhibit: bytes | None):
    calls = []

    def fake_http_get(url, timeout=30):
        calls.append(url)
        if "submissions/CIK" in url:
            return submissions
        if url.endswith("/index.json"):
            return index
        if "Archives/edgar/data" in url:
            return exhibit
        return None

    return fake_http_get, calls


def test_fetch_earnings_release_happy_path(monkeypatch):
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0001341439-26-000123",
         "primary_doc": "orcl-8k_20260610.htm", "items": "2.02,9.01"},
        {"form": "10-K", "filing_date": "2026-06-18", "accession": "0001341439-26-000200",
         "primary_doc": "orcl-10k.htm"},
    ])
    index = _stub_index_json(["orcl-8k_20260610.htm", "orcl-ex99_1.htm", "logo.jpg"])
    fake, calls = _fake_http(submissions, index, _EXHIBIT_HTML)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("ORCL", "2026-07-02")
    assert r.get("unavailable") is not True
    assert r["ticker"] == "ORCL"
    assert r["form"] == "8-K"
    assert r["filing_date"] == "2026-06-10"
    assert r["accession_number"] == "0001341439-26-000123"
    assert r["exhibit"] == "orcl-ex99_1.htm"
    assert r["url"].endswith("orcl-ex99_1.htm")
    assert "expected to grow 27% to 29%" in r["content"]
    assert "$40 billion debt and equity financing plan" in r["content"]
    assert "not_text" not in r["content"]  # script stripped
    assert r["source"] == "sec.gov"
    # excerpts are computed even without truncation (they feed the pm_brief block)
    kws = {e["keyword"] for e in r["excerpts"]}
    assert any("financing" in k for k in kws)
    assert any("chief executive officer" in k for k in kws)
    # index.json fetched from the accession directory
    assert any(u.endswith("000134143926000123/index.json") for u in calls)


def test_fetch_skips_non_results_8k(monkeypatch):
    """The most recent 8-K (officer change, 5.02) must be skipped in favor of
    the older item-2.02 results 8-K."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-20", "accession": "0000000000-26-000002",
         "primary_doc": "b.htm", "items": "5.02"},
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0000000000-26-000001",
         "primary_doc": "a.htm", "items": "2.02,9.01"},
    ])
    index = _stub_index_json(["a.htm", "ex99_1.htm"])
    fake, _ = _fake_http(submissions, index, _EXHIBIT_HTML)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("MSFT", "2026-07-02")
    assert r.get("unavailable") is not True
    assert r["filing_date"] == "2026-06-10"


def test_fetch_ignores_results_8k_after_trade_date(monkeypatch):
    """Look-ahead bias guard: a results 8-K filed after trade_date must not be used."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0000000000-26-000001",
         "primary_doc": "a.htm", "items": "2.02"},
    ])
    index = _stub_index_json(["a.htm", "ex99_1.htm"])
    fake, _ = _fake_http(submissions, index, _EXHIBIT_HTML)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("MSFT", "2026-06-09")
    assert r.get("unavailable") is True


def test_fetch_no_results_8k_is_honest_na(monkeypatch):
    """Foreign filers etc. may have no item-2.02 8-K at all: honest n/a, no raise."""
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-05-01", "accession": "0000000000-26-000009",
         "primary_doc": "z.htm", "items": "8.01"},
    ])
    fake, _ = _fake_http(submissions, None, None)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("BABA", "2026-07-02")
    assert r.get("unavailable") is True
    assert "2.02" in r["reason"]


def test_fetch_index_unreachable_fail_soft(monkeypatch):
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0000000000-26-000001",
         "primary_doc": "a.htm", "items": "2.02"},
    ])
    fake, _ = _fake_http(submissions, None, _EXHIBIT_HTML)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("MSFT", "2026-07-02")
    assert r.get("unavailable") is True


def test_fetch_no_ex99_in_index_fail_soft(monkeypatch):
    from tradingagents.agents.utils import sec_edgar

    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0000000000-26-000001",
         "primary_doc": "a.htm", "items": "2.02"},
    ])
    index = _stub_index_json(["a.htm", "logo.jpg"])
    fake, _ = _fake_http(submissions, index, _EXHIBIT_HTML)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("MSFT", "2026-07-02")
    assert r.get("unavailable") is True
    assert "99" in r["reason"]


def test_fetch_truncates_but_excerpts_come_from_full_text(monkeypatch):
    """The guidance sentence sits past the truncation cut; the excerpt must
    still carry it (same targeted-excerpt contract as the 10-K fetch)."""
    from tradingagents.agents.utils import sec_edgar

    filler = b"<p>" + b"tables and boilerplate " * 300 + b"</p>"
    html = b"<html><body>" + filler + \
        b"<p>Total Revenues expected to grow 27% to 29% under our guidance.</p>" + \
        b"</body></html>"
    submissions = _stub_submissions_json([
        {"form": "8-K", "filing_date": "2026-06-10", "accession": "0000000000-26-000001",
         "primary_doc": "a.htm", "items": "2.02"},
    ])
    index = _stub_index_json(["a.htm", "ex99_1.htm"])
    fake, _ = _fake_http(submissions, index, html)
    monkeypatch.setattr(sec_edgar, "_http_get", fake)

    r = sec_edgar.fetch_earnings_release("MSFT", "2026-07-02", max_text_chars=2_000)
    assert r["content_truncated"] is True
    assert len(r["content"]) <= 2_000
    assert "expected to grow 27% to 29%" not in r["content"]
    joined = " ".join(e["text"] for e in r["excerpts"])
    assert "expected to grow 27% to 29%" in joined


def test_release_keywords_cover_goal_topics():
    """The goal names guidance / RPO / financing / capex / expects / CEO+CFO
    quotes as the excerpt targets."""
    from tradingagents.agents.utils.sec_edgar import RELEASE_EXCERPT_KEYWORDS
    kws = " ".join(RELEASE_EXCERPT_KEYWORDS)
    for topic in ("guidance", "remaining performance obligation", "financing",
                  "capital expenditure", "expect", "chief executive officer",
                  "chief financial officer"):
        assert topic in kws


# ---------------------------------------------------------------- formatters

def _happy_release():
    return {
        "ticker": "ORCL", "form": "8-K", "filing_date": "2026-06-10",
        "accession_number": "0001341439-26-000123", "exhibit": "orcl-ex99_1.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/1341439/000134143926000123/orcl-ex99_1.htm",
        "items": "2.02,9.01",
        "content": "Oracle Announces Q4 Results. Total Revenues expected to grow 27% to 29%.",
        "content_truncated": False,
        "excerpts": [
            {"keyword": "financing", "position": 10,
             "text": "a $40 billion debt and equity financing plan"},
        ],
        "source": "sec.gov",
    }


def test_format_block_happy_path():
    from tradingagents.agents.utils.sec_edgar import format_earnings_release_block
    block = format_earnings_release_block(_happy_release())
    assert "## Latest earnings release (SEC 8-K Ex-99.1)" in block
    assert "2026-06-10" in block
    assert "$40 billion debt and equity financing plan" in block
    assert "expected to grow 27% to 29%" in block  # head snippet carries opening prose
    assert "verbatim" in block
    assert "not disclosed" in block
    assert "orcl-ex99_1.htm" in block


def test_format_block_unavailable_is_honest_na():
    from tradingagents.agents.utils.sec_edgar import format_earnings_release_block
    block = format_earnings_release_block(
        {"unavailable": True, "reason": "no item-2.02 (results) 8-K on/before trade_date"})
    assert "## Latest earnings release (SEC 8-K Ex-99.1) — n/a" in block
    assert "no item-2.02" in block
    assert "not disclosed" in block  # do-not-fabricate directive
    assert "Do not cite" in block or "do not cite" in block


def test_format_block_none_input():
    from tradingagents.agents.utils.sec_edgar import format_earnings_release_block
    block = format_earnings_release_block(None)
    assert "n/a" in block


def test_format_md_full_document():
    from tradingagents.agents.utils.sec_edgar import format_earnings_release_md
    md = format_earnings_release_md(_happy_release())
    assert md.startswith("# Earnings press release — ORCL 8-K")
    assert "orcl-ex99_1.htm" in md
    assert "Total Revenues expected to grow 27% to 29%" in md
    assert "$40 billion debt and equity financing plan" in md


def test_format_md_empty_when_unavailable():
    from tradingagents.agents.utils.sec_edgar import format_earnings_release_md
    assert format_earnings_release_md({"unavailable": True, "reason": "x"}) == ""


# ---------------------------------------------------------------- researcher wiring

_OHLCV_STUB = (
    "# Stock data for MSFT\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-04-30,415.0,420.0,395.0,408.0,1000000,0.0,0.0\n"
    "2026-05-01,408.0,425.0,379.0,410.0,1000000,0.0,0.0\n"
)

_INDICATOR_STUB = lambda val: (
    f"## sample values from 2026-04-01 to 2026-05-01:\n\n2026-05-01: {val}\n"
)


def _stub_researcher_fetchers(monkeypatch, researcher):
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {"ticker": t, "revenue": 100})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {"sentiment": 0.5})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0), "close_200_sma": _INDICATOR_STUB(380.0),
        "atr": _INDICATOR_STUB(4.2),
    })


def _run_pack(tmp_path, monkeypatch):
    from pathlib import Path
    from tradingagents.agents import researcher
    _stub_researcher_fetchers(monkeypatch, researcher)
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-01",
             "peers": ["GOOG"], "raw_dir": str(tmp_path / "raw")}
    raw = Path(state["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "pm_brief.md").write_text("# PM Pre-flight Brief\n", encoding="utf-8")
    researcher.fetch_research_pack(state)
    return raw


def test_researcher_writes_release_json_md_and_pm_brief_block(tmp_path, monkeypatch):
    from tradingagents.agents.utils import sec_edgar
    monkeypatch.setattr(sec_edgar, "fetch_earnings_release",
                        lambda t, d: _happy_release())
    raw = _run_pack(tmp_path, monkeypatch)

    assert (raw / "earnings_release.json").exists()
    saved = json.loads((raw / "earnings_release.json").read_text(encoding="utf-8"))
    assert saved["filing_date"] == "2026-06-10"
    md = (raw / "earnings_release.md").read_text(encoding="utf-8")
    assert "Total Revenues expected to grow 27% to 29%" in md
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Latest earnings release (SEC 8-K Ex-99.1)" in brief
    assert "$40 billion debt and equity financing plan" in brief


def test_researcher_release_unavailable_writes_honest_na(tmp_path, monkeypatch):
    from tradingagents.agents.utils import sec_edgar
    monkeypatch.setattr(sec_edgar, "fetch_earnings_release",
                        lambda t, d: {"unavailable": True, "ticker": t,
                                      "reason": "no item-2.02 (results) 8-K on/before trade_date"})
    raw = _run_pack(tmp_path, monkeypatch)

    # json records the miss; no md (nothing to quote); pm_brief carries n/a
    assert (raw / "earnings_release.json").exists()
    assert not (raw / "earnings_release.md").exists()
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Latest earnings release (SEC 8-K Ex-99.1) — n/a" in brief
    assert "no item-2.02" in brief


def test_researcher_release_fetch_raising_is_fail_open(tmp_path, monkeypatch):
    from tradingagents.agents.utils import sec_edgar

    def boom(t, d):
        raise RuntimeError("edgar exploded")

    monkeypatch.setattr(sec_edgar, "fetch_earnings_release", boom)
    raw = _run_pack(tmp_path, monkeypatch)  # must not raise

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Latest earnings release (SEC 8-K Ex-99.1) — unavailable" in brief
    assert "edgar exploded" in brief
