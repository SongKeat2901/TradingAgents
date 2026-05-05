"""Tests for PM Pre-flight node."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_compute_calendar(monkeypatch):
    """Prevent any test from accidentally hitting yfinance.network. Tests that
    need calendar content override this with their own monkeypatch."""
    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        lambda d, t: {"trade_date": d, "_unavailable": []},
    )


@pytest.fixture(autouse=True)
def _stub_fetch_latest_filing(monkeypatch):
    """Prevent any test from accidentally hitting SEC EDGAR network. Tests
    that need filing content override this with their own monkeypatch."""
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: {"unavailable": True, "reason": "stubbed", "ticker": t},
    )


_VALID_BRIEF = """\
# PM Pre-flight Brief: MSFT 2026-05-01

## Ticker validation
- Trading day: Friday 2026-05-01
- Sector (yfinance): Technology / Software
- Market cap: Mega-cap

## Business model classification
- yfinance sector: Technology / Software
- Actual business model: **Cloud + productivity software + AI infrastructure**.

Interpretation rules for analysts:
- Revenue is enterprise software + cloud (Azure) + Office.

## Peer set
- GOOG: nearest cloud + AI infra peer
- META: nearest scale + AI infra peer
- AAPL: nearest mega-cap tech peer

## Past-lesson summary
- No prior decision on this ticker in memory log.

## What this run must answer
1. Is Azure growth durable?
2. Is AI capex generating ROI?
3. Is the multiple still defensible?
"""


def test_pm_preflight_writes_brief_to_raw_dir(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)

    brief_path = tmp_path / "raw" / "pm_brief.md"
    assert brief_path.exists()
    content = brief_path.read_text(encoding="utf-8")
    assert "Business model classification" in content
    assert out["pm_brief"] == content


def test_pm_preflight_extracts_peers_from_brief(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)
    assert sorted(out["peers"]) == ["AAPL", "GOOG", "META"]


def test_pm_preflight_extracts_peers_with_markdown_bold(tmp_path):
    """LLMs sometimes emit '- **GOOGL**: ...' instead of '- GOOGL: ...'.
    The regex must tolerate optional markdown bold/italic around the ticker."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    bold_brief = """\
# PM Pre-flight Brief: MSFT 2026-05-01

## Peer set
- **GOOGL**: Hyperscaler peer (GCP)
- *AMZN*: AWS comp (italic variant)
- ORCL: plain form
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=bold_brief)
    node = create_pm_preflight_node(fake_llm)
    out = node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    })
    assert sorted(out["peers"]) == ["AMZN", "GOOGL", "ORCL"]


def test_pm_preflight_handles_no_peers_etf(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    spy_brief = """\
# PM Pre-flight Brief: SPY 2026-05-01

## Business model classification
- yfinance sector: ETF / Index
- Actual business model: **S&P 500 index ETF** — peer comparison not applicable.

## Peer set
(none — index ETF; compare to other broad-market ETFs only on request)
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=spy_brief)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "SPY",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)
    assert out["peers"] == []


def test_pm_preflight_appends_calendar_block_to_brief(tmp_path, monkeypatch):
    """PM Pre-flight must compute the calendar (via the peers it just extracted)
    and append a deterministic 'Reporting status' block after the LLM content."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_calendar = {
        "trade_date": "2026-05-01",
        "_unavailable": [],
        "MSFT": {
            "last_reported": "2026-04-29",
            "fiscal_period": "FY26 Q3",
            "next_expected": "2026-07-25",
            "source": "yfinance",
        },
        "GOOGL": {
            "last_reported": "2026-04-22",
            "fiscal_period": "Q1 2026",
            "next_expected": "2026-07-23",
            "source": "yfinance",
        },
    }
    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        lambda d, t: fake_calendar,
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# PM Brief: MSFT 2026-05-01\n\n## Peer set\n- GOOGL: hyperscaler peer\n")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)

    raw = Path(state["raw_dir"])
    assert (raw / "calendar.json").exists()

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert brief.startswith("# PM Brief: MSFT 2026-05-01")
    assert "## Reporting status (relative to trade_date 2026-05-01)" in brief
    assert "MSFT" in brief
    assert "FY26 Q3 reported 2026-04-29" in brief
    assert "GOOGL" in brief
    assert "Q1 2026 reported 2026-04-22" in brief
    assert "already happened" in brief
    assert "2026-07-25" in brief
    assert "2026-07-23" in brief
    assert out["pm_brief"] == brief


def test_pm_preflight_writes_calendar_json_using_extracted_peers(tmp_path, monkeypatch):
    """compute_calendar must be called with the peer list extracted from the
    LLM brief, not with state['peers'] (PM Pre-flight runs before peers are
    written to state by anyone else)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    captured = {}

    def fake_compute_calendar(trade_date, tickers):
        captured["trade_date"] = trade_date
        captured["tickers"] = list(tickers)
        return {"trade_date": trade_date, "_unavailable": []}

    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        fake_compute_calendar,
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=(
        "# PM Pre-flight Brief: MSFT 2026-05-01\n\n"
        "## Peer set\n"
        "- GOOGL: peer one\n"
        "- AMZN: peer two\n"
        "- ORCL: peer three\n"
    ))

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    })

    assert captured["trade_date"] == "2026-05-01"
    assert captured["tickers"][0] == "MSFT"
    assert set(captured["tickers"]) == {"MSFT", "GOOGL", "AMZN", "ORCL"}


def test_pm_preflight_calendar_block_renders_unavailable_tickers(tmp_path, monkeypatch):
    """A ticker marked unavailable in calendar.json should render '(yfinance unavailable)'."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        lambda d, t: {
            "trade_date": "2026-05-01",
            "_unavailable": ["TICKERX"],
            "MSFT": {
                "last_reported": "2026-04-29",
                "fiscal_period": "FY26 Q3",
                "next_expected": "2026-07-25",
                "source": "yfinance",
            },
            "TICKERX": {"unavailable": True, "reason": "yfinance returned no earnings dates"},
        },
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief")

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    })

    brief = (Path(tmp_path) / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "TICKERX" in brief
    assert "yfinance unavailable" in brief.lower() or "(yfinance unavailable)" in brief


def test_pm_preflight_system_prompt_has_temporal_anchor():
    """Option A: PM Pre-flight _SYSTEM must include a Temporal anchor section
    instructing the LLM not to fabricate past-vs-future status."""
    from tradingagents.agents.managers.pm_preflight import _SYSTEM
    assert "Temporal anchor" in _SYSTEM or "trade date as \"today\"" in _SYSTEM
    assert "data to follow" in _SYSTEM or "already occurred" in _SYSTEM


def test_pm_preflight_writes_sec_filing_md_when_filing_available(tmp_path, monkeypatch):
    """If fetch_latest_filing returns a happy-path dict, PM Pre-flight writes
    raw/sec_filing.md AND appends a 'Recent SEC filing' footer to pm_brief.md."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_filing = {
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
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: fake_filing,
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    node(state)

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert sec_path.exists()
    sec_content = sec_path.read_text(encoding="utf-8")
    assert "MSFT 10-Q filed 2026-04-29" in sec_content
    assert "Azure and other cloud services revenue increased 40%" in sec_content

    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Recent SEC filing (relative to trade_date 2026-05-01)" in brief
    assert "MSFT 10-Q filed 2026-04-29" in brief
    assert "treat as **known data**" in brief.lower() or "Treat as **known data**" in brief


def test_pm_preflight_omits_sec_filing_when_unavailable(tmp_path, monkeypatch):
    """If fetch_latest_filing returns unavailable, PM Pre-flight must NOT write
    raw/sec_filing.md and must NOT add the filing footer to pm_brief.md."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: {"unavailable": True, "reason": "EDGAR unreachable", "ticker": t},
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    node(state)

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert not sec_path.exists()
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "Recent SEC filing" not in brief


def test_pm_preflight_handles_fetcher_exception(tmp_path, monkeypatch):
    """If fetch_latest_filing raises, PM Pre-flight must degrade gracefully:
    no sec_filing.md written, no footer, pipeline still returns normally."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    def _raises(t, d):
        raise RuntimeError("simulated EDGAR client crash")

    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing", _raises
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)  # MUST NOT raise

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert not sec_path.exists()
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "Recent SEC filing" not in brief
    assert out["pm_brief"] == brief


def test_pm_preflight_appends_peer_ratios_block(tmp_path, monkeypatch):
    """If raw/peers.json is present, PM Pre-flight writes raw/peer_ratios.json
    AND appends a '## Peer ratios' block to pm_brief.md after the calendar
    + SEC filing blocks."""
    import json as _json
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    # peers.json with a single peer that has all rows
    peers_json = {
        "GOOGL": {
            "ticker": "GOOGL",
            "trade_date": "2026-05-01",
            "fundamentals": "PE Ratio (TTM): 29.23\nForward PE: 26.68\n",
            "balance_sheet": "",
            "cashflow": "# header\nCapital Expenditure,-35700000000\n",
            "income_statement": (
                "# header\n"
                "Total Revenue,109900000000\n"
                "Operating Income,39700000000\n"
            ),
        }
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "peers.json").write_text(_json.dumps(peers_json), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    pr_path = raw / "peer_ratios.json"
    assert pr_path.exists()
    pr = _json.loads(pr_path.read_text(encoding="utf-8"))
    assert pr["trade_date"] == "2026-05-01"
    # 35.7B / 109.9B = 32.48%
    assert abs(pr["GOOGL"]["latest_quarter_capex_to_revenue"] - 32.48) < 0.05

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios (computed from raw/peers.json, trade_date 2026-05-01)" in brief
    assert "GOOGL" in brief
    assert "32.5%" in brief  # rendered with 1 decimal
    assert "29.23x" in brief
    # The peer block must come AFTER any calendar/SEC-filing blocks (last appended).
    assert brief.rfind("## Peer ratios") > brief.rfind("# PM Pre-flight Brief")


def test_pm_preflight_skips_peer_ratios_when_peers_json_missing(tmp_path):
    """No peers.json → no peer_ratios.json written, no peer block in brief."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    # NO peers.json

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    assert not (raw / "peer_ratios.json").exists()
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios" not in brief


def test_pm_preflight_handles_peer_ratios_compute_exception(tmp_path, monkeypatch):
    """If compute_peer_ratios raises (e.g., unexpected JSON shape), PM
    Pre-flight degrades gracefully — no crash, no peer_ratios.json,
    no block, but brief.md still has the LLM content."""
    import json as _json
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "peers.json").write_text(_json.dumps({"GOOGL": {}}), encoding="utf-8")

    def _raises(*a, **kw):
        raise RuntimeError("simulated peer-compute crash")

    monkeypatch.setattr(
        "tradingagents.agents.utils.peer_ratios.compute_peer_ratios", _raises
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    out = node(state)  # MUST NOT raise

    assert not (raw / "peer_ratios.json").exists()
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios" not in brief
    # The LLM-written content + earlier appended blocks (calendar / sec_filing)
    # are unaffected. Just verify the brief still looks like the LLM brief.
    assert out["pm_brief"] == brief
