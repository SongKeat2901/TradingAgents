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


@pytest.mark.parametrize("section, expected", [
    # The TIGR 2026-06-05 failure: em-dash separator, no colon -> regex matched
    # nothing -> peers.json wrote {} -> Phase 6.4 invariant crashed the run.
    ("- **FUTU** — Futu Holdings; closest competitor.\n"
     "- **HOOD** — Robinhood; benchmark.\n"
     "- **IBKR** — Interactive Brokers; institutional.",
     ["FUTU", "HOOD", "IBKR"]),
    # Regression: every previously-supported colon format must still parse.
    ("- AMKR: OSAT comp\n- TSM: foundry leader", ["AMKR", "TSM"]),
    ("- **COHR**: AI optics\n- **LITE**: datacom", ["COHR", "LITE"]),
    ("- **NOW** (ServiceNow): SaaS peer", ["NOW"]),
    ("- **NOW (ServiceNow):** SaaS peer", ["NOW"]),
    ("- *FN*: transceiver", ["FN"]),
    # Spaced plain-hyphen separator also parses.
    ("- **CRM** - Salesforce; SaaS", ["CRM"]),
    # The COIN 2026-06-24 failure: MULTIPLE tickers comma-listed on ONE bullet
    # line with a shared em-dash rationale -> the first-ticker matcher read zero
    # -> peers.json wrote {} -> Phase 6.4 invariant crashed the run.
    ("- **HOOD**, **CRCL**, **NDAQ**, **MSTR** — same US-tradable, yfinance set",
     ["HOOD", "CRCL", "NDAQ", "MSTR"]),
    # Bare (unbolded) comma list with a colon also parses.
    ("- HOOD, CRCL, NDAQ: crypto-broker comps", ["HOOD", "CRCL", "NDAQ"]),
    # Prose bullets (acronyms, hyphenated words) must NOT be read as tickers.
    ("- US-listed broker with AI exposure\n- ETF wrapper, no earnings", []),
])
def test_extract_peers_tolerates_separator_drift(section, expected):
    from tradingagents.agents.managers.pm_preflight import _extract_peers
    brief = "# Brief\n\n## Peer set\n" + section + "\n\n## Past-lesson summary\n- x\n"
    assert _extract_peers(brief) == expected


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


def test_pm_preflight_extracts_peers_with_parenthesized_company_name(tmp_path):
    """Regression for TSCO 2026-05-06: the LLM emits the form
    `- **BOOT** (Boot Barn): rationale` for retail/specialty peers. The prior
    regex required `**:` directly and silently dropped these lines, causing
    peers.json to write `{}` and the Researcher's Phase 6.4 invariant gate
    to fire. The expanded regex must accept the parenthesized expansion."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    tsco_brief = """\
# PM Pre-flight Brief: TSCO 2026-05-06

## Peer set
- **BOOT** (Boot Barn): rural/Western-lifestyle specialty retailer with overlapping customer.
- **CHWY** (Chewy): pet-category benchmark — pet/livestock is ~half of TSCO's revenue.
- **DKS** (Dick's Sporting Goods): comparable specialty big-box hardlines retailer.
- **ORLY** (O'Reilly Automotive): cleanest publicly-traded comp for "specialty hardlines".
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=tsco_brief)
    node = create_pm_preflight_node(fake_llm)
    out = node({
        "company_of_interest": "TSCO",
        "trade_date": "2026-05-06",
        "raw_dir": str(tmp_path / "raw"),
    })
    assert sorted(out["peers"]) == ["BOOT", "CHWY", "DKS", "ORLY"]


def test_pm_preflight_extracts_peers_when_bold_wraps_whole_label(tmp_path):
    """Regression for NOW (ServiceNow) 2026-05-29: the LLM wrapped the ENTIRE
    label including the colon in bold — `- **CRM (Salesforce):** rationale` —
    so the colon is followed by `**`, not whitespace. The prior regex's `:\\s`
    requirement dropped every line, peers.json wrote `{}`, and the run crashed
    ~75s in at the Phase 6.4 invariant. The tolerant regex must accept it."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    now_brief = """\
# PM Pre-flight Brief: NOW 2026-05-29

## Peer set
- **CRM (Salesforce):** closest public comp — enterprise SaaS platform.
- **WDAY (Workday):** enterprise back-office SaaS with similar cRPO model.
- **MSFT (Microsoft):** strategic frenemy — Power Platform/Copilot overlap.

## What this run must answer
1. x
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=now_brief)
    node = create_pm_preflight_node(fake_llm)
    out = node({
        "company_of_interest": "NOW",
        "trade_date": "2026-05-29",
        "raw_dir": str(tmp_path / "raw"),
    })
    assert sorted(out["peers"]) == ["CRM", "MSFT", "WDAY"]


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


_BRIEF_NO_PEERS = """\
# PM Pre-flight Brief: MSFT 2026-05-01

## Ticker validation
- Trading day: Friday 2026-05-01

## Highlights of the rest
- (the LLM omitted the Peer set section entirely — wk26 MSFT/NOW failure mode)

## What this run must answer
1. Is Azure growth durable?
"""


def test_pm_preflight_retries_and_recovers_when_peers_missing(tmp_path):
    """wk26 2026-06-24 (MSFT/NOW): the LLM omitted the '## Peer set' section,
    yielding empty peers -> the Researcher Phase 6.4 invariant aborted the run
    ~2 min in. PM Pre-flight must re-invoke with a corrective instruction and
    recover when the retry produces a usable peer section."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        AIMessage(content=_BRIEF_NO_PEERS),   # 1st draw: no peer section
        AIMessage(content=_VALID_BRIEF),      # retry: proper peers
    ]
    node = create_pm_preflight_node(fake_llm)
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-01",
             "raw_dir": str(tmp_path / "raw")}
    out = node(state)
    assert sorted(out["peers"]) == ["AAPL", "GOOG", "META"]
    assert fake_llm.invoke.call_count == 2  # initial empty + 1 recovering retry


def test_pm_preflight_fails_closed_after_peer_retries_exhausted(tmp_path):
    """If peers stay unextractable after the retries, PM Pre-flight gives up
    with empty peers (the downstream Phase 6.4 invariant then aborts) rather
    than looping forever — and the brief is still persisted."""
    from tradingagents.agents.managers.pm_preflight import (
        create_pm_preflight_node, _PEER_RETRY_LIMIT)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_BRIEF_NO_PEERS)
    node = create_pm_preflight_node(fake_llm)
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-01",
             "raw_dir": str(tmp_path / "raw")}
    out = node(state)
    assert out["peers"] == []
    assert fake_llm.invoke.call_count == 1 + _PEER_RETRY_LIMIT
    assert (tmp_path / "raw" / "pm_brief.md").exists()


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


def test_pm_preflight_appends_surprise_block_to_brief(tmp_path, monkeypatch):
    """PM Pre-flight must append a '## Surprise history' block, built from
    the same calendar.json, right after the Reporting status block."""
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
            "surprises": [
                {"date": "2026-04-29", "reported": 3.45, "estimate": 3.40, "surprise_pct": 1.47},
                {"date": "2026-01-29", "reported": 3.22, "estimate": 3.20, "surprise_pct": 0.63},
                {"date": "2025-10-29", "reported": 3.10, "estimate": 3.05, "surprise_pct": 1.64},
                {"date": "2025-07-30", "reported": 2.95, "estimate": 2.90, "surprise_pct": 1.72},
            ],
        },
        "GOOGL": {
            "last_reported": "2026-04-22",
            "fiscal_period": "Q1 2026",
            "next_expected": "2026-07-23",
            "source": "yfinance",
            "surprises": [],
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
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")

    assert "## Surprise history" in brief
    assert brief.index("## Surprise history") > brief.index("## Reporting status")
    assert "### MSFT" in brief
    assert "2026-04-29" in brief
    assert "3.45" in brief
    assert "+1.47%" in brief
    assert "beat 4 of last 4" in brief
    # GOOGL has an empty surprises list — no section rendered for it.
    assert "### GOOGL" not in brief
    assert out["pm_brief"] == brief


def test_pm_preflight_surprise_block_omitted_when_no_surprises(tmp_path, monkeypatch):
    """No ticker has surprise history -> no '## Surprise history' block at all
    (free-data honesty: don't render an empty section)."""
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
            # No "surprises" key at all (mirrors pre-Task-7 calendar.json shape).
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
    node(state)

    raw = Path(state["raw_dir"])
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Surprise history" not in brief


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


def test_pm_preflight_injects_canonical_ticker_identity(tmp_path, monkeypatch):
    """Phase 7.6: when yfinance returns a canonical identity, it must be
    prepended to the system prompt as immutable ground truth. Closes the
    ASX same-symbol-different-company failure mode."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    captured = {}

    def fake_invoke(messages):
        captured["system"] = messages[0].content
        captured["human"] = messages[1].content
        return AIMessage(content=(
            "# PM Pre-flight Brief: ASX 2026-05-07\n\n"
            "## Peer set\n- AMKR: OSAT direct competitor\n"
        ))

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = fake_invoke

    # Mock yfinance to return ASE Technology Holding identity
    fake_info = {
        "longName": "ASE Technology Holding Co., Ltd.",
        "country": "Taiwan",
        "sector": "Technology",
        "industry": "Semiconductor Equipment & Materials",
        "quoteType": "EQUITY",
        "marketCap": 12_000_000_000,  # mid-cap by global standards
    }
    fake_ticker = MagicMock()
    fake_ticker.info = fake_info
    monkeypatch.setattr("yfinance.Ticker", lambda sym: fake_ticker)

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "ASX",
        "trade_date": "2026-05-07",
        "raw_dir": str(tmp_path / "raw"),
    })

    sys_prompt = captured["system"]
    # Prefix landed at the top
    assert "AUTHORITATIVE TICKER IDENTITY" in sys_prompt
    # Canonical identity present
    assert "ASE Technology Holding Co., Ltd." in sys_prompt
    assert "Taiwan" in sys_prompt
    # Instruction to override prior knowledge
    assert "MUST describe THIS company" in sys_prompt
    # The ASX example reference (helps the LLM understand the failure mode)
    assert "ASX" in sys_prompt and "Australian Securities Exchange" in sys_prompt


def test_pm_preflight_skips_identity_injection_on_yfinance_error(tmp_path, monkeypatch):
    """If yfinance fails / returns no info, the prompt falls back to the
    existing system template — pipeline should still run."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    captured = {}

    def fake_invoke(messages):
        captured["system"] = messages[0].content
        return AIMessage(content="# PM Pre-flight Brief: XXXX 2026-05-07\n## Peer set\n- AMKR: x\n")

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = fake_invoke

    # yfinance Ticker lookup raises
    def _raise(_sym):
        raise RuntimeError("yfinance unreachable")
    monkeypatch.setattr("yfinance.Ticker", _raise)

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "XXXX",
        "trade_date": "2026-05-07",
        "raw_dir": str(tmp_path / "raw"),
    })

    # No identity prefix → system prompt starts with the regular template
    sys_prompt = captured["system"]
    assert "AUTHORITATIVE TICKER IDENTITY" not in sys_prompt
    # Standard template still in place
    assert "Portfolio Manager performing pre-flight" in sys_prompt


def test_pm_preflight_skips_identity_when_long_name_missing(tmp_path, monkeypatch):
    """Some delisted / obscure tickers return info dicts with no longName.
    The prefix is omitted in that case (don't inject empty fields)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    captured = {}

    def fake_invoke(messages):
        captured["system"] = messages[0].content
        return AIMessage(content="# PM Pre-flight Brief\n## Peer set\n- AMKR: x\n")

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = fake_invoke

    fake_ticker = MagicMock()
    fake_ticker.info = {"sector": "Technology"}  # missing longName
    monkeypatch.setattr("yfinance.Ticker", lambda sym: fake_ticker)

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "DELISTED",
        "trade_date": "2026-05-07",
        "raw_dir": str(tmp_path / "raw"),
    })

    assert "AUTHORITATIVE TICKER IDENTITY" not in captured["system"]


def test_pm_preflight_calendar_block_renders_etf_as_structural_na(tmp_path, monkeypatch):
    """Phase 6.8: ETF rows in the calendar must render with the structural
    N/A label (passive instrument — no earnings reporting), NOT as
    `(yfinance unavailable)`. Surfaced by COIN 2026-05-06 where IBIT
    appeared as `IBIT (yfinance unavailable) unknown (yfinance
    unavailable)` — misleading because ETFs don't report earnings."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    monkeypatch.setattr(
        "tradingagents.agents.utils.calendar.compute_calendar",
        lambda d, t: {
            "trade_date": "2026-05-06",
            "_unavailable": ["IBIT"],
            "COIN": {
                "last_reported": "2026-02-12",
                "fiscal_period": "Q4 2025",
                "next_expected": "2026-05-07",
                "source": "yfinance",
            },
            "IBIT": {
                "unavailable": True,
                "structural": True,
                "instrument_type": "ETF",
                "reason": "etf — no earnings reporting",
            },
        },
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief")

    node = create_pm_preflight_node(fake_llm)
    node({
        "company_of_interest": "COIN",
        "trade_date": "2026-05-06",
        "raw_dir": str(tmp_path / "raw"),
    })

    brief = (Path(tmp_path) / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    # The misleading "yfinance unavailable" must NOT appear for the ETF
    ibit_line = next(line for line in brief.splitlines() if line.startswith("| IBIT"))
    assert "yfinance unavailable" not in ibit_line.lower()
    # Instead the structural N/A label is shown
    assert "no earnings reporting" in ibit_line.lower()
    assert "etf" in ibit_line.lower()


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


# NOTE: Phase 6.4 peer-ratio injection lives in researcher.py (which writes
# peers.json). PM Pre-flight runs BEFORE the Researcher, so peers.json doesn't
# exist when PM Pre-flight runs. The 3 corresponding tests are in
# tests/test_researcher.py::test_researcher_writes_peer_ratios_json_and_appends_block
# and friends.
