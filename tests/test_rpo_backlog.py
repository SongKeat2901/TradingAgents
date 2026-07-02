"""RPO / backlog deep-dive block (pro-deck technique B, deck pp67-68).

Deterministic from SEC XBRL companyconcept
`us-gaap/RevenueRemainingPerformanceObligation`: total RPO + QoQ additions +
RPO/market-cap + RPO/TTM-revenue, with a peer comparison. Self-gating: names
without fresh RPO facts render "not applicable" (never fabricated).
"""
import pytest

from tradingagents.agents.utils.rpo_backlog import (
    STALE_DAYS,
    compute_rpo_backlog,
    dedupe_rpo_facts,
    format_rpo_block,
)

pytestmark = pytest.mark.unit


_RAW_FACTS = [
    {"end": "2025-08-31", "val": 455.3e9, "form": "10-Q", "filed": "2025-09-10"},
    {"end": "2025-11-30", "val": 523.3e9, "form": "10-Q", "filed": "2025-12-10"},
    {"end": "2026-02-28", "val": 552.6e9, "form": "10-Q", "filed": "2026-03-11"},
    # duplicate end from a later filing wins
    {"end": "2026-02-28", "val": 552.6e9, "form": "10-K", "filed": "2026-06-22"},
    {"end": "2026-05-31", "val": 638.0e9, "form": "10-K", "filed": "2026-06-22"},
    # future fact (after trade_date) must be dropped
    {"end": "2026-08-31", "val": 700.0e9, "form": "10-Q", "filed": "2026-09-10"},
]


def test_dedupe_filters_future_and_keeps_latest_filed():
    facts = dedupe_rpo_facts(_RAW_FACTS, "2026-07-01")
    ends = [f["end"] for f in facts]
    assert ends == ["2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31"]
    feb = [f for f in facts if f["end"] == "2026-02-28"][0]
    assert feb["form"] == "10-K"  # later-filed duplicate won


def test_compute_happy_path_matches_deck_numbers():
    facts = dedupe_rpo_facts(_RAW_FACTS, "2026-07-01")
    r = compute_rpo_backlog("ORCL", facts, "2026-07-01",
                            market_cap=410.5e9, revenue_ttm=67.4e9)
    assert r["applicable"] is True
    assert r["rpo_total"] == pytest.approx(638.0e9)
    assert r["as_of"] == "2026-05-31"
    # QoQ additions: 638.0 - 552.6 = 85.4B (deck p67 "+85")
    assert r["history"][-1]["qoq_add"] == pytest.approx(85.4e9, rel=1e-3)
    assert r["rpo_to_market_cap"] == pytest.approx(638.0 / 410.5, abs=0.01)
    assert r["rpo_to_revenue_ttm"] == pytest.approx(638.0 / 67.4, abs=0.05)


def test_not_applicable_when_no_facts():
    r = compute_rpo_backlog("KO", [], "2026-07-01", market_cap=250e9)
    assert r["applicable"] is False
    assert "not" in r["reason"].lower()


def test_not_applicable_when_stale():
    stale = [{"end": "2020-06-30", "val": 41.0e9, "form": "10-Q", "filed": "2020-07-30"}]
    r = compute_rpo_backlog("AMZN", dedupe_rpo_facts(stale, "2026-07-01"), "2026-07-01",
                            market_cap=2440e9)
    assert r["applicable"] is False
    assert "stale" in r["reason"].lower() or "not" in r["reason"].lower()
    assert STALE_DAYS >= 365  # annual filers must survive the gate


def test_ratios_none_when_denominators_missing():
    facts = dedupe_rpo_facts(_RAW_FACTS, "2026-07-01")
    r = compute_rpo_backlog("ORCL", facts, "2026-07-01")
    assert r["applicable"] is True
    assert r["rpo_to_market_cap"] is None
    assert r["rpo_to_revenue_ttm"] is None


def test_peer_rows():
    facts = dedupe_rpo_facts(_RAW_FACTS, "2026-07-01")
    peers = [
        {"ticker": "MSFT",
         "facts": [{"end": "2026-03-31", "val": 633e9, "form": "10-Q", "filed": "2026-04-30"}],
         "market_cap": 2620e9},
        {"ticker": "AMZN",
         "facts": [{"end": "2020-06-30", "val": 41e9, "form": "10-Q", "filed": "2020-07-30"}],
         "market_cap": 2440e9},
    ]
    r = compute_rpo_backlog("ORCL", facts, "2026-07-01",
                            market_cap=410.5e9, revenue_ttm=67.4e9, peers=peers)
    msft = [p for p in r["peers"] if p["ticker"] == "MSFT"][0]
    assert msft["applicable"] is True
    assert msft["rpo_to_market_cap"] == pytest.approx(633 / 2620, abs=0.01)
    amzn = [p for p in r["peers"] if p["ticker"] == "AMZN"][0]
    assert amzn["applicable"] is False  # stale -> honest n/a


def test_format_block_happy_path():
    facts = dedupe_rpo_facts(_RAW_FACTS, "2026-07-01")
    peers = [{"ticker": "MSFT",
              "facts": [{"end": "2026-03-31", "val": 633e9, "form": "10-Q", "filed": "2026-04-30"}],
              "market_cap": 2620e9}]
    r = compute_rpo_backlog("ORCL", facts, "2026-07-01",
                            market_cap=410.5e9, revenue_ttm=67.4e9, peers=peers)
    block = format_rpo_block(r, "2026-07-01")
    assert "## RPO / backlog deep-dive" in block
    assert "$638.0B" in block
    assert "85.4" in block  # additions column
    assert "1.55x" in block  # RPO / market cap
    assert "MSFT" in block
    assert "verbatim" in block
    # waterfall honesty: points at the filing excerpt, never invents buckets
    assert "excerpt" in block.lower() or "not disclosed" in block.lower()


def test_format_block_not_applicable():
    r = compute_rpo_backlog("KO", [], "2026-07-01")
    block = format_rpo_block(r, "2026-07-01")
    assert "## RPO / backlog deep-dive" in block
    assert "not applicable" in block.lower()
    assert "do not fabricate" in block.lower() or "Do not cite" in block


def test_researcher_wires_rpo_backlog():
    import inspect
    from tradingagents.agents import researcher
    src = inspect.getsource(researcher)
    assert "rpo_backlog" in src
    assert "format_rpo_block" in src


def test_financial_role_prompt_has_rpo_discipline():
    from tradingagents.agents.analysts.fundamentals_roles import _SYSTEM_FINANCIAL
    assert "RPO / backlog deep-dive" in _SYSTEM_FINANCIAL
    assert "waterfall" in _SYSTEM_FINANCIAL.lower()
