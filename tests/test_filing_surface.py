import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.sec_edgar import (
    _collect_filings, format_filing_surface_block,
)
from datetime import datetime

_SUBS = {"filings": {"recent": {
    "form": ["8-K", "10-Q", "8-K", "DEF 14A", "8-K"],
    "filingDate": ["2026-06-30", "2026-04-29", "2026-04-29", "2025-10-15", "2027-01-01"],
    "accessionNumber": ["0-1", "0-2", "0-3", "0-4", "0-5"],
    "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm", "e.htm"],
    "items": ["8.01", "", "2.02,9.01", "", "1.01"],
}}}


def test_collect_filters_form_and_future_dates():
    td = datetime(2026, 7, 1)
    eights = _collect_filings(_SUBS, td, ("8-K",), 5)
    # the 2027-01-01 8-K is after trade_date -> excluded; 2 remain
    assert [f["filing_date"] for f in eights] == ["2026-06-30", "2026-04-29"]
    assert eights[1]["items"] == "2.02,9.01"


def test_collect_limit():
    td = datetime(2026, 7, 1)
    assert len(_collect_filings(_SUBS, td, ("8-K",), 1)) == 1


def test_collect_def14a():
    td = datetime(2026, 7, 1)
    proxies = _collect_filings(_SUBS, td, ("DEF 14A",), 1)
    assert proxies[0]["filing_date"] == "2025-10-15"


def test_block_renders_items_legend_and_proxy():
    surface = {"ticker": "MSFT",
               "recent_8k": [{"filing_date": "2026-04-29", "items": "2.02,9.01", "url": "u"}],
               "latest_def14a": {"filing_date": "2025-10-15", "url": "p"}}
    block = format_filing_surface_block(surface)
    assert "## SEC filing surface" in block
    assert "results of operations" in block and "financial statements/exhibits" in block
    assert "2025-10-15" in block and "verbatim" in block


def test_block_unavailable():
    block = format_filing_surface_block({"unavailable": True, "reason": "CIK not found for ticker XYZ"})
    assert "n/a" in block and "do not cite" in block.lower()


def test_block_none_reported():
    block = format_filing_surface_block({"ticker": "X", "recent_8k": [], "latest_def14a": None})
    assert "none reported" in block.lower()
