import inspect

import pytest

pytestmark = pytest.mark.unit

_LABELS = ["Shares Short", "Shares Short Prior Month", "Short Ratio Days To Cover",
           "Short Percent Of Float", "Analyst Recommendation", "Analyst Recommendation Mean",
           "Number Of Analyst Opinions", "Target Mean Price", "Target Median Price",
           "Target High Price", "Target Low Price", "Current Price"]


def test_get_fundamentals_exposes_sentiment_fields():
    from tradingagents.dataflows import y_finance
    src = inspect.getsource(y_finance.get_fundamentals)
    for lbl in _LABELS:
        assert f'"{lbl}"' in src, f"missing field label: {lbl}"


from tradingagents.agents.utils.sentiment_consensus import (
    compute_sentiment_consensus, format_sentiment_block,
)

_BLOB = (
    "# Fundamentals\nName: Acme\nCurrent Price: 100\n"
    "Shares Short: 120\nShares Short Prior Month: 100\n"
    "Short Ratio Days To Cover: 2.5\nShort Percent Of Float: 0.0128\n"
    "Analyst Recommendation: strong_buy\nAnalyst Recommendation Mean: 1.34\n"
    "Number Of Analyst Opinions: 55\nTarget Mean Price: 150\nTarget Median Price: 145\n"
    "Target High Price: 200\nTarget Low Price: 110\n"
)


def test_compute_sentiment_consensus():
    r = compute_sentiment_consensus({"fundamentals": _BLOB})
    assert r["short_pct_float"] == 1.28          # 0.0128*100
    assert r["days_to_cover"] == 2.5
    assert r["short_mom_change_pct"] == 20.0      # (120-100)/100*100
    assert r["rating"] == "strong_buy" and r["n_analysts"] == 55
    assert r["target_mean"] == 150 and r["target_upside_pct"] == 50.0  # 150/100-1


def test_missing_fields_na():
    r = compute_sentiment_consensus({"fundamentals": "# f\nName: X\n"})
    assert r["short_pct_float"] is None and r["target_upside_pct"] is None


def test_block_render():
    block = format_sentiment_block(compute_sentiment_consensus({"fundamentals": _BLOB}))
    assert "## Sentiment & consensus" in block
    assert "strong_buy" in block and "verbatim" in block
    na = format_sentiment_block(compute_sentiment_consensus({"fundamentals": ""}))
    assert "unavailable" in na.lower() or "n/a" in na.lower()


def test_num_malformed_token_does_not_crash():
    blob = (
        "# Fundamentals\nName: Acme\nCurrent Price: 100\n"
        "Shares Short: 1.2.3\nShares Short Prior Month: 100\n"
        "Short Ratio Days To Cover: 2.5\nShort Percent Of Float: 0.0128\n"
    )
    r = compute_sentiment_consensus({"fundamentals": blob})
    assert r["short_mom_change_pct"] is None  # Shares Short unparseable -> degrades to None
    assert r["days_to_cover"] == 2.5           # other valid fields still parse
    assert r["short_pct_float"] == 1.28


def test_target_upside_reference_price_fallback():
    blob = "# Fundamentals\nName: Acme\nTarget Mean Price: 150\n"
    r = compute_sentiment_consensus({"fundamentals": blob}, reference_price=100)
    assert r["target_upside_pct"] == 50.0
    r_no_ref = compute_sentiment_consensus({"fundamentals": blob})
    assert r_no_ref["target_upside_pct"] is None
