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
