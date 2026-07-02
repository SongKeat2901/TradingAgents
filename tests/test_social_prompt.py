import pytest

pytestmark = pytest.mark.unit


def test_social_cites_sentiment_block():
    from tradingagents.agents.analysts import social_media_analyst as sa
    low = sa._SYSTEM.lower()
    assert "sentiment & consensus" in low or ("short interest" in low and "consensus" in low)
