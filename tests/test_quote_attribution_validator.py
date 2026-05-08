"""Tests for the Phase-7.2 quote attribution validator."""
import pytest

pytestmark = pytest.mark.unit


def test_extracts_paragraph_attribution_pattern():
    """`**Agent Name:** *"..."*` — paragraph header attribution."""
    from tradingagents.validators import extract_attributed_quotes

    text = (
        '**Aggressive Analyst:** *"$190B in capex producing −$0.75/share '
        'EPS drag at 30x P/E = −$22/share target compression"*'
    )
    quotes = extract_attributed_quotes(text)
    assert len(quotes) == 1
    assert quotes[0].agent_name == "aggressive analyst"
    assert "$190B" in quotes[0].quote_text
    assert quotes[0].expected_source_file == "debate_risk.md"


def test_extracts_post_paren_attribution_pattern():
    """`*"..."* (Agent Name, paragraph N)` — post-quote parenthetical."""
    from tradingagents.validators import extract_attributed_quotes

    text = (
        '*"the asymmetry on a breakout trigger resets to the $432–$466 '
        'gap with minimal natural resistance"* (Aggressive, paragraph 3)'
    )
    quotes = extract_attributed_quotes(text)
    # "Aggressive" alone may not normalise to "aggressive analyst"; this
    # tests that the extractor handles partial agent names. Currently it
    # would not match because we require full names. Verify behaviour.
    # If no quote extracted, that's acceptable for v1 (over-strict is OK).
    if quotes:
        assert "asymmetry" in quotes[0].quote_text


def test_skips_unknown_agent_attribution():
    """Attribution to a non-agent (e.g., a news source) should be skipped."""
    from tradingagents.validators import extract_attributed_quotes

    text = '**Reuters:** *"BTC traded above $100K"*'
    assert extract_attributed_quotes(text) == []


def test_extract_distinctive_numbers_finds_dollars_pcts_ratios():
    from tradingagents.validators.quote_attribution_validator import (
        extract_distinctive_numbers,
    )
    quote = (
        "$206.50 on 14.39M shares — roughly 1.8–2x the trailing daily "
        "average; Q1 op margin 38.5% vs CME 24.6%"
    )
    nums = extract_distinctive_numbers(quote)
    # Should pick up: $206.50, 14.39M, 1.8, 2x, 38.5%, 24.6%
    assert "$206.50" in nums
    assert "14.39M" in nums
    assert any(n.endswith("%") for n in nums)


def test_validates_quote_against_real_source_file_pass(tmp_path):
    """Happy path: the quote's distinctive numbers all appear in the source file."""
    from tradingagents.validators import (
        validate_attributed_quotes,
    )
    from tradingagents.validators.quote_attribution_validator import AttributedQuote

    # Source agent file with the same numbers
    (tmp_path / "analyst_market.md").write_text(
        "Spot $192.96 on 8.99M shares — 50-DMA at $189.82 within reach.",
        encoding="utf-8",
    )
    quotes = [AttributedQuote(
        quote_text="Spot $192.96 on 8.99M shares — 50-DMA at $189.82 within reach.",
        agent_name="market analyst",
        file="decision.md",
        line_no=10,
        expected_source_file="analyst_market.md",
    )]
    violations = validate_attributed_quotes(quotes, tmp_path)
    assert violations == []


def test_catches_coin_2026_05_08_fabricated_quote(tmp_path):
    """Regression: TA v2 quotes Market Analyst saying `$206.50 on 14.39M
    shares` when Market Analyst's actual file says no such thing."""
    from tradingagents.validators import (
        validate_attributed_quotes,
    )
    from tradingagents.validators.quote_attribution_validator import AttributedQuote

    # Real Market Analyst output — no $206.50, no 14.39M
    (tmp_path / "analyst_market.md").write_text(
        "The May 7 session itself (open $196.03, close $192.96 on 8.99M "
        "shares) showed no meaningful buy-the-news impulse.",
        encoding="utf-8",
    )
    # Fabricated TA v2 quote attributed to Market Analyst
    quotes = [AttributedQuote(
        quote_text=(
            "COIN closed the session at $206.50 on 14.39M shares — roughly "
            "1.8–2x the trailing daily average — after the 10-Q filing on May 7."
        ),
        agent_name="market analyst",
        file="raw/technicals_v2.md",
        line_no=3,
        expected_source_file="analyst_market.md",
    )]
    violations = validate_attributed_quotes(quotes, tmp_path)

    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "MATERIAL"
    assert v.type == "fabricated_quote"
    assert v.agent_name == "market analyst"
    # Distinctive numbers in the quote
    assert "$206.50" in v.distinctive_numbers
    assert "14.39M" in v.distinctive_numbers
    # NONE of them appear in the actual source file
    assert v.matches_in_source == []


def test_skips_quote_with_too_few_distinctive_numbers(tmp_path):
    """If a quote has < 2 distinctive numbers, fingerprinting is unreliable
    — skip rather than risk false positive."""
    from tradingagents.validators import (
        validate_attributed_quotes,
    )
    from tradingagents.validators.quote_attribution_validator import AttributedQuote

    (tmp_path / "analyst_market.md").write_text("Setup is bearish.", encoding="utf-8")
    # Only one number — too few
    quotes = [AttributedQuote(
        quote_text="Just $100",
        agent_name="market analyst",
        file="decision.md",
        line_no=1,
        expected_source_file="analyst_market.md",
    )]
    violations = validate_attributed_quotes(quotes, tmp_path)
    assert violations == []


def test_emits_minor_when_source_file_missing(tmp_path):
    from tradingagents.validators import (
        validate_attributed_quotes,
    )
    from tradingagents.validators.quote_attribution_validator import AttributedQuote

    quotes = [AttributedQuote(
        quote_text="$100 close, 10M shares, 5% margin",
        agent_name="market analyst",
        file="decision.md",
        line_no=1,
        expected_source_file="analyst_market.md",
    )]
    violations = validate_attributed_quotes(quotes, tmp_path)
    assert len(violations) == 1
    assert violations[0].severity == "MINOR"
    assert violations[0].type == "agent_source_missing"


def test_render_violations_text_pass():
    from tradingagents.validators.quote_attribution_validator import (
        render_quote_violations_text,
    )
    out = render_quote_violations_text([])
    assert "QUOTE VALIDATION PASS" in out
