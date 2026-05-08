"""Tests for the Phase-7.1 claim extractor."""
import pytest

pytestmark = pytest.mark.unit


def test_extracts_iso_date_close():
    from tradingagents.validators import extract_date_close_claims
    text = "technicals through the 2026-05-08 close at $206.50."
    claims = extract_date_close_claims(text)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-05-08"
    assert claims[0].price == 206.50


def test_extracts_month_day_close_using_anchor_year():
    from tradingagents.validators import extract_date_close_claims
    text = "the May 8 session subsequently closed at $206.50 per prices.json"
    claims = extract_date_close_claims(text, anchor_year=2026)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-05-08"
    assert claims[0].price == 206.50
    assert claims[0].date_raw == "May 8"


def test_extracts_month_day_year_close():
    from tradingagents.validators import extract_date_close_claims
    text = "On May 8, 2026, COIN closed at $206.50"
    claims = extract_date_close_claims(text)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-05-08"
    assert claims[0].price == 206.50


def test_extracts_coin_2026_05_08_fabrication_pattern_explicit():
    """Regression: the exact prose pattern that fabricated COIN $206.50 in
    decision.md / decision_executive.md / technicals_v2.md."""
    from tradingagents.validators import extract_date_close_claims
    # From actual COIN 2026-05-08 decision.md after Fix #10 (TA v2 still drifted)
    text = (
        "**Data freshness:** 10-Q for Q1 2026 (period ending 2026-03-31) filed "
        "2026-05-07; technicals through the 2026-05-08 close (per prices.json: "
        "open $205.31, high $210.47, low $202.81, close $206.50, volume "
        "14,390,000); news through 2026-05-07."
    )
    claims = extract_date_close_claims(text)
    # Should extract at least the 2026-05-08 close $206.50 claim
    iso_dates = {c.date_iso for c in claims}
    assert "2026-05-08" in iso_dates
    # The $206.50 specifically should appear in the extracted prices
    prices = {c.price for c in claims}
    assert 206.50 in prices


def test_skips_attribution_pattern_with_close_before_date():
    """Edge case: the LLM sometimes writes `*"... closed at $X.XX ... after
    the filing on May 7"*` — close BEFORE date, with the date acting as a
    separate referent. This is too ambiguous to extract reliably without
    false positives, so the extractor skips it. The Phase 6.9 deterministic
    block + the canonical date+close patterns cover the load-bearing
    fabrications; this fuzzy shape is out of scope for Phase 7.1."""
    from tradingagents.validators import extract_date_close_claims
    text = (
        '**Revision 1 — Market Analyst:** *"COIN closed the session at '
        '$206.50 on 14.39M shares — roughly 1.8–2x the trailing daily '
        'average — after the 10-Q filing on May 7."*'
    )
    claims = extract_date_close_claims(text)
    # No claim should be produced (close precedes the date contextually)
    assert claims == []


def test_returns_empty_on_empty_text():
    from tradingagents.validators import extract_date_close_claims
    assert extract_date_close_claims("") == []
    assert extract_date_close_claims(None) == []  # type: ignore[arg-type]


def test_handles_thousands_separator_in_price():
    """Plain prose with no period in the bridge: thousands-separator
    parsing must work on prices like $1,054,500.00."""
    from tradingagents.validators import extract_date_close_claims
    text = "On 2026-05-08 BRK closed at $1,054,500.00"  # no '.' in bridge
    claims = extract_date_close_claims(text)
    assert len(claims) >= 1
    assert claims[0].price == 1_054_500.00


def test_skips_when_period_in_bridge_to_avoid_sentence_spanning():
    """Sentence-boundary defense: when a `.` appears between date and
    close (e.g., ticker `BRK.A`, abbreviations, or actual sentence
    breaks), the extractor refuses to pair them. Trade-off: loses some
    valid `BRK.A`-like cases but prevents false positives across sentence
    boundaries (a far more common concern)."""
    from tradingagents.validators import extract_date_close_claims
    text = "On 2026-05-08 we trimmed. The next session closed at $200.00"
    claims = extract_date_close_claims(text)
    # No claim — the period ended the relationship between 2026-05-08 and the close
    assert claims == []


def test_skips_unparseable_dates():
    """Patterns like 'last Monday close $200' don't have resolvable dates;
    extractor should yield nothing rather than crashing."""
    from tradingagents.validators import extract_date_close_claims
    text = "Last Monday's session closed at $200.00 — defensible support."
    claims = extract_date_close_claims(text)
    # No ISO date should be resolvable; depending on regex specificity, the
    # extractor may yield 0 or some claim with date_iso=None. Either is
    # acceptable. We just want no crash and no bogus dates.
    for c in claims:
        if c.date_iso is not None:
            # If something WAS extracted, it must be a valid ISO date
            assert len(c.date_iso) == 10


def test_line_numbers_are_1_indexed():
    from tradingagents.validators import extract_date_close_claims
    text = "Header\n\n\nOn 2026-05-08 closed at $200.00\n"
    claims = extract_date_close_claims(text)
    assert len(claims) == 1
    assert claims[0].line_no == 4  # header + blank + blank + claim


def test_handles_multiple_claims_in_order():
    from tradingagents.validators import extract_date_close_claims
    text = (
        "On 2026-05-05 closed at $197.75. The next session, 2026-05-06, "
        "closed at $197.96. And on 2026-05-07 the close was $192.96."
    )
    claims = extract_date_close_claims(text)
    iso_dates = [c.date_iso for c in claims]
    # All three dates extracted
    assert "2026-05-05" in iso_dates
    assert "2026-05-06" in iso_dates
    assert "2026-05-07" in iso_dates


def test_v2_inner_date_in_parens_overrides_outer_date():
    """Phase 7.1 v2 (Fix #11): when bridge contains a date in parens, the
    close binds to the INNER date, not the outer.

    MSFT 2026-05-08 false positive:
      "Jan 7, 2026 peak before the FY26 Q2 earnings crash
       (Jan 29, 2026: close $432.51 on 128.9M shares)"

    Pre-fix: bound Jan 7, 2026 ↔ $432.51 (wrong; Jan 7 close was $482.37).
    Post-fix: binds Jan 29, 2026 ↔ $432.51 (correct)."""
    from tradingagents.validators import extract_date_close_claims

    text = (
        "Jan 7, 2026 peak before the FY26 Q2 earnings crash "
        "(Jan 29, 2026: close $432.51 on 128.9M shares)"
    )
    claims = extract_date_close_claims(text)
    assert len(claims) == 1
    # Inner date (Jan 29) wins over outer (Jan 7)
    assert claims[0].date_iso == "2026-01-29"
    assert claims[0].price == 432.51


def test_v2_inner_date_after_semicolon_overrides_outer_date():
    """MSFT 2026-05-08 false positive #2:
      "intraday high range of May 7 session; Apr 22 close at $432.92"

    Pre-fix: bound May 7 ↔ $432.92 (wrong).
    Post-fix: binds Apr 22 ↔ $432.92 (correct)."""
    from tradingagents.validators import extract_date_close_claims

    text = "intraday high range of May 7 session; Apr 22 close at $432.92"
    claims = extract_date_close_claims(text, anchor_year=2026)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-04-22"
    assert claims[0].price == 432.92


def test_v2_keeps_real_fabrication_when_no_inner_date_in_bridge():
    """The 2 REAL MSFT fabrications must still fire after the v2 fix:
      "The May 8 close at $434.84 on 23.5M shares"

    Bridge between May 8 and the close contains no other date, so the
    outer date stands. The validator still flags this as
    fabricated_future_close."""
    from tradingagents.validators import extract_date_close_claims

    text = "The May 8 close at $434.84 on 23.5M shares is constructive."
    claims = extract_date_close_claims(text, anchor_year=2026)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-05-08"  # outer date stands
    assert claims[0].price == 434.84


def test_v2_real_coin_fabrication_still_caught():
    """COIN regression — fix must not break Phase 7.1's load-bearing case."""
    from tradingagents.validators import extract_date_close_claims

    text = "the May 8 session subsequently closed at $206.50 per prices.json"
    claims = extract_date_close_claims(text, anchor_year=2026)
    assert len(claims) == 1
    assert claims[0].date_iso == "2026-05-08"
    assert claims[0].price == 206.50


def test_match_text_includes_surrounding_context():
    """Match_text should include enough surrounding prose for human review
    of the violation (typically 100-150 chars)."""
    from tradingagents.validators import extract_date_close_claims
    text = "After the print on 2026-05-08, COIN closed at $206.50 — buyers stepped in."
    claims = extract_date_close_claims(text)
    assert len(claims) == 1
    # Match text should include both the trigger context and what follows
    assert "2026-05-08" in claims[0].match_text
    assert "buyers" in claims[0].match_text or "$206.50" in claims[0].match_text
