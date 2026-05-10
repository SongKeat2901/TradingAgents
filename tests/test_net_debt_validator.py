"""Tests for the Phase-7.5 net-debt definitional consistency validator."""
import json

import pytest

pytestmark = pytest.mark.unit


# MSFT-style net_debt.json from the 2026-05-07 audit.
# yfinance Net Debt $8.16B (incl. capital leases); other defensible
# derivations: Total Debt − Cash = $X.XXB; (LTD+CD) − Cash+STI = ?, etc.
_MSFT_NET_DEBT_JSON = {
    "trade_date": "2026-05-07",
    "as_of_quarter": "2026-03-31",
    "net_debt": 8_160_000_000.0,                  # yfinance
    "net_debt_source": "yfinance",
    "total_debt": 56_970_000_000.0,
    "long_term_debt": 39_270_000_000.0,
    "current_debt": 1_000_000_000.0,              # placeholder
    "capital_lease_obligations": 16_700_000_000.0,
    "cash_and_equivalents": 40_262_000_000.0,
    "short_term_investments": None,
    "other_short_term_investments": None,
    "cash_plus_short_term_investments": 78_230_000_000.0,
    "unavailable": False,
}


def _write_net_debt(tmp_path, data=_MSFT_NET_DEBT_JSON):
    p = tmp_path / "raw" / "net_debt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_extracts_net_debt_with_value_first():
    from tradingagents.validators import extract_net_debt_claims
    text = "Authoritative Net Debt: $8.16B (yfinance row, col 0 of balance_sheet)"
    claims = extract_net_debt_claims(text)
    assert len(claims) == 1
    assert claims[0].is_cash is False
    assert claims[0].value_dollars == 8_160_000_000.0


def test_extracts_net_cash_with_value_first():
    from tradingagents.validators import extract_net_debt_claims
    text = "$38.0B cash-only net cash position"
    claims = extract_net_debt_claims(text)
    assert len(claims) == 1
    assert claims[0].is_cash is True
    assert claims[0].value_dollars == 38_000_000_000.0


def test_extracts_label_first_pattern():
    from tradingagents.validators import extract_net_debt_claims
    text = "net cash of $21.3B (Total Debt − Cash+STI)"
    claims = extract_net_debt_claims(text)
    assert len(claims) >= 1
    assert claims[0].is_cash is True
    assert claims[0].value_dollars == 21_300_000_000.0


def test_validates_yfinance_net_debt_passes(tmp_path):
    """Canonical yfinance Net Debt $8.16B must pass against MSFT cells."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    text = "Authoritative Net Debt: $8.16B"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="pm_brief.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path)
    assert violations == []


def test_validates_total_debt_minus_cash_passes(tmp_path):
    """`Total Debt − Cash` derivation: $56.97B − $40.26B = $16.71B."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    text = "Total Debt $56.97B − Cash $40.26B = $16.71B net debt"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="analyst.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path)
    # All derivations defensible: $16.71B = Total Debt − Cash → PASS
    drift_violations = [v for v in violations if v.type == "definitional_drift"]
    # One of the extracted should be the $16.71B claim — must NOT be flagged
    assert not any(v.claimed_dollars == 16_710_000_000.0 for v in drift_violations)


def test_flags_fabricated_net_debt(tmp_path):
    """A claim that doesn't derive from any cell within tolerance is
    flagged as definitional drift."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    # $190B is not derivable from MSFT cells (max would be ~Total Debt $57B
    # or Cash+STI $78B — neither is close to $190B).
    text = "Bear thesis: $190B in net debt drains FCF"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="decision.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "MATERIAL"
    assert v.type == "definitional_drift"
    assert v.claimed_dollars == 190_000_000_000.0
    assert v.delta_dollars > 100_000_000_000.0  # huge gap


def test_skips_when_net_debt_data_unavailable(tmp_path):
    """If net_debt.json marks the ticker unavailable, can't validate."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path, data={"unavailable": True, "reason": "missing rows"})

    text = "Net Debt: $190B (fabricated)"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="decision.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path)
    # Can't validate without cells; return empty (not a violation)
    assert violations == []


def test_emits_no_data_violation_when_file_missing(tmp_path):
    from tradingagents.validators import validate_net_debt_claims
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    claims = [NetDebtClaim(
        label="net debt", is_cash=False, value_raw="$100B",
        value_dollars=100_000_000_000.0, file="decision.md",
        line_no=1, match_text="$100B net debt",
    )]
    violations = validate_net_debt_claims(claims, tmp_path / "raw" / "net_debt.json")
    assert len(violations) == 1
    assert violations[0].type == "no_net_debt_data"
    assert violations[0].severity == "MINOR"


def test_render_violations_pass_message():
    from tradingagents.validators.net_debt_validator import render_net_debt_violations_text
    out = render_net_debt_violations_text([])
    assert "PASS" in out
    assert "0 violations" in out


def test_skips_peer_attributed_claim_when_main_ticker_known(tmp_path):
    """v1.1 false-positive fix: 'AMZN's net debt of $17.26B' in a MSFT report
    must NOT be flagged — that's a peer claim and should be validated by
    Phase 7.3 against peer_ratios.json, not against MSFT's net_debt.json."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    text = "AMZN's net debt of $17.26B, while modest versus EBITDA"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="analyst.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    # When main_ticker is provided as MSFT, peer claim is skipped
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    assert violations == []


def test_validates_main_ticker_claim_even_when_main_ticker_known(tmp_path):
    """Main-ticker prefix must NOT cause skipping — `MSFT's net debt $X` in
    an MSFT report still gets validated."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    # Genuinely fabricated claim attributed to MSFT
    text = "MSFT's net debt is $190B (fabricated)"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="decision.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    # MSFT-attributed $190B should still be flagged as drift
    assert len(violations) == 1
    assert violations[0].claimed_dollars == 190_000_000_000.0


def test_filters_common_uppercase_non_tickers():
    """Common uppercase tokens like 'EBITDA' / 'GAAP' / 'TTM' should NOT
    be mistaken for peer ticker prefixes."""
    from tradingagents.validators.net_debt_validator import (
        _claim_attributed_to_other_ticker,
    )
    # These should all return False (no foreign ticker found)
    assert not _claim_attributed_to_other_ticker(
        "EBITDA-based net debt $100B", main_ticker="MSFT"
    )
    assert not _claim_attributed_to_other_ticker(
        "TTM net debt $50B", main_ticker="MSFT"
    )
    # But a real peer ticker should be detected
    assert _claim_attributed_to_other_ticker(
        "AMZN's net debt of $17B", main_ticker="MSFT"
    )


def test_does_not_pair_label_to_dollar_across_semicolon(tmp_path):
    """v1.1 false-positive fix: `"net cash" and stops; the data shows
    $16.70B of lease obligations` should NOT pair "net cash" with $16.70B
    — the semicolon ends the phrase association."""
    from tradingagents.validators import extract_net_debt_claims
    text = (
        "are $78.23B. The crowd sees \"net cash\" and stops; the data shows "
        "that $16.70B of lease obligations are the offsetting concern"
    )
    claims = extract_net_debt_claims(text)
    # No claim should be extracted — "net cash" and "$16.70B" are
    # separated by a semicolon, breaking the phrase association.
    assert claims == []


def test_v1_2_peer_attribution_detection_uses_full_line(tmp_path):
    """Phase 7.5 v1.2 false-positive fix: when a peer ticker prefix appears
    earlier in the SAME LINE (not just within 30 chars), peer-attribution
    detection must still skip the claim. Surfaced by the 2026-05-08 AAOI
    run where 'FN trades at 36.6x forward with $956M in organically-
    generated net cash' had FN ~37 chars before $956M — outside the prior
    lookback window."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    text = (
        "Meanwhile FN trades at 36.6x forward with $956M in organically-"
        "generated net cash and positive FCF every quarter."
    )
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="debate_risk.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    # Validator with main_ticker="MSFT" should detect FN in the line and skip
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    assert violations == []  # FN-attributed, deferred to Phase 7.3


def test_v1_2_peer_attribution_with_pm_brief_reference(tmp_path):
    """Same v1.2 fix, second variant: 'FN carries zero net debt
    (ND/EBITDA = −1.99×, net debt −$956M per pm_brief.md)' — peer ticker
    is at the start of a parenthetical sentence."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    nd_path = _write_net_debt(tmp_path)

    text = (
        "FN carries zero net debt (ND/EBITDA = -1.99x, net debt -$956M "
        "per pm_brief.md) vs. AAOI's $42M net debt position."
    )
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="analyst_news.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    # FN's $956M and AAOI's $42M both appear; both have peer ticker context
    # (FN before $956M; AAOI's before $42M is the main ticker so wouldn't
    # be skipped — but $42M is far below all canonical MSFT derivations
    # so it would still flag if we treated AAOI as the ticker. With
    # main_ticker=MSFT, both should defer.)
    assert violations == []


def test_v1_5_recognizes_peer_ticker_with_colon_delimiter(tmp_path):
    """RMBS 2026-05-08 false positive: a peer-comparison bullet line

      "- **MRVL: net debt of $1.83B** (ND/EBITDA 0.70x on $2.63B EBITDA ...

    was flagged as definitional drift on RMBS net debt. The peer-attribution
    detector `_claim_attributed_to_other_ticker` uses regex
    `\\b[A-Z]{2,5}(?:'s|\\s)` which doesn't match 'MRVL:' (colon-delimited
    table-row form). Add `:` as a recognized delimiter."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    rmbs_data = {
        "trade_date": "2026-05-08",
        "as_of_quarter": "2026-03-31",
        "financial_currency": "USD",
        "net_debt": -762_735_000.0,
        "total_debt": 23_404_000.0,
        "cash_and_equivalents": 134_324_000.0,
        "cash_plus_short_term_investments": 786_139_000.0,
        "unavailable": False,
    }
    nd_path = _write_net_debt(tmp_path, data=rmbs_data)

    text = (
        "- **MRVL: net debt of $1.83B** (ND/EBITDA 0.70x on $2.63B "
        "EBITDA — moderate; manageable at current rate of operating cash "
        "generation)"
    )
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="analyst_fundamentals.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="RMBS")
    # MRVL is the peer; the claim should be deferred to Phase 7.3.
    assert violations == [], (
        f"MRVL-attributed claim flagged as RMBS drift: "
        f"{[(v.severity, v.match_text[:60]) for v in violations]}"
    )


def test_v1_6_does_not_pair_label_to_value_across_slash_ratio_operator():
    """NVDA 2026-05-08 false positive: a ratio expression

      "ND/EBITDA: NVDA computed as (-$51.52B net cash) / $133.2B TTM EBITDA = -0.39x"

    was extracted as "$133.2B net cash" because `_PATTERN_LABEL_FIRST` bridge
    `[^\\n.;|]{0,20}?` allowed `) / ` between 'net cash' and `$133.2B`. The
    `/` is the ratio division operator — the value AFTER it is the
    denominator (TTM EBITDA), not a continuation of the 'net cash' label.
    Add `/` to the bridge exclusion."""
    from tradingagents.validators import extract_net_debt_claims

    text = (
        "4. ND/EBITDA: NVDA computed as (-$51.52B net cash) / $133.2B "
        "TTM EBITDA = -0.39x; peers from pm_brief.md verbatim."
    )
    claims = extract_net_debt_claims(text)
    # Only $51.52B (real net cash claim) should be extracted; $133.2B is
    # TTM EBITDA (denominator of the ratio), not net cash.
    # Use absolute-value tolerance because float arithmetic on `133.2 * 1e9`
    # gives 133199999999.99998 (not the integer 133_200_000_000.0).
    for c in claims:
        assert abs(c.value_dollars - 133_200_000_000.0) > 1_000_000, (
            f"$133.2B (TTM EBITDA denominator) was paired with '{c.label}' "
            f"across `/`: value_dollars={c.value_dollars}, "
            f"match_text={c.match_text!r}"
        )
    # The legitimate $51.52B net cash claim should still be extracted
    assert any(
        abs(c.value_dollars - 51_520_000_000.0) < 1_000_000 for c in claims
    ), "the legitimate $51.52B net cash claim was not extracted"


def test_v1_4_does_not_pair_value_to_label_across_semicolon():
    """AAPL 2026-05-08 false positive: a self-citing claim contained

      "**AAPL net debt of $39.14B** (computed as Total Debt $84.71B
       − Cash $45.57B; source: yfinance Net Debt row, pm_brief.md)"

    The regex `_PATTERN_VALUE_FIRST` paired $45.57B (Cash) with "Net Debt"
    in the source citation 30 chars later, bridging across `;`. The
    sibling `_PATTERN_LABEL_FIRST` already excluded `;`; this fix brings
    `_PATTERN_VALUE_FIRST` symmetric."""
    from tradingagents.validators import extract_net_debt_claims

    text = (
        "**AAPL net debt of $39.14B** (computed as Total Debt $84.71B "
        "− Cash $45.57B; source: yfinance Net Debt row, pm_brief.md)"
    )
    claims = extract_net_debt_claims(text)
    # Only the AAPL net debt of $39.14B should be extracted; $45.57B
    # (Cash) and $84.71B (Total Debt) are not net-debt claims.
    values = {c.value_dollars for c in claims}
    assert 45_570_000_000.0 not in values, (
        "$45.57B (Cash) was paired with 'Net Debt' across semicolon"
    )
    assert 84_710_000_000.0 not in values, (
        "$84.71B (Total Debt) was paired with 'Net Debt' across semicolon"
    )


def test_v1_3_does_not_pair_label_to_dollar_across_table_cell():
    """ASX 2026-05-08 false positive: a markdown table row contained

      "| FCF deficit widens to NT$190B+ projected net debt | $27.52 (Fib 38.2%) |"

    The `|` is a markdown table cell separator. `$27.52` is the next cell's
    Fib level — NOT a net-debt value. Pre-fix: extractor paired
    `net debt | $27.52` because `|` was inside the 20-char bridge.
    Post-fix: `|` excluded from the bridge char class."""
    from tradingagents.validators import extract_net_debt_claims

    text = (
        "| Bear | $30.08 break on >=9M shares; FCF deficit widens to "
        "NT$190B+ projected net debt | $27.52 (Fib 38.2%) | ~22% |"
    )
    claims = extract_net_debt_claims(text)
    # No claim should pair $27.52 with "net debt" across the | boundary.
    for c in claims:
        assert c.value_dollars != 27_520_000_000.0  # $27.52B
        assert c.value_dollars != 27_520_000.0       # $27.52M
        assert c.value_raw != "$27.52"


def test_v1_3_skips_non_usd_currency_prefix_NTD():
    """ASX 2026-05-08 false positive: `NT$190B+ projected net debt`
    extracted as $190 USD. The validator's canonical is USD-only, so
    non-USD prefixed values must be skipped (out of scope)."""
    from tradingagents.validators import extract_net_debt_claims

    text = "projected net debt drifts toward NT$190B by year-end."
    claims = extract_net_debt_claims(text)
    # The NT$ prefix marks this as TWD, not USD — extractor should skip.
    assert claims == []


def test_v1_3_skips_non_usd_currency_prefix_other():
    """Symmetric coverage for other foreign currency markers."""
    from tradingagents.validators import extract_net_debt_claims

    for prefix in ("NT$", "JP¥", "€", "£", "¥", "C$", "A$", "HK$", "S$"):
        text = f"net debt of {prefix}5.3B per the latest 10-Q"
        claims = extract_net_debt_claims(text)
        assert claims == [], (
            f"non-USD prefix {prefix!r} produced claims: "
            f"{[(c.value_raw, c.match_text) for c in claims]}"
        )


def test_v1_3_skips_validation_when_reporting_currency_non_usd(tmp_path):
    """ASX (Taiwan-domiciled): yfinance balance-sheet cells are reported
    in TWD. Storing them as if USD makes the canonical $169B (5x ASX's
    actual $25B market cap). Phase 7.5 v1.3: when net_debt.json carries
    `financial_currency != "USD"`, skip validation with a single MINOR
    `skipped_non_usd_reporter` notice and do not flag any claims."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    twd_data = dict(_MSFT_NET_DEBT_JSON)
    twd_data["financial_currency"] = "TWD"
    nd_path = _write_net_debt(tmp_path, data=twd_data)

    # USD-prefixed claim that would otherwise drift against TWD canonical
    text = "Authoritative Net Debt: $5.3B (USD-equivalent of NT$169B)."
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="decision.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]

    violations = validate_net_debt_claims(claims, nd_path, main_ticker="ASX")
    # Exactly one MINOR notice, no MATERIAL drift flags
    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "MINOR"
    assert v.type == "skipped_non_usd_reporter"


def test_v1_3_validates_normally_when_currency_usd(tmp_path):
    """Sanity check: financial_currency = USD validates as before."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    usd_data = dict(_MSFT_NET_DEBT_JSON)
    usd_data["financial_currency"] = "USD"
    nd_path = _write_net_debt(tmp_path, data=usd_data)

    text = "MSFT Authoritative Net Debt: $8.16B (yfinance row)"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="pm_brief.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]

    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    assert violations == []  # passes against canonical


def test_v1_3_validates_normally_when_currency_field_missing(tmp_path):
    """Legacy net_debt.json files (pre-7.7) lack the financial_currency
    field. Treat absence as USD (backwards compatibility) so existing
    runs and US-ticker re-runs continue to validate."""
    from tradingagents.validators import (
        extract_net_debt_claims,
        validate_net_debt_claims,
    )
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    legacy_data = dict(_MSFT_NET_DEBT_JSON)
    # Explicitly do NOT add financial_currency
    assert "financial_currency" not in legacy_data
    nd_path = _write_net_debt(tmp_path, data=legacy_data)

    text = "Authoritative Net Debt: $8.16B"
    claims = extract_net_debt_claims(text)
    claims = [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file="pm_brief.md",
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]

    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    assert violations == []  # backwards compat: missing field treated as USD


def test_v1_3_keeps_usd_dollar_after_non_usd_in_same_sentence():
    """Defensive: when both NT$ and $ appear, only the $ form should be
    extracted. ASX prose:
      `net debt drifts toward NT$185-195B (USD 5.8-6.1B) at constant FX`
    Only the USD figure is extractable; the NT$ form is skipped."""
    from tradingagents.validators import extract_net_debt_claims

    text = "projected net debt of NT$185B (or $5.85B USD) at constant FX"
    claims = extract_net_debt_claims(text)
    # The USD claim should be extracted; the NT$ one skipped.
    values = [c.value_dollars for c in claims]
    assert 5_850_000_000.0 in values
    assert 185.0 not in values  # bare $185 (no unit)
    assert 185_000_000_000.0 not in values  # $185B


def test_to_dollars_handles_billions_and_millions():
    from tradingagents.validators.net_debt_validator import _to_dollars
    assert _to_dollars("8.16", "B") == 8_160_000_000.0
    assert _to_dollars("78,272", "M") == 78_272_000_000.0
    assert _to_dollars("38.0", "B") == 38_000_000_000.0
    assert _to_dollars("not-a-number", "B") is None
