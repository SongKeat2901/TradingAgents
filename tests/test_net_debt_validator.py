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


def _with_file(claims, fname):
    """Rebuild frozen NetDebtClaim(s) with `file` set (mirrors research_validation)."""
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    return [NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file=fname,
        line_no=c.line_no, match_text=c.match_text,
    ) for c in claims]


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


def test_skips_googl_2026_05_26_net_cash_from_operations_false_positive():
    """GOOGL 2026-05-26 false positive: a fundamentals reconciliation line
    says

      "Q1 FY26: Purchases of property and equipment $35,674M ÷ Net cash
       from operations $45,790M = 0.7790 = 77.90%."

    The LABEL_FIRST pattern matched 'Net cash from operations $45,790M'
    as a net-cash POSITION claim and flagged it against canonical $49.34B
    (which is GOOGL's balance-sheet Net Cash position). But the LLM
    clearly means **OCF (operating cash flow)** — a totally different
    metric in the cashflow statement, not the balance sheet.

    Fix: negative lookahead `(?!\\s+from\\s+(?:operations|operating|ops))`
    after the 'net (cash|debt)' label group, so 'net cash from
    operations' is not matched as a net-cash claim.
    """
    from tradingagents.validators import extract_net_debt_claims
    text = (
        "Q1 FY26: Purchases of property and equipment $35,674M ÷ "
        "Net cash from operations $45,790M = 0.7790 = 77.90%. "
        "Source: 10-Q Consolidated Statements of Cash Flows. "
        "Separately, the balance sheet shows GOOGL net cash position $49.3B."
    )
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    # $45,790M (OCF) must NOT be extracted
    assert 45_790_000_000.0 not in values, (
        f"'Net cash from operations $45,790M' (OCF) must not be matched; got {values}"
    )
    # The legitimate balance-sheet $49.3B net cash should still extract
    assert 49_300_000_000.0 in values, (
        f"legitimate 'net cash position $49.3B' must still extract; got {values}"
    )


def test_skips_msft_2026_05_21_inline_subtraction_false_positive():
    """MSFT 2026-05-21 rerun false positive: the LLM showed the math
    inline as

        "Net debt (cash-only) = $40,262M − $32,105M = $8,157M (≈$8.2B)"

    The LABEL_FIRST regex paired `Net debt` with the FIRST $X after the
    `=` separator — `$40,262M` (the minuend) — and flagged it against
    canonical $8.157B as definitional drift. But the math is correct:
    $40,262M (long-term debt) − $32,105M (cash) = $8,157M (net debt),
    which matches canonical.

    Fix: extend LABEL_FIRST to optionally consume an inline-subtraction
    prefix `$A − $B =` so the captured value is the computed result,
    not the minuend. Mirror of the Phase 7.12 peer_metric inline-
    equation fix.
    """
    from tradingagents.validators import extract_net_debt_claims
    text = (
        "**Net debt (cash-only) = $40,262M − $32,105M = $8,157M (≈$8.2B)** "
        "— this report uses the cash-only definition for headline figures."
    )
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    # The $8,157M result is the actual claimed net debt — should extract.
    assert 8_157_000_000.0 in values, (
        f"should capture the inline-subtraction result $8,157M; got {values}"
    )
    # $40,262M (minuend) and $32,105M (subtrahend) must NOT be extracted
    # as net-debt magnitudes — they're components of the derivation.
    assert 40_262_000_000.0 not in values, (
        f"$40,262M minuend must not be extracted; got {values}"
    )


def test_phase_8_2_skips_mstr_positional_comparator_in_bridge():
    """MSTR 2026-05-29 false positive: "$0.06B higher than yfinance Net Debt".
    The VALUE_FIRST regex paired $0.06B with "Net Debt" via the bridge
    " higher than yfinance ". The Phase 8.1 delta-bridge regex didn't
    include "higher"/"lower"/"above"/"below"/"more"/"less" — only the
    bidirectional change words. Phase 8.2 extends to positional
    comparators."""
    from tradingagents.validators import extract_net_debt_claims
    text = "Computed: $8.26B − $2.21B = $6.05B (this is $0.06B higher than yfinance Net Debt)."
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    # $0.06B (60M) is a delta amount, not a net-debt position
    assert 60_000_000.0 not in values, (
        f"$0.06B delta higher-than must not extract; got {values}"
    )


def test_phase_8_2_skips_mstr_additive_plus_in_bridge():
    """MSTR 2026-05-29 false positive: "net debt + $10.0B preferred
    liquidation preference". The LABEL_FIRST regex paired "net debt" with
    "$10.0B" via the bridge " + " — but $10.0B is preferred-stock liability,
    not net debt itself. Phase 8.2 adds `\\s\\+\\s` to the delta-bridge
    regex so additive expressions are skipped."""
    from tradingagents.validators import extract_net_debt_claims
    text = "MSTR: $5.99B net debt + $10.0B preferred liquidation preference = $15.99B total senior obligations."
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    # $5.99B (the actual net-debt claim) should extract
    assert 5_990_000_000.0 in values, (
        f"$5.99B legitimate net-debt claim should extract; got {values}"
    )
    # $10.0B (preferred liability across `+` bridge) must NOT extract
    assert 10_000_000_000.0 not in values, (
        f"$10.0B preferred across + bridge must not extract; got {values}"
    )


def test_phase_8_2_keeps_legitimate_net_debt_of_X():
    """Defense: ensure the expanded comparator set doesn't kill legitimate
    'net debt of $X' claims that happen to use neutral words in the bridge."""
    from tradingagents.validators import extract_net_debt_claims
    text = "GOOGL's authoritative net debt of $39.44B per yfinance Net Debt row."
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    assert 39_440_000_000.0 in values, (
        f"legitimate 'net debt of $39.44B' must extract; got {values}"
    )


def test_phase_8_1_skips_avgo_increase_delta_bridge():
    """AVGO 2026-05-07 false positive: the LLM wrote

        "AVGO's Q1 FY26 net debt *increased* $2.92B sequentially..."
        "...Q1 FY26 sequential net debt *increase* of $2.92B is material."

    The LABEL_FIRST regex paired `net debt` with `$2.92B`, but $2.92B is
    the AMOUNT OF INCREASE, not the net-debt position. The bridge
    between label and value contains 'increased' / 'increase' — the
    Phase 8.1 delta-bridge guard skips when the bridge has any of
    increas|decreas|chang|swing|delta|rose|risen|fell|fallen.
    The 'sequentially' tail also triggers the tail-side guard."""
    from tradingagents.validators import extract_net_debt_claims
    text1 = "AVGO's Q1 FY26 net debt *increased* $2.92B sequentially to $51.88B."
    text2 = "Q1 FY26 sequential net debt *increase* of $2.92B is material."
    for text in (text1, text2):
        claims = extract_net_debt_claims(text)
        # $2.92B is a delta amount, not a position; must NOT extract
        assert 2_920_000_000.0 not in {c.value_dollars for c in claims}, (
            f"$2.92B delta in {text!r} must not be extracted; got {[c.value_dollars for c in claims]}"
        )
    # Sanity: $51.88B (the resulting position in text1) IS a real magnitude
    # claim and should be extractable, but here it's paired with "to" (not
    # "net debt"), so the regex won't match — that's expected. Just make
    # sure we did not extract $2.92B.


def test_phase_8_1_skips_orcl_dollar_range_low_endpoint():
    """ORCL 2026-05-07 false positive: "Net debt ≈ $5–6B". The value
    regex matches `$5` (no unit suffix — `$5` is a valid match) and the
    tail `–6B` would have been ignored. Phase 8.1 adds a tail-guard
    `[\\-–]\\s*\\d` to recognise that the value is the LOW endpoint
    of a range; skip."""
    from tradingagents.validators import extract_net_debt_claims
    text = "For peer comparability: **Net debt ≈ $5–6B** in the cited convention."
    claims = extract_net_debt_claims(text)
    # The bare $5 (range low endpoint) must NOT be extracted
    assert 5.0 not in {c.value_dollars for c in claims}, (
        f"$5 range endpoint must not be extracted; got {[c.value_dollars for c in claims]}"
    )


def test_phase_8_1_keeps_legitimate_increase_word_when_not_in_bridge():
    """Defense: only skip when the delta word is in the BRIDGE between
    label and value. A sentence like "net debt of $40,262M; the increase
    over Q4 was driven by..." should still extract $40,262M because
    'increase' is in the TAIL, not the bridge (between 'net debt' and
    '$40,262M', the bridge is `of `)."""
    from tradingagents.validators import extract_net_debt_claims
    text = "MSFT net debt of $40,262M; the increase over Q4 was driven by buybacks."
    claims = extract_net_debt_claims(text)
    assert 40_262_000_000.0 in {c.value_dollars for c in claims}, (
        f"legitimate $40,262M claim with 'increase' in the TAIL must extract; got {[c.value_dollars for c in claims]}"
    )


def test_skips_meta_2026_05_21_delta_phrase_false_positive():
    """META 2026-05-21 rerun false positive: a formula-discrepancy
    disclosure section says

        "...yields $5.59B net debt, approximately $30B lower."

    The LABEL_FIRST pattern paired `net debt` with the $30B that follows
    it across the comma+approximately bridge — but that $30B is a DELTA
    ("$30B lower") between two methodologies, not a claim that META's
    net debt is $30B. The headline figure cited in the same disclosure
    is $35.32B (the canonical yfinance Net Debt row).

    Fix: when the $X that LABEL_FIRST captures is immediately followed
    by a comparator word (lower/higher/less/more/below/above/different
    /apart/shy/short/away/over/under), skip the claim — it's a delta,
    not a magnitude.
    """
    from tradingagents.validators import extract_net_debt_claims
    text = (
        "An alternative calculation using all available liquidity — "
        "Total Debt ($86.77B) − (Cash + STI) ($81.18B) — yields "
        "$5.59B net debt, approximately $30B lower. This report uses "
        "$35.32B as the headline figure per pm_brief.md instruction."
    )
    claims = extract_net_debt_claims(text)
    values = sorted({c.value_dollars for c in claims})
    # The $5.59B claim is paired with the literal "net debt" label and is
    # a real magnitude — should still extract (and be checked against
    # canonical derivations downstream).
    assert 5_590_000_000.0 in values, (
        f"should still capture $5.59B alternative-calc claim; got {values}"
    )
    # The $30B is a DELTA ("approximately $30B lower") — must NOT be
    # extracted as a net-debt magnitude claim.
    assert 30_000_000_000.0 not in values, (
        f"$30B delta phrase must not be extracted as a net-debt claim; got {values}"
    )


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


def test_phase_8_2_peer_attribution_recognizes_markdown_bold_ticker():
    """MSTR 2026-05-29 false positive: peer-bullet `- **RIOT** — ... Net
    Debt $636M ...` was validated against MSTR canonicals because the
    PEER_TICKER_PATTERN regex required `'s`, whitespace, or `:` after the
    ticker. The trailing `**` (markdown bold close) meant RIOT was never
    matched, so the claim fell through to MSTR-subject validation and
    flagged drift vs MSTR's $2.2B Cash+STI. Phase 8.2 adds `*` and `_`
    to the delimiter alternation."""
    from tradingagents.validators.net_debt_validator import (
        _claim_attributed_to_other_ticker,
    )
    # The actual MSTR 2026-05-29 RIOT bullet shape
    msg = (
        "- **RIOT** — bitcoin miner. Authoritative raw/peers.json cells: "
        "Q1 capex/revenue **78.7%**, Q1 op margin **−72.6%**, "
        "Net Debt **$636M**, TTM EBITDA **−$327M**"
    )
    assert _claim_attributed_to_other_ticker(msg, main_ticker="MSTR"), (
        "**RIOT** with markdown-bold delimiter must be recognized as a "
        "peer-attributed claim"
    )
    # Same shape with single-asterisk italic emphasis
    assert _claim_attributed_to_other_ticker(
        "*AAPL* Net Debt $50B is the comparator.", main_ticker="MSFT"
    )
    # Note: underscore-italic `_AAPL_` is not supported because `_` is a
    # word character in regex (no \b boundary between `_` and `A`); the
    # report style consistently uses asterisks (**TICKER**), so this
    # edge case is documented but not handled.


def test_phase_8_2_peer_attribution_still_returns_false_for_main_ticker_in_bold():
    """Defense: if the subject ticker itself appears in markdown bold
    (e.g., `**MSTR** Net Debt $5.99B`), it must NOT be flagged as
    peer-attributed. Only mentions of OTHER tickers should trip."""
    from tradingagents.validators.net_debt_validator import (
        _claim_attributed_to_other_ticker,
    )
    assert not _claim_attributed_to_other_ticker(
        "**MSTR** Net Debt $5.99B per yfinance row", main_ticker="MSTR"
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


def test_component_cell_near_label_not_flagged(tmp_path):
    """Phase 9: a Cash/STI component cell cited inside a reconciliation near a
    'net debt' label must not be flagged as definitional_drift (INTC FP)."""
    from tradingagents.validators.net_debt_validator import (
        extract_net_debt_claims, validate_net_debt_claims,
    )
    import json as _json
    nd = {"net_debt": 27_780_000_000, "total_debt": 45_030_000_000,
          "cash_and_equivalents": 17_250_000_000,
          "cash_plus_short_term_investments": 32_790_000_000,
          "long_term_debt": 40_000_000_000, "current_debt": 5_030_000_000,
          "capital_lease_obligations": 0}
    p = tmp_path / "net_debt.json"; p.write_text(_json.dumps(nd))
    # "$17.25B" (Cash) sits right before a "net cash" label in a reconciliation
    text = "Net debt $27.78B; the $17.25B cash and $15.54B short-term investments net cash bridge."
    claims = extract_net_debt_claims(text)
    for c in claims:
        c2 = type(c)(**{**c.__dict__, "file": "decision.md"})
    claims = [type(c)(**{**c.__dict__, "file": "decision.md"}) for c in claims]
    v = validate_net_debt_claims(claims, p)
    mats = [x for x in v if x.severity == "MATERIAL"]
    assert mats == [], [(x.claimed_value, x.match_text) for x in mats]


def test_net_debt_raise_flow_not_flagged():
    """Phase 9: '$29.9B net debt raise' is a debt issuance flow, not a position."""
    from tradingagents.validators.net_debt_validator import extract_net_debt_claims
    claims = extract_net_debt_claims("Q1 saw a $29.9B net debt raise to fund capex.")
    assert claims == [], [c.value_raw for c in claims]


def test_ebitda_denominator_after_division_glyph_not_bound():
    """Phase 9: '$96.15B net debt ÷ $27.44B TTM EBITDA' must not bind the
    EBITDA denominator to 'net debt' (÷/× are ratio operators like /)."""
    from tradingagents.validators.net_debt_validator import extract_net_debt_claims
    c = extract_net_debt_claims("Leverage: $96.15B net debt ÷ $27.44B TTM EBITDA = 3.50x.")
    assert all(x.value_raw != "$27.44B" for x in c), [(x.value_raw, x.label) for x in c]
    c2 = extract_net_debt_claims("$5.0B net debt × 2 ignored $99.9B EBITDA")
    assert all(x.value_raw != "$99.9B" for x in c2)


# ---------------------------------------------------------------------------
# Phase 10 — wk25 FP guards: off-balance-sheet, peer (1-letter), pro-forma,
# historical (from→to FROM side).
# ---------------------------------------------------------------------------

def _write_orcl_net_debt(tmp_path):
    """ORCL-like net_debt.json: yfinance net_debt $96.15B."""
    data = {
        "trade_date": "2026-06-05",
        "as_of_quarter": "2026-02-28",
        "financial_currency": "USD",
        "net_debt": 96_150_000_000.0,
        "total_debt": 88_470_000_000.0,
        "long_term_debt": 85_470_000_000.0,
        "current_debt": 3_000_000_000.0,
        "capital_lease_obligations": 8_000_000_000.0,
        "cash_and_equivalents": 11_680_000_000.0,
        "short_term_investments": None,
        "cash_plus_short_term_investments": 11_680_000_000.0,
        "unavailable": False,
    }
    p = tmp_path / "raw" / "net_debt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_txn_net_debt(tmp_path):
    """TXN-like net_debt.json: yfinance net_debt $10.50B."""
    data = {
        "trade_date": "2026-06-05",
        "as_of_quarter": "2026-03-31",
        "financial_currency": "USD",
        "net_debt": 10_500_000_000.0,
        "total_debt": 13_800_000_000.0,
        "long_term_debt": 12_800_000_000.0,
        "current_debt": 1_000_000_000.0,
        "capital_lease_obligations": 500_000_000.0,
        "cash_and_equivalents": 3_300_000_000.0,
        "short_term_investments": None,
        "cash_plus_short_term_investments": 3_300_000_000.0,
        "unavailable": False,
    }
    p = tmp_path / "raw" / "net_debt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_googl_net_debt(tmp_path):
    """GOOGL-like net_debt.json: yfinance net_debt (negative = net cash) -$39.44B."""
    data = {
        "trade_date": "2026-06-05",
        "as_of_quarter": "2026-03-31",
        "financial_currency": "USD",
        "net_debt": -39_440_000_000.0,
        "total_debt": 28_950_000_000.0,
        "long_term_debt": 14_760_000_000.0,
        "current_debt": 14_190_000_000.0,
        "capital_lease_obligations": 0.0,
        "cash_and_equivalents": 30_760_000_000.0,
        "short_term_investments": None,
        "cash_plus_short_term_investments": 68_390_000_000.0,
        "unavailable": False,
    }
    p = tmp_path / "raw" / "net_debt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_sats_net_debt(tmp_path):
    """SATS-like net_debt.json: yfinance net_debt $5.10B (EchoStar/SATS)."""
    data = {
        "trade_date": "2026-06-05",
        "as_of_quarter": "2026-03-31",
        "financial_currency": "USD",
        "net_debt": 5_100_000_000.0,
        "total_debt": 5_900_000_000.0,
        "long_term_debt": 5_800_000_000.0,
        "current_debt": 100_000_000.0,
        "capital_lease_obligations": 0.0,
        "cash_and_equivalents": 800_000_000.0,
        "short_term_investments": None,
        "cash_plus_short_term_investments": 800_000_000.0,
        "unavailable": False,
    }
    p = tmp_path / "raw" / "net_debt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_peers(tmp_path, tickers: list[str]):
    """Write a minimal peers.json with the given ticker keys."""
    data = {t: {"ticker": t} for t in tickers}
    p = tmp_path / "raw" / "peers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_claim(c, file="analyst_fundamentals.md"):
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    return NetDebtClaim(
        label=c.label, is_cash=c.is_cash, value_raw=c.value_raw,
        value_dollars=c.value_dollars, file=file,
        line_no=c.line_no, match_text=c.match_text,
    )


def test_wk25_orcl_off_balance_sheet_skipped(tmp_path):
    """ORCL wk25 FP: '$261B in off-balance-sheet data-center commitments' is
    NOT the subject's current net debt. The validator must skip it.
    The real net debt ($96.15B) must NOT be flagged."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd_path = _write_orcl_net_debt(tmp_path)
    _write_peers(tmp_path, ["MSFT", "AMZN", "GOOGL"])

    text = (
        "ORCL carries $96.15B net debt and $261B in off-balance-sheet "
        "data-center commitments that do not appear on the balance sheet."
    )
    claims = [_make_claim(c) for c in extract_net_debt_claims(text)]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="ORCL")

    # $261B off-balance-sheet must be skipped (no definitional_drift violation for it)
    drift = [v for v in violations if v.type == "definitional_drift"]
    flagged_values = {v.claimed_dollars for v in drift}
    assert 261_000_000_000.0 not in flagged_values, (
        f"$261B (off-balance-sheet commitment) wrongly flagged as drift: {drift}"
    )
    # $96.15B is the real net debt and must also not be flagged
    assert 96_150_000_000.0 not in flagged_values, (
        f"$96.15B (correct net debt) should pass validation"
    )


def test_wk25_sats_peer_one_letter_ticker_skipped(tmp_path):
    """SATS wk25 FP: 'vs T ($117.87B net debt, 2.65x ND/EBITDA)' — AT&T's
    single-letter ticker 'T' is in peers.json but the existing PEER_TICKER_PATTERN
    requires 2-5 chars so it would NOT catch T. The Phase 10 guard uses the
    peers.json list directly to detect 1-letter tickers."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd_path = _write_sats_net_debt(tmp_path)
    # T (AT&T) is in SATS's peer list
    _write_peers(tmp_path, ["T", "DISH", "VZ", "TMUS"])

    text = (
        "vs T ($117.87B net debt, 2.65x ND/EBITDA): AT&T's spectrum "
        "counterparty; SATS net debt $5.10B."
    )
    claims = [_make_claim(c) for c in extract_net_debt_claims(text)]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="SATS")

    drift = [v for v in violations if v.type == "definitional_drift"]
    flagged_values = {v.claimed_dollars for v in drift}
    assert 117_870_000_000.0 not in flagged_values, (
        f"$117.87B (AT&T peer T's net debt) wrongly flagged as SATS drift: {drift}"
    )
    # SATS's own $5.10B must pass clean
    assert 5_100_000_000.0 not in flagged_values


def test_wk25_txn_pro_forma_forward_skipped(tmp_path):
    """TXN wk25 FP: 'Silicon Labs ~$16.5B pro-forma net debt … is a forward
    H1 2027 estimate, not a current cell (current authoritative Net Debt $10.50B)'
    — $16.5B is a labeled pro-forma forward estimate; current $10.50B matches
    canonical."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd_path = _write_txn_net_debt(tmp_path)
    _write_peers(tmp_path, ["MCHP", "ON", "STM", "SLAB"])

    text = (
        "Silicon Labs ~$16.5B pro-forma net debt post-acquisition is a "
        "forward H1 2027 estimate, not a current cell "
        "(current authoritative Net Debt $10.50B per pm_brief.md)."
    )
    claims = [_make_claim(c) for c in extract_net_debt_claims(text)]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="TXN")

    drift = [v for v in violations if v.type == "definitional_drift"]
    flagged_values = {v.claimed_dollars for v in drift}
    assert 16_500_000_000.0 not in flagged_values, (
        f"$16.5B pro-forma forward estimate wrongly flagged as drift: {drift}"
    )
    # Current $10.50B must also pass
    assert 10_500_000_000.0 not in flagged_values


def test_wk25_googl_historical_from_side_skipped(tmp_path):
    """GOOGL wk25 FP: 'Before the Wiz transaction … net debt (yfinance)
    from $15.84B to $39.44B' — $15.84B is the labeled pre-event historical
    figure (FROM side of a from→to range); current is $39.44B.
    The FROM figure must be skipped; current $39.44B must not be flagged."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd_path = _write_googl_net_debt(tmp_path)
    _write_peers(tmp_path, ["MSFT", "META", "AMZN"])

    # "from $15.84B to $39.44B" — bridge between 'net debt' and '$15.84B'
    # is 'from', which is NOT in the existing delta-bridge guard. The Phase
    # 10 historical guard must catch it via "from $X to" pattern in context.
    text = (
        "Before the Wiz transaction closed, GOOGL net debt (yfinance) "
        "from $15.84B to $39.44B — reflecting the acquisition financing."
    )
    claims = [_make_claim(c) for c in extract_net_debt_claims(text)]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="GOOGL")

    drift = [v for v in violations if v.type == "definitional_drift"]
    flagged_values = {v.claimed_dollars for v in drift}
    assert 15_840_000_000.0 not in flagged_values, (
        f"$15.84B pre-event historical (FROM side) wrongly flagged: {drift}"
    )
    # $39.44B is the current figure and must match canonical (net cash ~$39.44B abs)
    assert 39_440_000_000.0 not in flagged_values


def test_wk25_regression_unqualified_fabrication_still_flagged(tmp_path):
    """Regression: an unqualified wrong net-debt claim must STILL be flagged.
    No off-balance-sheet / peer / pro-forma / historical qualifier → drift."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    from tradingagents.validators.net_debt_validator import NetDebtClaim

    # Use MSFT canonical ($8.16B). Claim $50B — huge fabrication, no qualifier.
    nd_path = _write_net_debt(tmp_path)
    _write_peers(tmp_path, ["AMZN", "GOOGL", "AAPL"])

    text = "Net debt is $50B."
    claims = [_make_claim(c) for c in extract_net_debt_claims(text)]
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")

    drift = [v for v in violations if v.type == "definitional_drift"]
    assert len(drift) == 1, f"unqualified $50B fabrication must be flagged; got {drift}"
    assert drift[0].claimed_dollars == 50_000_000_000.0


# ---------------------------------------------------------------------------
# Phase 10 adversarial narrowing — false-negative (FN) tests.
# The original over-broad guards suppressed GENUINE fabrications.
# These 6 cases must FLAG as definitional_drift after the narrowing.
# Canonical: MSFT $8.16B. Fabricated figures are $80B or $99B.
# ---------------------------------------------------------------------------

def _fn_setup(tmp_path):
    """Helper: write MSFT net_debt.json + generic peer list, return nd_path."""
    nd_path = _write_net_debt(tmp_path)
    _write_peers(tmp_path, ["AMZN", "GOOGL", "AAPL"])
    return nd_path


def _fn_claims(text, file="analyst_fundamentals.md"):
    from tradingagents.validators import extract_net_debt_claims
    return [_make_claim(c, file=file) for c in extract_net_debt_claims(text)]


def test_fn_was_dollar_plain_reporting_flags(tmp_path):
    """Guard 3 narrowing: 'was $' was too broad — it also matched current-
    reporting prose like 'Net debt was $99B per yfinance'.  After dropping
    bare 'was $', this fabrication must now flag.
    Note: use $99B, not $80B — $80B is within 5% of MSFT's Cash+STI $78.23B
    and would pass the canonical check regardless of guard behaviour."""
    from tradingagents.validators import validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)
    text = "Net debt was $99B per yfinance."
    claims = _fn_claims(text)
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    drift = [v for v in violations if v.type == "definitional_drift"]
    assert drift, (
        "'Net debt was $99B per yfinance' was wrongly skipped; "
        "it must flag as definitional_drift (canonical ~$8B)"
    )
    assert any(v.claimed_dollars == 99_000_000_000.0 for v in drift), drift


def test_fn_bare_forward_prose_flags(tmp_path):
    """Guard 2 narrowing: standalone 'forward' is ubiquitous analyst prose
    ('Looking forward, net debt of $99B remains a concern') and must NOT
    suppress a genuine fabrication after removing bare 'forward'."""
    from tradingagents.validators import validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)
    text = "Looking forward, net debt of $99B remains a concern."
    claims = _fn_claims(text)
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    drift = [v for v in violations if v.type == "definitional_drift"]
    assert drift, (
        "'Looking forward, net debt of $99B' was wrongly skipped; "
        "standalone 'forward' must not suppress a fabrication"
    )
    assert any(v.claimed_dollars == 99_000_000_000.0 for v in drift), drift


def test_fn_bare_estimate_prose_flags(tmp_path):
    """Guard 2 narrowing: bare 'estimate' / 'estimated' is common analyst
    prose ('Analysts estimate $99B net debt for the current quarter') and
    must not suppress a genuine fabrication after removing bare 'estimate'."""
    from tradingagents.validators import validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)
    text = "Analysts estimate $99B net debt for the current quarter."
    claims = _fn_claims(text)
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    drift = [v for v in violations if v.type == "definitional_drift"]
    assert drift, (
        "'Analysts estimate $99B net debt' was wrongly skipped; "
        "bare 'estimate' must not suppress a fabrication"
    )
    assert any(v.claimed_dollars == 99_000_000_000.0 for v in drift), drift


def test_fn_bare_before_non_event_flags(tmp_path):
    """Guard 3 narrowing: bare 'before' is too broad — 'Net debt before
    interest is $99B' has nothing to do with a pre-acquisition historical
    figure.  After restricting 'before' to event-nouns only, this must flag.
    Note: use $99B — $80B is within 5% of MSFT's Cash+STI $78.23B and would
    pass the canonical check regardless of guard behaviour."""
    from tradingagents.validators import validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)
    text = "Net debt before interest is $99B."
    claims = _fn_claims(text)
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    drift = [v for v in violations if v.type == "definitional_drift"]
    assert drift, (
        "'Net debt before interest is $99B' was wrongly skipped; "
        "'before interest' is not an event-noun qualifier"
    )
    assert any(v.claimed_dollars == 99_000_000_000.0 for v in drift), drift


def test_fn_rises_to_offering_flags(tmp_path):
    """Guard 2 narrowing: bare 'rises to' was too broad — 'Net debt rises
    to $99B following last month's offering' describes a current balance-
    sheet move, not a conditional scenario. After removing bare 'rises? to'
    (keeping only 'would rise'), this must flag."""
    from tradingagents.validators import validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)
    text = "Net debt rises to $99B following last month's offering."
    claims = _fn_claims(text)
    violations = validate_net_debt_claims(claims, nd_path, main_ticker="MSFT")
    drift = [v for v in violations if v.type == "definitional_drift"]
    assert drift, (
        "'Net debt rises to $99B following offering' was wrongly skipped; "
        "bare 'rises to' must not suppress a fabrication"
    )
    assert any(v.claimed_dollars == 99_000_000_000.0 for v in drift), drift


def test_fn_obs_precedes_value_flags_subject_figure(tmp_path):
    """Guard 1 narrowing (positional OBS): 'Net debt of $50B includes $20B
    in off-balance-sheet leases' — the $50B is the SUBJECT figure (OBS
    appears AFTER the claimed value in the window), not an OBS item itself.
    After making OBS positional, $50B must flag while an OBS-suffix figure
    like '$261B in off-balance-sheet ...' is still skipped."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd_path = _fn_setup(tmp_path)

    # Case A: OBS mention is AFTER $50B in the sentence → $50B must flag
    text_a = "Net debt of $50B includes $20B in off-balance-sheet leases."
    claims_a = _fn_claims(text_a)
    violations_a = validate_net_debt_claims(claims_a, nd_path, main_ticker="MSFT")
    drift_a = [v for v in violations_a if v.type == "definitional_drift"]
    assert drift_a, (
        "'Net debt of $50B includes $20B in off-balance-sheet leases' "
        "was wrongly skipped; $50B is the subject figure, not the OBS item"
    )
    assert any(v.claimed_dollars == 50_000_000_000.0 for v in drift_a), drift_a

    # Case B: OBS mention is AFTER the value (ORCL pattern) → value must still skip
    # Re-use the ORCL fixture: $261B appears right before 'off-balance-sheet ...'
    nd_orcl = _write_orcl_net_debt(tmp_path / "orcl_sub")
    text_b = (
        "ORCL carries $96.15B net debt and $261B in off-balance-sheet "
        "data-center commitments that do not appear on the balance sheet."
    )
    claims_b = [_make_claim(c) for c in extract_net_debt_claims(text_b)]
    violations_b = validate_net_debt_claims(claims_b, nd_orcl, main_ticker="ORCL")
    drift_b = [v for v in violations_b if v.type == "definitional_drift"]
    flagged_b = {v.claimed_dollars for v in drift_b}
    assert 261_000_000_000.0 not in flagged_b, (
        f"$261B (OBS suffix) must still be skipped after positional narrowing: {drift_b}"
    )


def test_phase_9_2_net_cash_outlay_capex_term_not_extracted():
    """ORCL 2026-07-01 false positives (5 of the run's 6 MATERIAL blockers):
    analyst_fundamentals.md quoted the 8-K Ex-99.1 supplemental table
    "Net Cash Outlay for Capital Expenditures" verbatim, and the extractor
    bound "net cash" to the surrounding dollar figures as balance-sheet
    net-cash-position claims:

      line 57:  "... $4,592 = Net Cash Outlay for Capital Expenditures ..."
      line 59:  "... financing $3.3B + net cash outlay $47.7B ..."
      line 59:  "The remaining ~$15.7B of net cash outlay is not itemized ..."
      line 166: "... net cash outlay to $47,726M rather than ..."

    "Net cash outlay" is a capex-funding term (GAAP-supplemental), never a
    net-cash position — no claim should be extracted from any of these."""
    from tradingagents.validators import extract_net_debt_claims

    lines = [
        "> Capital Expenditures $(55,663) − Other Short-Term Financing Cash "
        "Flow Related to Capital Expenditures $3,345 − Customer Prepayments "
        "with Significant Financing Component for Capital Expenditures "
        "$4,592 = Net Cash Outlay for Capital Expenditures $(47,726)",
        "i.e., **guided capex $55.7B = customer prepayments $4.6B + "
        "short-term capex-linked financing $3.3B + net cash outlay $47.7B**, "
        "of which operating cash flow contributed $31,977M for the year. "
        "The remaining ≈$15.7B of net cash outlay is not itemized by the "
        "release as capex-specific",
        "bringing the FY26 Q4 net cash outlay to $47,726M rather than the "
        "headline $55,663M",
    ]
    for text in lines:
        claims = extract_net_debt_claims(text)
        assert claims == [], (
            f"'net cash outlay' extracted as a net-cash claim: "
            f"{[(c.label, c.value_raw) for c in claims]} in {text[:70]!r}"
        )


def test_phase_9_2_regression_plain_net_cash_still_extracted():
    """Guard: the outlay lookahead must not swallow genuine net-cash claims."""
    from tradingagents.validators import extract_net_debt_claims

    claims = extract_net_debt_claims(
        "SAP carries a net cash position of $2.18B against ORCL's leverage."
    )
    assert len(claims) == 1 and claims[0].is_cash


def test_phase_9_2_bold_outlay_and_outflow_not_extracted():
    """Reviewer tightenings: word-level bold ("net cash **outlay**") and the
    sibling GAAP flow term "outflow" are the same flow-vs-position class."""
    from tradingagents.validators import extract_net_debt_claims

    for text in (
        "The FY26 net cash **outlay** $47.7B for capex dwarfs OCF.",
        "the quarter's net cash outflow of $47.7B reflects capex timing",
    ):
        claims = extract_net_debt_claims(text)
        assert claims == [], (
            f"flow term extracted as position claim: "
            f"{[(c.label, c.value_raw) for c in claims]} in {text[:50]!r}")


def test_phase_9_2_position_claim_next_to_outlay_still_extracted():
    """Guard: a genuine position claim followed by a separate outlay clause
    keeps extracting."""
    from tradingagents.validators import extract_net_debt_claims

    claims = extract_net_debt_claims(
        "ORCL holds $8.16B net cash; outlay for capex was $47.7B.")
    assert [(c.value_raw, c.is_cash) for c in claims] == [("$8.16B", True)]


# --- wk29 cadence (2026-07-17): whose-number / wrong-kind-of-number FPs -------
# Five net-debt false positives in one batch, all where a dollar figure near
# "net debt" was NOT the subject's net-debt position. See the wk29 memory.

# TXN-style: real net debt $10.50B; the report also cites market cap $258.53B
# right beside "net debt" inside an EV bridge.
_TXN_NET_DEBT_JSON = {
    "trade_date": "2026-07-17", "financial_currency": "USD",
    "net_debt": 10_500_000_000.0, "net_debt_source": "yfinance",
    "total_debt": 14_050_000_000.0, "long_term_debt": 12_050_000_000.0,
    "current_debt": 2_000_000_000.0, "capital_lease_obligations": 0.0,
    "cash_and_equivalents": 3_550_000_000.0, "short_term_investments": None,
    "other_short_term_investments": None,
    "cash_plus_short_term_investments": 3_550_000_000.0, "unavailable": False,
}

# MARA-style: real net debt $1.90B; market cap $4.45B cited beside "net debt".
_MARA_NET_DEBT_JSON = {
    "trade_date": "2026-07-17", "financial_currency": "USD",
    "net_debt": 1_900_000_000.0, "net_debt_source": "yfinance",
    "total_debt": 2_463_960_000.0, "long_term_debt": 2_463_960_000.0,
    "current_debt": 0.0, "capital_lease_obligations": 0.0,
    "cash_and_equivalents": 563_960_000.0, "short_term_investments": None,
    "other_short_term_investments": None,
    "cash_plus_short_term_investments": 563_960_000.0, "unavailable": False,
}


def test_wk29_skips_txn_market_cap_beside_net_debt(tmp_path):
    """TXN 2026-07-17: 'market cap $258.53B + net debt $10.50B' — the $258.53B
    is market cap, not net debt. Must not flag it; the real $10.50B validates."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_TXN_NET_DEBT_JSON)
    text = "Enterprise value is $269.03B (market cap $258.53B + net debt $10.50B)."
    claims = _with_file(extract_net_debt_claims(text), "decision_executive.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="TXN")
    flagged = {v.claimed_value for v in viols}
    assert not any("258" in f for f in flagged), (
        f"market-cap $258.53B must not be flagged as net debt; got {flagged}"
    )


def test_wk29_skips_mara_market_cap_beside_net_debt(tmp_path):
    """MARA 2026-07-17: 'market cap of only $4.45B and net debt of $1.90B'."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_MARA_NET_DEBT_JSON)
    text = ("MARA carries $4.00B of original convertible-note face value against "
            "a market cap of only $4.45B and net debt of $1.90B.")
    claims = _with_file(extract_net_debt_claims(text), "analyst_fundamentals.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="MARA")
    flagged = {v.claimed_value for v in viols}
    assert not any("4.45" in f for f in flagged), (
        f"market-cap $4.45B must not be flagged as net debt; got {flagged}"
    )


def test_wk29_market_cap_guard_still_flags_real_fabricated_net_debt(tmp_path):
    """Defense: a fabricated net-debt figure NOT preceded by a competing
    metric label must still be flagged (guard must not over-suppress)."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_TXN_NET_DEBT_JSON)
    text = "The balance sheet shows a net debt of $258.53B, a solvency red flag."
    claims = _with_file(extract_net_debt_claims(text), "decision.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="TXN")
    assert any("258" in v.claimed_value for v in viols), (
        "fabricated $258.53B net debt (no competing-metric prefix) must still flag"
    )


def test_wk29_skips_orcl_net_debt_grew_yoy_delta():
    """ORCL 2026-07-17: 'net debt grew $16.47B in one year' — $16.47B is a
    YoY change, not a position. 'grew' was missing from the delta-bridge set."""
    from tradingagents.validators import extract_net_debt_claims
    text = "Net debt/EBITDA of 3.22x is genuine; net debt grew $16.47B in one year."
    claims = extract_net_debt_claims(text)
    assert not any(abs(c.value_dollars - 16.47e9) < 1e6 for c in claims), (
        f"$16.47B YoY delta ('grew') must not extract; got {[c.value_dollars for c in claims]}"
    )


def test_wk29_skips_googl_net_debt_issuance_financing_flow(tmp_path):
    """GOOGL 2026-07-17: '$31.4B of net debt-issuance proceeds' — a financing
    cash flow, not a net-debt position."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    from tests.test_net_debt_validator import _MSFT_NET_DEBT_JSON
    nd = _write_net_debt(tmp_path)
    text = "The release shows Q1 alone brought in $31.4B of net debt-issuance proceeds."
    claims = _with_file(extract_net_debt_claims(text), "analyst_fundamentals.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="GOOGL")
    assert not any("31.4" in v.claimed_value for v in viols), (
        f"net debt-issuance financing flow must not be flagged; got {[v.claimed_value for v in viols]}"
    )


def test_wk29_skips_orcl_funding_gap(tmp_path):
    """ORCL 2026-07-17: '~$24B annual funding gap — pushing net debt/EBITDA
    to 3.22x' — $24B is an FCF funding gap, not a net-debt position."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_TXN_NET_DEBT_JSON)  # any USD canonical
    text = ("Oracle burned $23.686B of free cash flow while spending $55.663B on "
            "capex against only $31.977B of operating cash flow — a ~$24B annual "
            "funding gap — pushing net debt/EBITDA to 3.22x.")
    claims = _with_file(extract_net_debt_claims(text), "analyst_fundamentals.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="ORCL")
    assert not any("24" in v.claimed_value and v.claimed_dollars > 20e9 for v in viols), (
        f"$24B funding gap must not be flagged as net debt; got {[v.claimed_value for v in viols]}"
    )


def test_wk29_skips_echo_contractual_obligations(tmp_path):
    """ECHO 2026-07-17: a '2026 contractual-obligations figure (debt + interest
    + leases + purchase obligations) of $13.21B' is a maturity/obligations
    total, not the net-debt position."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_TXN_NET_DEBT_JSON)
    # Verbatim ECHO 2026-07-17 analyst_fundamentals.md — the flagged $13.21B is
    # a PARENTHETICAL restatement of "$13,213,574 thousand", so the "of $"
    # introducer is separated from $13.21B by another dollar figure.
    text = ("- Against that: the filing's own debt-maturity-ladder table (10-K, "
            "verbatim) shows **$7,279,749 thousand ($7.28B) of long-term debt "
            "obligations maturing in fiscal-year 2026 alone**, against a total 2026 "
            "contractual-obligations figure (debt + interest + leases + purchase "
            "obligations) of **$13,213,574 thousand ($13.21B)** — and the pm_brief's "
            "own Net debt block shows only $1.52B of cash + short-term investments.")
    claims = _with_file(extract_net_debt_claims(text), "analyst_fundamentals.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="ECHO")
    assert not any("13.21" in v.claimed_value for v in viols), (
        f"$13.21B contractual-obligations total must not be flagged; got {[v.claimed_value for v in viols]}"
    )


def test_wk29_obligations_guard_still_flags_real_net_debt_near_maturing(tmp_path):
    """Defense: a fabricated net-debt position sitting near 'maturing' (but
    NOT introduced by an obligations 'of $') must still be flagged."""
    from tradingagents.validators import extract_net_debt_claims, validate_net_debt_claims
    nd = _write_net_debt(tmp_path, data=_TXN_NET_DEBT_JSON)
    text = "With $5B maturing in 2026, the balance sheet still carries net debt of $99B."
    claims = _with_file(extract_net_debt_claims(text), "decision.md")
    viols = validate_net_debt_claims(claims, nd, main_ticker="TXN")
    assert any("99" in v.claimed_value for v in viols), (
        "fabricated $99B net debt near 'maturing' (no obligations 'of') must still flag"
    )
