"""Tests for the Phase-7.3 peer-metric validator."""
import json

import pytest

pytestmark = pytest.mark.unit


def _write_peer_data(tmp_path):
    """Real MARA peer ratios + peers from the 2026-05-06 cadence."""
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-06",
        "_unavailable": [],
        "RIOT": {
            "latest_quarter_capex_to_revenue": 78.7,
            "latest_quarter_op_margin": -72.6,
            "ttm_pe": None,
            "forward_pe": None,
            "net_debt": 636_497_000,
            "ttm_ebitda": -326_712_000,
            "nd_ebitda": None,
        },
        "CIFR": {
            "latest_quarter_capex_to_revenue": 385.4,
            "latest_quarter_op_margin": -383.42,
            "ttm_pe": 17.0,
            "forward_pe": 40.95,
            "net_debt": -150_000_000,
            "ttm_ebitda": 197_000_000,
            "nd_ebitda": -0.76,
        },
        "CLSK": {
            "latest_quarter_capex_to_revenue": 20.4,
            "latest_quarter_op_margin": -37.83,
            "ttm_pe": None,
            "forward_pe": 28.5,
            "net_debt": 200_000_000,
            "ttm_ebitda": -229_000_000,
            "nd_ebitda": None,
        },
    }
    peers = {
        "RIOT": {"ticker": "RIOT"},
        "CIFR": {"ticker": "CIFR"},
        "CLSK": {"ticker": "CLSK"},
    }
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")
    return raw


def test_passes_when_metrics_match_peer_ratios(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = (
        "RIOT capex/revenue 78.7%; CIFR forward P/E 40.95x; CLSK op margin "
        "−37.8% (per raw/peer_ratios.json)."
    )
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_catches_clsk_op_margin_sign_flip_fabrication(tmp_path):
    """The exact MARA failure: claimed `CLSK op margin ~5%` when actual
    is −37.83%. Sign + magnitude both wrong."""
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = "CLSK op margin ~5% per raw/peers.json"
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert len(v) == 1
    assert v[0].severity == "MATERIAL"
    assert v[0].type == "wrong_peer_metric"
    assert v[0].ticker == "CLSK"
    assert v[0].claimed_value.strip() == "~5%" or "5" in v[0].claimed_value
    assert "-37.83" in str(v[0].actual_value)


def test_catches_riot_ev_ebitda_fabricated_attribution(tmp_path):
    """`RIOT EV/EBITDA ~12×` is NOT a peer_ratios.json column; if attributed
    to peer_ratios.json it's a fabricated source."""
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = (
        "Per raw/peers.json: RIOT EV/EBITDA ~12×, ND/EBITDA <1×; CIFR "
        "EV/EBITDA ~9×, P/S ~6×, ND/EBITDA ~1.5×."
    )
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    # Should flag at least the RIOT EV/EBITDA fabrication (EV/EBITDA isn't in peer_ratios)
    fabrications = [x for x in v if x.type == "fabricated_metric_attribution"]
    assert any(x.ticker == "RIOT" and "ev/ebitda" in x.metric.lower() for x in fabrications)


def test_v2_catches_aaoi_pm_fabrication_long_bridge_with_equals_separator(tmp_path):
    """AAOI 2026-05-08 PM-level fabrication missed by v1 regex: ticker is
    far from the metric (50+ chars of prose), and metric-value uses `=`
    separator instead of whitespace.

      "**FN (Fabrinet)** — closest comp on optical-transceiver mix and
       hyperscaler exposure; per `raw/peers.json` Fwd P/E = 36.6x,
       TTM operating margin ≈ 11.4%, ND/EBITDA ≈ −2.1x"

    Actual peer_ratios.json: FN forward_pe=36.64, op_margin=10.1,
    nd_ebitda=-1.99. Claimed 11.4% op margin is 13% off canonical; 2.1x
    ND/EBITDA vs −1.99x is +0.11 abs delta on a magnitude of 1.99 (5.5%).
    Both should flag."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-08",
        "_unavailable": [],
        "FN": {
            "latest_quarter_capex_to_revenue": 4.56,
            "latest_quarter_op_margin": 10.1,
            "ttm_pe": 54.31,
            "forward_pe": 36.64,
            "net_debt": -955_888_000,
            "ttm_ebitda": 479_712_000,
            "nd_ebitda": -1.99,
        },
        "COHR": {
            "latest_quarter_capex_to_revenue": 9.11,
            "latest_quarter_op_margin": 11.78,
            "ttm_pe": 155.64,
            "forward_pe": 41.15,
            "net_debt": 2_488_127_000,
            "ttm_ebitda": 1_313_731_968,
            "nd_ebitda": 1.89,
        },
        "LITE": {
            "latest_quarter_capex_to_revenue": 12.56,
            "latest_quarter_op_margin": 9.6,
            "ttm_pe": 158.14,
            "forward_pe": 50.04,
            "net_debt": 2_629_600_000,
            "ttm_ebitda": 508_800_000,
            "nd_ebitda": 5.17,
        },
    }
    peers = {"FN": {"ticker": "FN"}, "COHR": {"ticker": "COHR"}, "LITE": {"ticker": "LITE"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "  - **FN (Fabrinet)** — closest comp on optical-transceiver mix "
        "and hyperscaler exposure; per `raw/peers.json` Fwd P/E = 36.6x, "
        "TTM operating margin ≈ 11.4%, ND/EBITDA ≈ −2.1x (net cash).\n"
        "  - **COHR (Coherent Corp.)** — direct AI-optics platform overlap; "
        "per `raw/peers.json` Fwd P/E = 22.4x, TTM operating margin ≈ 7.8%, "
        "ND/EBITDA ≈ 3.6x.\n"
        "  - **LITE (Lumentum)** — datacom/transceiver competitor; per "
        "`raw/peers.json` Fwd P/E = 28.1x, TTM operating margin ≈ 4.9%, "
        "ND/EBITDA ≈ 1.9x."
    )

    violations = validate_peer_metrics(
        text, "decision.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    # All four MATERIAL drifts must fire (COHR Fwd P/E 22.4 vs 41.15;
    # LITE Fwd P/E 28.1 vs 50.04; COHR ND/EBITDA 3.6 vs 1.89; LITE
    # ND/EBITDA 1.9 vs 5.17).
    by_key = {(v.ticker, v.metric.lower().strip()): v for v in violations
              if v.type == "wrong_peer_metric"}
    cohr_fpe = [v for k, v in by_key.items() if k[0] == "COHR" and "p/e" in k[1]]
    lite_fpe = [v for k, v in by_key.items() if k[0] == "LITE" and "p/e" in k[1]]
    cohr_nde = [v for k, v in by_key.items() if k[0] == "COHR" and "ebitda" in k[1]]
    lite_nde = [v for k, v in by_key.items() if k[0] == "LITE" and "ebitda" in k[1]]
    assert cohr_fpe, f"missed COHR Fwd P/E fabrication; got {sorted(by_key)}"
    assert lite_fpe, f"missed LITE Fwd P/E fabrication; got {sorted(by_key)}"
    assert cohr_nde, f"missed COHR ND/EBITDA fabrication; got {sorted(by_key)}"
    assert lite_nde, f"missed LITE ND/EBITDA fabrication; got {sorted(by_key)}"


def test_v3_skips_nvda_tsm_nd_ebitda_metric_kind_mismatch(tmp_path):
    """NVDA 2026-05-21 false positive: a fundamentals notes line says

      "TSM ND/EBITDA figures are TWD-denominated as provided in pm_brief
       peer table ($936.16B net debt / $2,856.03B TTM EBITDA = 0.33x;
       use the ratio, not the raw dollar figures)."

    The peer-metric regex bound 'TTM EBITDA' (the divisor's label) to
    '0.33x' (the equation result) and flagged TTM EBITDA = 0.33x against
    canonical $2.86 trillion. But 0.33x is a multiple (ND/EBITDA ratio
    result), not a $ magnitude — TTM EBITDA can NEVER be a multiple.

    Fix: when the captured value's kind ('ratio') doesn't match the
    metric's expected kind ('billions'/'millions' for TTM EBITDA), skip.
    """
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-21",
        "_unavailable": [],
        "TSM": {
            "latest_quarter_capex_to_revenue": 31.01,
            "latest_quarter_op_margin": 58.1,
            "ttm_pe": 34.72,
            "forward_pe": 20.73,
            "net_debt": 936_160_000_000.0,
            "ttm_ebitda": 2_856_031_092_736.0,
            "nd_ebitda": 0.33,
        },
    }
    peers = {"TSM": {"ticker": "TSM"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "TSM ND/EBITDA figures are TWD-denominated as provided in pm_brief "
        "peer table ($936.16B net debt / $2,856.03B TTM EBITDA = 0.33x; "
        "use the ratio, not the raw dollar figures, for cross-currency "
        "comparisons)."
    )
    violations = validate_peer_metrics(
        text, "analyst_fundamentals.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    # The wrong_peer_metric FP that binds 0.33x to TTM EBITDA must NOT fire
    ttm_ebitda_vios = [v for v in violations if v.type == "wrong_peer_metric"
                       and "ebitda" in v.metric.lower() and "nd" not in v.metric.lower()]
    assert not ttm_ebitda_vios, (
        f"TTM EBITDA must not bind to 0.33x ratio value; got {ttm_ebitda_vios}"
    )


def test_phase_8_1_skips_on_inline_subtraction_peer_metric(tmp_path):
    """ON 2026-05-07 false positive: the LLM showed inline net-debt math

        "- Net Debt = $11,177M − $3,202M = **$7,975M**"

    The Phase 7.12 peer_metric prefix-eater handled DIVISION (`$X / $Y =`)
    but not SUBTRACTION (`$X − $Y =`). The regex captured `$11,177M` (the
    minuend) as NXPI's net_debt and flagged it as wrong_peer_metric vs
    canonical $8.342B. Phase 8.1 generalises the operator to `[/−-]` so
    both division and subtraction work."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    # Canonical NXPI net_debt = $8,342M (the bare yfinance Net Debt cell).
    # The LLM's computed $7,975M is within 5% — should pass tolerance.
    peer_ratios = {
        "trade_date": "2026-05-07",
        "_unavailable": [],
        "NXPI": {
            "latest_quarter_capex_to_revenue": 5.0,
            "latest_quarter_op_margin": 30.0,
            "ttm_pe": 23.0,
            "forward_pe": 18.0,
            "net_debt": 8_342_000_000.0,
            "ttm_ebitda": 4_000_000_000.0,
            "nd_ebitda": 2.08,
        },
    }
    peers = {"NXPI": {"ticker": "NXPI"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = "- **NXPI**: Net Debt = $11,177M − $3,202M = **$7,975M**"
    vios = validate_peer_metrics(
        text, "decision.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    # The minuend $11,177M must not bind to NXPI net debt; the result
    # $7,975M is within 5% of canonical $8,342M and should pass.
    nxpi_minuend_vios = [v for v in vios if v.ticker == "NXPI"
                         and "11,177" in str(v.claimed_value)]
    assert not nxpi_minuend_vios, (
        f"NXPI minuend $11,177M must not flag as net debt; got {nxpi_minuend_vios}"
    )


def test_v2_2_skips_amzn_inline_equation_false_positive(tmp_path):
    """AMZN 2026-05-21 false positive: when the LLM shows the math inline as
    `<METRIC> = $X / $Y = Z%`, the v2 regex previously captured the first
    value after the metric phrase ($X, the numerator) instead of the final
    Z% computed answer. That misread `MSFT capex/revenue = $30,900M /
    $83,100M = 37.2%` as a claim of `$30,900M` for capex/revenue and flagged
    it against canonical 37.25% — a clear false positive: the math is
    correct (30900/83100 = 37.2%, ~37.25% canonical).

    Source: AMZN 2026-05-21 decision.md line 10, validation_report.json
    phase_7_3_peer_metric violation (2 of 2, also same shape for GOOGL).
    """
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-21",
        "_unavailable": [],
        "MSFT": {
            "latest_quarter_capex_to_revenue": 37.25,
            "latest_quarter_op_margin": 46.33,
            "ttm_pe": 24.81,
            "forward_pe": 21.56,
            "net_debt": 8_157_000_000.0,
            "ttm_ebitda": 184_457_003_008.0,
            "nd_ebitda": 0.04,
        },
        "GOOGL": {
            "latest_quarter_capex_to_revenue": 32.46,
            "latest_quarter_op_margin": 36.12,
            "ttm_pe": 29.6,
            "forward_pe": 26.83,
            "net_debt": 39_438_000_000.0,
            "ttm_ebitda": 161_315_995_648.0,
            "nd_ebitda": 0.24,
        },
    }
    peers = {"MSFT": {"ticker": "MSFT"}, "GOOGL": {"ticker": "GOOGL"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "- **Peer capex/revenue ratios (Q1 2026, inline arithmetic):**\n"
        "  - **MSFT Q1 capex/revenue = $30,900M / $83,100M = 37.2%** "
        "(raw/peers.json MSFT capex cell $30.9B ÷ revenue cell $83.1B)\n"
        "  - **GOOGL Q1 capex/revenue = $35,700M / $109,900M = 32.5%** "
        "(raw/peers.json GOOGL capex cell $35.7B ÷ revenue cell $109.9B)\n"
    )

    violations = validate_peer_metrics(
        text, "decision.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    # Inline-equation math is correct (37.2% ≈ 37.25, 32.5% ≈ 32.46);
    # NO peer-metric violation should fire on either line. The validator
    # must consume the `$X / $Y =` prefix and bind to the final percent.
    cap_vios = [v for v in violations if v.type == "wrong_peer_metric"
                and "capex" in v.metric.lower()]
    assert not cap_vios, (
        f"false positive on AMZN inline-equation form; got {cap_vios}"
    )


def test_v2_1_skips_metric_attributed_to_subject_ticker(tmp_path):
    """NVDA 2026-05-08 false positive: a fundamentals line like

      "ND/EBITDA: NVDA computed as (-$51.52B net cash) / $133.2B TTM EBITDA = -0.39x"

    was flagged as 'INTC TTM EBITDA -0.39x' because Phase 7.3 v2's
    lookback only scans for peer-set tickers. When the subject (NVDA)
    is the closest ticker before the metric, the validator should
    bind to the subject and skip — peer-cell verification doesn't
    apply to subject-attributed metrics. Without this fix, the
    lookback fell through past NVDA to the previous peer mention
    (INTC), producing a wrong-peer-metric flag.

    Fix: validate_peer_metrics now accepts main_ticker; when the
    nearest ticker before a metric-value is the main_ticker, skip
    the claim."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-08",
        "_unavailable": [],
        "INTC": {
            "latest_quarter_capex_to_revenue": 26.78,
            "latest_quarter_op_margin": 6.88,
            "ttm_pe": None,
            "forward_pe": 81.61,
            "net_debt": 27_784_000_000,
            "ttm_ebitda": 14_174_000_128,
            "nd_ebitda": 1.96,
        },
        "AMD": {
            "latest_quarter_capex_to_revenue": 1.0,
            "latest_quarter_op_margin": 5.0,
            "ttm_pe": 151.7,
            "forward_pe": 30.0,
            "net_debt": -6_700_000_000,
            "ttm_ebitda": 7_400_000_000,
            "nd_ebitda": -0.90,
        },
    }
    peers = {"INTC": {"ticker": "INTC"}, "AMD": {"ticker": "AMD"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "Source notes:\n"
        "- INTC Q1 2026 (Mar 2026) vs Q1 2025 (Mar 2025).\n"
        "4. ND/EBITDA: NVDA computed as (-$51.52B net cash) / $133.2B "
        "TTM EBITDA = -0.39x; peers from pm_brief.md verbatim."
    )

    violations = validate_peer_metrics(
        text, "analyst_fundamentals.md",
        raw / "peer_ratios.json", raw / "peers.json",
        main_ticker="NVDA",
    )
    # The "-0.39x" is NVDA's ND/EBITDA, not INTC's TTM EBITDA. Phase 7.3
    # should bind to NVDA (subject), recognize it as a subject-claim,
    # and skip.
    assert violations == [], (
        f"NVDA-subject claim flagged as INTC peer drift: "
        f"{[(v.ticker, v.metric, v.claimed_value) for v in violations]}"
    )


def test_v2_value_parser_keeps_sign_for_dollar_prefixed_negatives():
    """`_parse_value("-$956M")` must return (-956, "millions") not (956,).
    AAOI 2026-05-08 surfaced this: the FN net cash claim "−$956M" matches
    canonical -955_888_000 within 0.01% — but the parser dropped the sign,
    flipping a clean match into a 200%+ drift."""
    from tradingagents.validators.peer_metric_validator import _parse_value

    for prefix in ("-$956M", "−$956M", "$-956M", "-956M"):
        val, kind = _parse_value(prefix)
        assert val == -956.0, f"{prefix!r}: parsed {val}, expected -956"
        assert kind == "millions"

    # Sanity: positive forms still parse positive
    val, _ = _parse_value("$956M")
    assert val == 956.0
    val, _ = _parse_value("956M")
    assert val == 956.0

    # Same coverage for billions
    for prefix in ("-$2.49B", "−$2.49B", "$-2.49B", "-2.49B"):
        val, kind = _parse_value(prefix)
        assert val == -2.49, f"{prefix!r}: parsed {val}, expected -2.49"
        assert kind == "billions"


def test_v2_keeps_immediate_form_passing(tmp_path):
    """Backwards compat: the existing tight form
    `RIOT capex/revenue 78.7%` must still parse cleanly under v2."""
    from tradingagents.validators import validate_peer_metrics
    raw = _write_peer_data(tmp_path)
    text = "RIOT capex/revenue 78.7%; CIFR forward P/E 40.95x"
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_skips_when_peers_json_missing(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    text = "RIOT op margin 5%"
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_skips_unavailable_peers(tmp_path):
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-06",
        "_unavailable": ["RIOT"],
        "RIOT": {"unavailable": True, "reason": "missing rows"},
    }
    peers = {"RIOT": {"ticker": "RIOT"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")
    text = "RIOT op margin 5%"
    # Should skip — peer data marked unavailable, can't verify
    v = validate_peer_metrics(text, "decision.md", raw / "peer_ratios.json", raw / "peers.json")
    assert v == []


def test_phase_8_1_skips_compound_word_ticker_false_positive_soun(tmp_path):
    """SOUN 2026-05-29 false positive: the LLM described CRNC's metrics in
    a paragraph that also contained the phrase 'voice-AI pure-play'. The
    ticker `AI` (C3.ai) is in the peer set; the prior `\\b...\\b` regex
    matched `AI` inside the compound word `voice-AI` (hyphen counts as a
    word boundary), and the metric-binding lookback bound CRNC's
    `op margin -3.6%` to ticker AI → compared against AI's actual value
    (-263.6%) → spurious wrong_peer_metric MATERIAL. Same shape recurred
    for BBAI and PLTR.

    Fix: ticker regex now uses negative lookbehind/lookahead rejecting
    preceding or following letter/hyphen, so 'voice-AI', 'AI-government',
    'non-AI' no longer match. Real ticker references still match.
    """
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-29",
        "_unavailable": [],
        "CRNC": {
            "latest_quarter_capex_to_revenue": 0.8,
            "latest_quarter_op_margin": -3.6,
            "ttm_pe": None,
            "forward_pe": 16.79,
            "net_debt": 64_000_000.0,
            "ttm_ebitda": 40_000_000.0,
            "nd_ebitda": 1.60,
        },
        "AI": {
            "latest_quarter_capex_to_revenue": 0.82,
            "latest_quarter_op_margin": -263.63,
            "ttm_pe": None,
            "forward_pe": None,
            "net_debt": -617_000_000.0,
            "ttm_ebitda": -453_099_008.0,
            "nd_ebitda": None,
        },
    }
    peers = {"CRNC": {"ticker": "CRNC"}, "AI": {"ticker": "AI"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "- **CRNC (Cerence)** — voice-AI pure-play; closest functional comp; "
        "Q1 capex/revenue 0.8%, Q1 op margin -3.6%, Forward P/E 16.79x, "
        "TTM EBITDA $40M, ND/EBITDA 1.60x."
    )
    vios = validate_peer_metrics(
        text, "decision.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    # The CRNC values are correct (match peer_ratios.json). The bug bound them
    # to AI via 'voice-AI'. After the fix, no violation should fire.
    crnc_vios = [v for v in vios if v.ticker == "CRNC"]
    ai_vios = [v for v in vios if v.ticker == "AI"]
    assert not crnc_vios, f"CRNC values are correct; got: {crnc_vios}"
    assert not ai_vios, (
        f"AI must NOT be bound to CRNC's metrics via 'voice-AI'; got: {ai_vios}"
    )


def test_phase_8_1_bbai_ai_slash_government_misbinding(tmp_path):
    """Second SOUN false positive caught after the hyphen-only fix: the BBAI
    description contains "small-cap AI/government" — the `/` separator still
    let `AI` match. Every BBAI metric (capex/rev 0.9%, op margin -66.9%,
    TTM EBITDA -$63M) bound to ticker AI instead of BBAI. Fix: extend the
    ticker boundary rejection to include `/`."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-29",
        "_unavailable": [],
        "BBAI": {
            "latest_quarter_capex_to_revenue": 0.93,
            "latest_quarter_op_margin": -66.9,
            "ttm_pe": None, "forward_pe": None,
            "net_debt": 20_000_000.0,
            "ttm_ebitda": -63_223_000.0,
            "nd_ebitda": None,
        },
        "AI": {
            "latest_quarter_capex_to_revenue": 0.82,
            "latest_quarter_op_margin": -263.63,
            "ttm_pe": None, "forward_pe": None,
            "net_debt": -617_000_000.0,
            "ttm_ebitda": -453_099_008.0,
            "nd_ebitda": None,
        },
    }
    peers = {"BBAI": {"ticker": "BBAI"}, "AI": {"ticker": "AI"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "- **BBAI (BigBear.ai)** — small-cap AI/government; retail-driven "
        "sentiment comp; Q1 capex/revenue 0.9%, Q1 op margin -66.9%, "
        "TTM EBITDA -$63M."
    )
    vios = validate_peer_metrics(
        text, "decision.md", raw / "peer_ratios.json", raw / "peers.json",
    )
    ai_vios = [v for v in vios if v.ticker == "AI"]
    assert not ai_vios, (
        "AI must not be bound to BBAI's metrics via 'AI/government'; "
        f"got: {ai_vios}"
    )


def test_phase_8_1_tolerance_floor_handles_one_decimal_rounding():
    """SOUN had CRNC capex/rev 0.75 → LLM rendered '0.8%' (1-decimal
    rounding). The pure 5%-relative tolerance flagged it as MATERIAL drift
    because |0.05| / 0.75 = 6.7%. Fix: 0.1 absolute floor on the diff for
    ratio/pct values handles 1-decimal rendering of tiny percentages."""
    from tradingagents.validators.peer_metric_validator import _values_match
    # 1-decimal rounding edge cases that should now pass
    assert _values_match(0.8, 0.75, "pct") is True   # CRNC capex
    assert _values_match(0.9, 0.93, "pct") is True   # BBAI capex
    assert _values_match(0.5, 0.45, "pct") is True   # PLTR capex
    # Real drift on larger values still fails (the relative test bites)
    assert _values_match(30.0, 50.0, "pct") is False
    assert _values_match(12.0, 10.0, "ratio") is False
    # Within-relative on mid-size values still passes
    assert _values_match(50.0, 51.0, "pct") is True  # 2% relative
    # Cross-check: a real fabrication 2 percentage points off on a small base
    # should still flag once we exceed the 0.1 absolute floor
    assert _values_match(2.5, 0.5, "pct") is False   # 2pp diff = 400%


def test_phase_8_2_skips_subject_metric_in_peer_comparison_listing_tsm(tmp_path):
    """TSM 2026-05-29 false positive #1: line 11 of decision_executive.md
    had

      "The bull case rests on TSM compounding... Q1 FY26 op margin of
       58.1% is materially above every peer in the comparison set
       (next-best ASML at 36.0%; AMAT 31.9%), and the balance sheet
       carries roughly $74B USD of net cash (ND/EBITDA −0.83x)..."

    The "ND/EBITDA −0.83x" is TSM's subject ratio. The v2 lookback found
    the nearest peer (AMAT) and bound -0.83x to AMAT, flagging drift vs
    AMAT's nd_ebitda=0.02. Phase 8.2: when 2+ peers AND subject appear
    in the lookback AND nearest peer is >30 chars from the metric, the
    context is a peer-comparison list — treat as subject claim, skip."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-29",
        "_unavailable": [],
        "ASML": {"latest_quarter_op_margin": 36.02, "nd_ebitda": -0.45,
                 "forward_pe": 33.82},
        "AMAT": {"latest_quarter_op_margin": 31.9, "nd_ebitda": 0.02,
                 "forward_pe": 27.83},
    }
    peers = {"ASML": {"ticker": "ASML"}, "AMAT": {"ticker": "AMAT"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "The bull case rests on TSM compounding a leading-edge monopoly. "
        "Q1 FY26 op margin of 58.1% is materially above every peer in the "
        "comparison set (next-best ASML at 36.0%; AMAT 31.9%), and the "
        "balance sheet carries roughly $74B USD of net cash "
        "(ND/EBITDA −0.83x)."
    )

    violations = validate_peer_metrics(
        text, "decision_executive.md",
        raw / "peer_ratios.json", raw / "peers.json",
        main_ticker="TSM",
    )
    nd_violations = [v for v in violations if "nd/ebitda" in v.metric.lower()
                     or v.claimed_value == "−0.83x"]
    assert nd_violations == [], (
        f"TSM's own ND/EBITDA −0.83x flagged as AMAT drift: "
        f"{[(v.ticker, v.metric, v.claimed_value) for v in violations]}"
    )


def test_phase_8_2_skips_subject_metric_after_peer_list_sentence_break_tsm(tmp_path):
    """TSM 2026-05-29 false positive #2: line 66 of decision_executive.md
    had

      "TSM's Q1 op margin of 58.1% is materially above every peer in the
       set — best peer ASML 36.0%, then AMAT 31.9%, UMC 18.4%, INTC 6.9%
       — confirming that the leading-edge premium is a margin reality,
       not a multiple narrative. Forward P/E 21.44x prices TSM below
       ASML (33.82x), AMAT (27.83x), UMC (27.59x)..."

    The "Forward P/E 21.44x" is TSM's subject forward P/E. The v2
    lookback found INTC as the nearest peer (last in the comparator
    list) and bound 21.44x to INTC (fwd_pe=74.48 per peer_ratios.json),
    flagging drift. Phase 8.2 detects the peer-comparison-listing
    context (4 peers + subject + sentence break to the metric)."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-29",
        "_unavailable": [],
        "ASML": {"latest_quarter_op_margin": 36.02, "forward_pe": 33.82},
        "AMAT": {"latest_quarter_op_margin": 31.9, "forward_pe": 27.83},
        "UMC": {"latest_quarter_op_margin": 18.38, "forward_pe": 27.59},
        "INTC": {"latest_quarter_op_margin": 6.88, "forward_pe": 74.48},
    }
    peers = {t: {"ticker": t} for t in ("ASML", "AMAT", "UMC", "INTC")}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "TSM's Q1 op margin of 58.1% is materially above every peer in "
        "the set — best peer ASML 36.0%, then AMAT 31.9%, UMC 18.4%, "
        "INTC 6.9% — confirming that the leading-edge premium is a "
        "margin reality, not a multiple narrative. Forward P/E 21.44x "
        "prices TSM below ASML (33.82x), AMAT (27.83x), UMC (27.59x)."
    )

    violations = validate_peer_metrics(
        text, "decision_executive.md",
        raw / "peer_ratios.json", raw / "peers.json",
        main_ticker="TSM",
    )
    # Filter to the trailing "Forward P/E 21.44x" claim (other "Forward P/E"
    # values like 33.82x are legitimate peer claims for ASML/AMAT/UMC).
    bad = [v for v in violations if v.claimed_value == "21.44x"]
    assert bad == [], (
        f"TSM's own Forward P/E 21.44x flagged as INTC drift: "
        f"{[(v.ticker, v.metric, v.claimed_value) for v in violations]}"
    )


def test_phase_8_2_preserves_fn_single_peer_descriptive_attribution(tmp_path):
    """Defense: the FN — closest comp ... per peers.json Fwd P/E = 36.6x
    case (PM brief style with long bridge but single peer in scope) must
    still bind 36.6x to FN. Only one peer in the lookback → the
    2-peer-listing heuristic doesn't fire → nearest-ticker rule applies."""
    from tradingagents.validators import validate_peer_metrics
    raw = tmp_path / "raw"
    raw.mkdir()
    peer_ratios = {
        "trade_date": "2026-05-08",
        "_unavailable": [],
        "FN": {"latest_quarter_op_margin": 11.4, "forward_pe": 30.0,
               "nd_ebitda": -2.1},
    }
    peers = {"FN": {"ticker": "FN"}}
    (raw / "peer_ratios.json").write_text(json.dumps(peer_ratios), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers), encoding="utf-8")

    text = (
        "**FN (Fabrinet)** — closest comp on optical-transceiver mix and "
        "hyperscaler exposure; per `raw/peers.json` Fwd P/E = 36.6x, TTM "
        "operating margin ≈ 11.4%, ND/EBITDA ≈ −2.1x. AAOI's setup is "
        "lighter capex."
    )

    violations = validate_peer_metrics(
        text, "pm_brief.md",
        raw / "peer_ratios.json", raw / "peers.json",
        main_ticker="AAOI",
    )
    # FN's fwd_pe is 30.0 in our fixture; claimed 36.6x is a real drift
    # vs FN's 30.0 → MUST flag (single-peer descriptive paragraph stays
    # bound to FN; not swallowed by the 2-peer listing rule).
    fn_pe = [v for v in violations
             if v.ticker == "FN" and v.claimed_value == "36.6x"]
    assert len(fn_pe) == 1, (
        f"FN single-peer descriptive Fwd P/E claim must still validate: "
        f"violations={[(v.ticker, v.metric, v.claimed_value) for v in violations]}"
    )


def test_render_violations_text_pass():
    from tradingagents.validators.peer_metric_validator import render_peer_violations_text
    out = render_peer_violations_text([])
    assert "PEER VALIDATION PASS" in out


def test_value_parser_handles_common_formats():
    from tradingagents.validators.peer_metric_validator import _parse_value
    assert _parse_value("~12×")[0] == 12.0
    assert _parse_value("~12×")[1] == "ratio"
    assert _parse_value("38.5%")[0] == 38.5
    assert _parse_value("38.5%")[1] == "pct"
    assert _parse_value("$1.5B")[0] == 1.5
    assert _parse_value("$1.5B")[1] == "billions"
    assert _parse_value("-37.83%")[0] == -37.83
    # Unparseable
    assert _parse_value("low single-digit")[1] == "raw"
