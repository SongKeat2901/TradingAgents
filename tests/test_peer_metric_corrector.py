"""Phase 9 P2 — Python-authoritative peer-metric correction.

The 2026-05-31 audit found the dominant failure mode is small peer-P/E
inflations (e.g. GOOGL forward P/E cited 30.57x vs 29.59x true) — a 3.3%
lift that passes the validator's ±5% tolerance but fails a verbatim audit.
Rather than trust the LLM to copy numbers, snap every verifiable peer-metric
value in the report to the authoritative peer_ratios.json cell, formatted
exactly like the deterministic pm_brief table.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_RATIOS = {
    "trade_date": "2026-05-21",
    "_unavailable": [],
    "GOOGL": {
        "forward_pe": 29.59,
        "ttm_pe": 26.5,
        "latest_quarter_op_margin": 36.12,
        "latest_quarter_capex_to_revenue": 32.46,
        "net_debt": -36360000000,
        "ttm_ebitda": 150000000000,
        "nd_ebitda": None,
    },
    "AMZN": {
        "forward_pe": 27.04,
        "ttm_pe": 27.44,
        "latest_quarter_op_margin": 13.14,
        "latest_quarter_capex_to_revenue": 24.35,
        "net_debt": 54000000000,
        "ttm_ebitda": 130000000000,
        "nd_ebitda": 0.42,
    },
    "RIOT": {  # negative-margin peer, for the sign-flip case
        "forward_pe": None,
        "ttm_pe": None,
        "latest_quarter_op_margin": -37.83,
        "latest_quarter_capex_to_revenue": None,
        "net_debt": None,
        "ttm_ebitda": None,
        "nd_ebitda": None,
    },
}
_PEERS = {"GOOGL", "AMZN", "RIOT"}


def _correct(text, main="MSFT"):
    from tradingagents.validators.peer_metric_corrector import correct_peer_metrics_text
    return correct_peer_metrics_text(text, _RATIOS, _PEERS, main_ticker=main)


def test_inflated_forward_pe_snapped_to_verbatim():
    text = "Peers screen rich: GOOGL Forward P/E 30.57x versus the group."
    out, corrections = _correct(text)
    assert "29.59x" in out
    assert "30.57x" not in out
    assert len(corrections) == 1
    c = corrections[0]
    assert c.ticker == "GOOGL"
    assert c.old_value.strip().endswith("30.57x")
    assert c.new_value == "29.59x"


def test_correct_value_is_left_untouched():
    text = "GOOGL Forward P/E 29.59x is fair."
    out, corrections = _correct(text)
    assert out == text
    assert corrections == []


def test_markdown_bold_around_value_is_preserved():
    text = "GOOGL Forward P/E **30.57x** rich."
    out, corrections = _correct(text)
    assert "**29.59x**" in out
    assert len(corrections) == 1


def test_op_margin_sign_flip_corrected():
    text = "RIOT op margin 5.0% looks healthy."
    out, corrections = _correct(text)
    assert "-37.8%" in out
    assert "5.0%" not in out


def test_unit_glyph_preserved_when_times_sign_used():
    text = "AMZN TTM P/E 99.99× is wrong."
    out, corrections = _correct(text)
    assert "27.44×" in out  # keeps the × glyph the author used


def test_subject_ticker_metrics_are_not_touched():
    # MSFT is the subject; its own metrics are validated by the deterministic
    # block, not corrected here. (MSFT isn't even in the peer-ratios set.)
    text = "MSFT Forward P/E 99.0x is the subject."
    out, corrections = _correct(text)
    assert out == text
    assert corrections == []


def test_unavailable_peer_field_is_not_corrected():
    text = "GOOGL ND/EBITDA 9.9x cited."  # nd_ebitda is None in _RATIOS
    out, corrections = _correct(text)
    assert out == text
    assert corrections == []


def test_markdown_table_forward_pe_column_corrected():
    text = (
        "| Peer | Capex/Rev | Op Margin | Fwd P/E |\n"
        "|---|---:|---:|---:|\n"
        "| GOOGL | 32.46% | 36.12% | **30.57x** |\n"
        "| AMZN | 24.35% | 13.14% | 27.47x |\n"
    )
    out, corrections = _correct(text)
    assert "**29.59x**" in out, out          # GOOGL forward_pe 30.57 -> 29.59
    assert "27.04x" in out                    # AMZN 27.47 -> 27.04
    assert "30.57x" not in out and "27.47x" not in out
    # Correct op-margin / capex columns left untouched.
    assert "36.12%" in out and "32.46%" in out and "13.14%" in out
    assert len(corrections) == 2


def test_markdown_table_with_inline_computation_only_fixes_result():
    # Op-margin cell shows "A / B = X%"; if the result is wrong, only the
    # result is corrected (operands preserved).
    text = (
        "| Peer | Op Margin |\n"
        "|---|---|\n"
        "| AMZN | 23,852 / 181,519 = 99.99% |\n"
    )
    out, corrections = _correct(text)
    assert "23,852 / 181,519 = 13.14%" in out  # AMZN op margin in _RATIOS
    assert len(corrections) == 1


def test_correct_peer_metrics_in_run_rewrites_files_and_logs(tmp_path):
    from tradingagents.validators.peer_metric_corrector import correct_peer_metrics_in_run

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "peer_ratios.json").write_text(json.dumps(_RATIOS), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps({"GOOGL": {}, "AMZN": {}}), encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"company_of_interest": "MSFT"}), encoding="utf-8")
    (tmp_path / "decision.md").write_text(
        "Verdict.\n\nGOOGL Forward P/E 30.57x and AMZN TTM P/E 27.44x.\n", encoding="utf-8"
    )
    (tmp_path / "decision_executive.md").write_text(
        "GOOGL Forward P/E 30.57x in the exec read.\n", encoding="utf-8"
    )

    result = correct_peer_metrics_in_run(tmp_path)

    assert "29.59x" in (tmp_path / "decision.md").read_text(encoding="utf-8")
    assert "30.57x" not in (tmp_path / "decision.md").read_text(encoding="utf-8")
    assert "29.59x" in (tmp_path / "decision_executive.md").read_text(encoding="utf-8")
    # AMZN 27.44x was already correct → not counted
    assert result["total_corrections"] == 2  # one per file
    log = json.loads((raw / "peer_corrections.json").read_text(encoding="utf-8"))
    assert len(log["corrections"]) == 2
