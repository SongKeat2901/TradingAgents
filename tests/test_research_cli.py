"""Tests for the headless tradingresearch CLI."""

import pytest

pytestmark = pytest.mark.unit


def test_help_includes_required_flags(capsys):
    from cli.research import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--ticker", "--date", "--output-dir", "--token-source",
                 "--openclaw-profile-path", "--openclaw-profile-name",
                 "--deep", "--quick", "--debate-rounds", "--risk-rounds"):
        assert flag in out, f"flag {flag} missing from --help"


def test_parse_minimal_args_succeeds():
    from cli.research import build_parser

    parser = build_parser()
    ns = parser.parse_args(["--ticker", "NVDA", "--date", "2024-05-10",
                            "--output-dir", "/tmp/out"])
    assert ns.ticker == "NVDA"
    assert ns.date == "2024-05-10"
    assert ns.output_dir == "/tmp/out"
    assert ns.token_source == "keychain"  # default


def test_missing_required_arg_exits():
    from cli.research import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ticker", "NVDA"])  # missing --date and --output-dir


def test_main_runs_graph_writes_files_prints_json(tmp_path, monkeypatch, capsys):
    """CLI wires args → config → graph.propagate → writer → stdout JSON."""
    import json
    import cli.research as research

    captured_config = {}

    class FakeGraph:
        def __init__(self, debug, config):
            captured_config.update(config)

        def propagate(self, ticker, date):
            state = {
                "company_of_interest": ticker,
                "trade_date": date,
                "market_report": "m",
                "sentiment_report": "s",
                "news_report": "n",
                "fundamentals_report": "f",
                "investment_debate_state": {
                    "bull_history": "b", "bear_history": "be", "judge_decision": "j",
                },
                "risk_debate_state": {
                    "aggressive_history": "a", "neutral_history": "ne",
                    "conservative_history": "c", "judge_decision": "PM: BUY",
                },
                "final_trade_decision": "BUY",
            }
            return state, "BUY"

    monkeypatch.setattr(research, "TradingAgentsGraph", FakeGraph)

    out = tmp_path / "out"
    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(out),
    ])
    assert rc == 0

    # Files written
    assert (out / "decision.md").exists()
    assert (out / "state.json").exists()

    # Config flowed through
    assert captured_config["llm_provider"] == "claude_code"
    assert captured_config["deep_think_llm"] == "claude-sonnet-4-6"

    captured = capsys.readouterr().out.strip().splitlines()

    # Progress banner around the run
    assert any("[research] start" in line for line in captured)
    assert any("[research] done" in line for line in captured)

    # Final JSON line is last
    final_line = captured[-1]
    payload = json.loads(final_line)
    assert payload["decision"] == "BUY"
    assert payload["output_dir"] == str(out)
    assert payload["duration_s"] >= 0
