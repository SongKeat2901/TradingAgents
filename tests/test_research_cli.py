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
                 "--deep", "--quick", "--debate-rounds", "--risk-rounds",
                 "--telegram-notify"):
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


def test_auth_error_exits_1(tmp_path, monkeypatch, capsys):
    import cli.research as research
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeAuthError

    class BoomGraph:
        def __init__(self, debug, config): pass
        def propagate(self, *a, **kw): raise ClaudeCodeAuthError("token expired")

    monkeypatch.setattr(research, "TradingAgentsGraph", BoomGraph)
    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "token expired" in err


def test_unexpected_error_exits_2(tmp_path, monkeypatch, capsys):
    import cli.research as research

    class BoomGraph:
        def __init__(self, debug, config): pass
        def propagate(self, *a, **kw): raise RuntimeError("graph blew up")

    monkeypatch.setattr(research, "TradingAgentsGraph", BoomGraph)
    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "graph blew up" in err
    assert "Traceback" in err


def _stub_state(ticker, date):
    return {
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


def test_telegram_notify_skipped_without_env(tmp_path, monkeypatch):
    """If --telegram-notify is set but env var is missing, no notify call happens."""
    import cli.research as research

    class FakeGraph:
        def __init__(self, debug, config): pass
        def propagate(self, t, d): return _stub_state(t, d), "BUY"

    notify_calls = []

    def boom_notify(*a, **kw):
        notify_calls.append(("success", a, kw))

    monkeypatch.setattr(research, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(research, "notify_success", boom_notify)
    monkeypatch.delenv("TRADINGRESEARCH_BOT_TOKEN", raising=False)

    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path / "out"),
        "--telegram-notify", "-100123",
        "--no-daemonize",  # do not actually fork inside pytest
    ])
    assert rc == 0
    assert notify_calls == []


def test_telegram_notify_success_path(tmp_path, monkeypatch):
    """When env var + flag both set, notify_success runs after the JSON line."""
    import cli.research as research

    class FakeGraph:
        def __init__(self, debug, config): pass
        def propagate(self, t, d): return _stub_state(t, d), "BUY"

    notify_calls = []

    def fake_notify(bot_token, chat_id, output_dir, decision):
        notify_calls.append((bot_token, chat_id, output_dir, decision))

    monkeypatch.setattr(research, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(research, "notify_success", fake_notify)
    monkeypatch.setenv("TRADINGRESEARCH_BOT_TOKEN", "BOT_FAKE")

    out = tmp_path / "out"
    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(out),
        "--telegram-notify", "-100123",
        "--no-daemonize",
    ])
    assert rc == 0
    assert notify_calls == [("BOT_FAKE", "-100123", str(out), "BUY")]


def test_telegram_notify_auth_error_path(tmp_path, monkeypatch):
    """Auth error still triggers notify_failure when telegram args are present."""
    import cli.research as research
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeAuthError

    class BoomGraph:
        def __init__(self, debug, config): pass
        def propagate(self, *a, **kw): raise ClaudeCodeAuthError("token expired")

    notify_calls = []

    def fake_notify_failure(bot_token, chat_id, ticker, date, summary):
        notify_calls.append((bot_token, chat_id, ticker, date, summary))

    monkeypatch.setattr(research, "TradingAgentsGraph", BoomGraph)
    monkeypatch.setattr(research, "notify_failure", fake_notify_failure)
    monkeypatch.setenv("TRADINGRESEARCH_BOT_TOKEN", "BOT_FAKE")

    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path),
        "--telegram-notify", "-100",
        "--no-daemonize",
    ])
    assert rc == 1
    assert len(notify_calls) == 1
    assert notify_calls[0][0] == "BOT_FAKE"
    assert notify_calls[0][1] == "-100"
    assert "token expired" in notify_calls[0][4]


def test_telegram_send_failure_does_not_change_exit_code(tmp_path, monkeypatch, capsys):
    """If Telegram itself is unreachable, the CLI still returns the right exit code."""
    import cli.research as research

    class FakeGraph:
        def __init__(self, debug, config): pass
        def propagate(self, t, d): return _stub_state(t, d), "BUY"

    def angry_notify(*a, **kw):
        raise research.TelegramSendError("network down")

    monkeypatch.setattr(research, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(research, "notify_success", angry_notify)
    monkeypatch.setenv("TRADINGRESEARCH_BOT_TOKEN", "BOT_FAKE")

    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path / "out"),
        "--telegram-notify", "-100",
        "--no-daemonize",
    ])
    assert rc == 0  # success exit even though notify failed
    err = capsys.readouterr().err
    assert "telegram notify failed" in err
