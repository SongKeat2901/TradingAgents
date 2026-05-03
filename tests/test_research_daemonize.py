"""Tests for the self-daemonize path of the tradingresearch CLI."""

from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.unit


def test_daemonize_parent_prints_pid_and_exits(monkeypatch, tmp_path, capsys):
    from cli.research import _daemonize

    log = tmp_path / "log"
    fork_calls = []

    def fake_fork():
        fork_calls.append(None)
        return 12345  # parent path: child's pid

    monkeypatch.setattr(os, "fork", fake_fork)

    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        _daemonize(str(log))

    assert exc_info.value.code == 0
    assert exits == [0]
    out = capsys.readouterr().out
    assert "started pid=12345" in out
    assert len(fork_calls) == 1


def test_daemonize_second_fork_grandparent_exits(monkeypatch, tmp_path):
    """First fork returns 0 (in-child), second fork returns >0 (in-parent → exit)."""
    from cli.research import _daemonize

    log = tmp_path / "log"
    fork_returns = iter([0, 99])  # child path on first fork; parent on second
    setsid_called = []

    monkeypatch.setattr(os, "fork", lambda: next(fork_returns))
    monkeypatch.setattr(os, "setsid", lambda: setsid_called.append(True))

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)

    with pytest.raises(SystemExit):
        _daemonize(str(log))

    assert setsid_called == [True]


def test_daemonize_grandchild_setsid_and_redirects(monkeypatch, tmp_path):
    """First fork=0 (in-child), setsid called, second fork=0 (in-grandchild),
    grandchild reaches the dup2 block and redirects 0/1/2 to log/devnull.

    This is the only path that actually exercises the FD-redirect code in
    _daemonize; the prior tests only cover the fast-exit branches.
    """
    from cli.research import _daemonize

    log = tmp_path / "subdir" / "out.log"  # parent dir does not exist; mkdir must run
    fork_returns = iter([0, 0])  # child path on both forks
    setsid_called = []
    opened_paths: list[str] = []
    dup2_targets: list[tuple[int, int]] = []
    closed_fds: list[int] = []

    monkeypatch.setattr(os, "fork", lambda: next(fork_returns))
    monkeypatch.setattr(os, "setsid", lambda: setsid_called.append(True))

    fd_alloc = iter([100, 101])  # large enough fds to avoid stdio collision

    def fake_open(path, flags, mode=0o644):
        opened_paths.append(str(path))
        return next(fd_alloc)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "dup2", lambda src, dst: dup2_targets.append((src, dst)))
    monkeypatch.setattr(os, "close", lambda fd: closed_fds.append(fd))
    monkeypatch.setattr(os, "_exit", lambda c: (_ for _ in ()).throw(SystemExit(c)))

    _daemonize(str(log))

    assert setsid_called == [True]
    assert opened_paths[0] == str(log)
    assert opened_paths[1] == os.devnull
    assert sorted(dst for _, dst in dup2_targets) == [0, 1, 2]
    assert sorted(closed_fds) == sorted([100, 101])


def test_daemonize_skipped_when_no_daemonize_set(tmp_path, monkeypatch):
    """If --no-daemonize is set, main() must not call _daemonize."""
    import cli.research as research

    class FakeGraph:
        def __init__(self, debug, config): pass
        def propagate(self, t, d):
            return ({"company_of_interest": t, "trade_date": d,
                     "market_report": "", "sentiment_report": "",
                     "news_report": "", "fundamentals_report": "",
                     "investment_debate_state": {"bull_history": "", "bear_history": "",
                                                  "judge_decision": ""},
                     "risk_debate_state": {"aggressive_history": "", "neutral_history": "",
                                            "conservative_history": "", "judge_decision": ""},
                     "final_trade_decision": "BUY"},
                    "BUY")

    daemonize_calls = []

    monkeypatch.setattr(research, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(research, "_daemonize", lambda *a, **kw: daemonize_calls.append(a))
    # also disable telegram notify to avoid stub-related noise
    monkeypatch.setattr(research, "notify_success", lambda *a, **kw: None)
    monkeypatch.setenv("TRADINGRESEARCH_BOT_TOKEN", "BOT")

    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path),
        "--telegram-notify", "-100",
        "--no-daemonize",
    ])

    assert rc == 0
    assert daemonize_calls == []
