# Phase 5: Rate-limit-tolerant redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make `tradingresearch` self-detach on `--telegram-notify`, pace LLM calls via a shared `InMemoryRateLimiter`, retry 429s with backoff, and retire the `tradingresearch-bg` wrapper.

**Architecture:** Adds an `os.fork()`-based daemonize step at the top of `cli/research.py:main()` when `--telegram-notify` is set; threads a `pacing_seconds` config key through to a shared `InMemoryRateLimiter` instance passed into both `ChatAnthropic`s constructed by `ClaudeCodeClient.get_llm()`; bumps `max_retries` default; updates the SKILL.md and removes the wrapper.

**Tech Stack:** Python 3.13 (still ≥3.10 OK), `langchain_core.rate_limiters.InMemoryRateLimiter`, stdlib `os.fork`/`os.setsid`, pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-03-phase5-rate-limit-tolerant-redesign.md`

---

## File Structure

**Files to create:**
- `tests/test_research_daemonize.py` — fork-mock tests for daemonize path.

**Files to modify:**
- `cli/research.py` — add `--pacing-seconds`, `--no-daemonize` flags + `_daemonize()` helper invoked on `--telegram-notify` + `--no-daemonize` not set.
- `tradingagents/llm_clients/claude_code_client.py` — accept `rate_limiter` kwarg, forward to ChatAnthropic; raise default `max_retries` to 3.
- `tradingagents/graph/trading_graph.py` — for `claude_code` provider, build `InMemoryRateLimiter` from `pacing_seconds` config, share between deep + quick clients.
- `tradingagents/default_config.py` — add `pacing_seconds: 3` and a model-locking comment.
- `tests/test_research_cli.py` — assert new flags in --help; assert daemonize is invoked when `--telegram-notify` set; assert `--no-daemonize` skips the fork.
- `tests/test_claude_code_openclaw.py` — assert `rate_limiter` kwarg flows through to ChatAnthropic.

**Files to deploy on trueknot (separate runbook, not TDD):**
- New `~/.openclaw/workspace/skills/trading-research/SKILL.md`.
- Remove `~/local/bin/tradingresearch-bg`. Restore `~/local/bin/tradingresearch` symlink.
- Daemon kickstart.

---

## Task 1: `_daemonize()` helper + early-fork in main() on `--telegram-notify`

**Files:**
- Modify: `cli/research.py`
- Create: `tests/test_research_daemonize.py`

### Step 1: Write failing tests

Create `tests/test_research_daemonize.py`:

```python
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


def test_daemonize_child_setsid_and_redirect(monkeypatch, tmp_path):
    from cli.research import _daemonize

    log = tmp_path / "log"
    fork_returns = iter([0, 0])  # child path on both forks
    setsid_called = []
    dup2_targets: list[tuple[int, int]] = []

    monkeypatch.setattr(os, "fork", lambda: next(fork_returns))
    monkeypatch.setattr(os, "setsid", lambda: setsid_called.append(True))
    monkeypatch.setattr(os, "dup2", lambda src, dst: dup2_targets.append((src, dst)))

    # The real os._exit on second-fork's grandparent must still be intercepted.
    exits = []
    monkeypatch.setattr(
        os, "_exit", lambda c: (exits.append(c) or (_ for _ in ()).throw(SystemExit(c)))
    )

    # Second fork is a parent path so it _exit(0)s before the dup2 block.
    fork_returns = iter([0, 99])  # first child, second is parent
    monkeypatch.setattr(os, "fork", lambda: next(fork_returns))

    with pytest.raises(SystemExit):
        _daemonize(str(log))

    # setsid must have been called once between forks; the second-fork parent exits
    assert setsid_called == [True]


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
    monkeypatch.setenv("TRADINGRESEARCH_BOT_TOKEN", "BOT")

    rc = research.main([
        "--ticker", "NVDA", "--date", "2024-05-10",
        "--output-dir", str(tmp_path),
        "--telegram-notify", "-100",
        "--no-daemonize",
    ])

    assert rc == 0
    assert daemonize_calls == []
```

### Step 2: Run failing test

```bash
cd "/Users/songkeat/Documents/Python/Trading Agent/TradingAgents"
.venv/bin/python -m pytest tests/test_research_daemonize.py -v
```

Expected: `ImportError` on `_daemonize` (does not exist yet).

### Step 3: Implement `_daemonize` and wire it into `main()`

Edit `cli/research.py`. Add after the imports:

```python
def _daemonize(log_path: str) -> None:
    """Detach from the controlling terminal so the binary runs fire-and-forget.

    On the first fork, the original parent prints the child PID and exits 0.
    The child then setsid()s and forks a second time; the second fork's parent
    exits, leaving a grandchild orphaned to init (PPID=1). The grandchild
    redirects stdin/stdout/stderr to log_path so progress is captured.

    Standard double-fork daemonize pattern. Safe under macOS launchd-spawned
    skill subprocesses.
    """
    pid1 = os.fork()
    if pid1 > 0:
        # Original parent — print pid and exit before any further imports/work.
        print(f"started pid={pid1}", flush=True)
        os._exit(0)

    os.setsid()

    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)  # stdout
    os.dup2(log_fd, 2)  # stderr
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)  # stdin
    os.close(log_fd)
    os.close(devnull)
```

(`os` and `Path` are not yet imported in this file — add `import os` and `from pathlib import Path` to the imports block at the top.)

Add `--no-daemonize` flag to `build_parser()`:

```python
    p.add_argument(
        "--no-daemonize", action="store_true",
        help="Skip the self-daemonize step (for tests / direct foreground use).",
    )
```

Update `main()` to call `_daemonize()` early, before any heavy graph imports trigger LangGraph's lazy module loading:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Self-daemonize when running fire-and-forget. Must happen before any
    # imports that hold open file descriptors or threads (LangGraph, etc.).
    if args.telegram_notify and not args.no_daemonize:
        log_path = (
            Path(args.output_dir).parent.parent / "logs"
            / f"tradingresearch-{args.date}-{args.ticker}.log"
        )
        _daemonize(str(log_path))

    # ...rest of main() unchanged...
```

(Move the `from tradingagents.default_config import DEFAULT_CONFIG` and similar heavy imports below the daemonize call so the original parent doesn't load them before exiting. Keep the `argparse` and `os`/`Path` imports at module level — they're cheap.)

### Step 4: Run test → expect pass

```bash
.venv/bin/python -m pytest tests/test_research_daemonize.py -v
```

Expected: 3 tests pass.

### Step 5: Run full CLI suite to confirm no regression

```bash
.venv/bin/python -m pytest tests/test_research_cli.py tests/test_research_daemonize.py -v
```

Expected: existing 10 + new 3 = 13 tests pass.

### Step 6: Commit

```bash
git add cli/research.py tests/test_research_daemonize.py
git commit -m "feat(cli): self-daemonize on --telegram-notify (eliminates wrapper need)"
```

---

## Task 2: `--pacing-seconds` flag + shared `InMemoryRateLimiter`

**Files:**
- Modify: `cli/research.py`, `tradingagents/llm_clients/claude_code_client.py`, `tradingagents/graph/trading_graph.py`, `tradingagents/default_config.py`
- Modify: `tests/test_research_cli.py`, `tests/test_claude_code_openclaw.py`

### Step 1: Add the failing tests

Append to `tests/test_research_cli.py`:

```python
def test_pacing_seconds_flag_exposed(capsys):
    from cli.research import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--pacing-seconds" in out


def test_pacing_seconds_default_3(tmp_path):
    from cli.research import build_parser
    parser = build_parser()
    ns = parser.parse_args(["--ticker", "X", "--date", "2024-01-01",
                            "--output-dir", str(tmp_path)])
    assert ns.pacing_seconds == 3
```

Append to `tests/test_claude_code_openclaw.py`:

```python
def test_client_forwards_rate_limiter_to_chat_anthropic(tmp_path, monkeypatch):
    """A rate_limiter passed via kwargs lands on the underlying ChatAnthropic."""
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient

    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {"anthropic:default": {"type": "token", "token": "sk-ant-oat01-z"}},
    }), encoding="utf-8")

    captured: dict = {}

    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client.Anthropic",
        lambda **kw: object(),
    )
    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client.AsyncAnthropic",
        lambda **kw: object(),
    )

    def fake_chat(**kw):
        captured.update(kw)
        return type("Stub", (), {"_client": None, "_async_client": None})()

    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client._OAuthChatAnthropic",
        fake_chat,
    )

    sentinel = object()
    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
        rate_limiter=sentinel,
    )
    client.get_llm()
    assert captured.get("rate_limiter") is sentinel
```

### Step 2: Run → expect failures

```bash
.venv/bin/python -m pytest tests/test_research_cli.py::test_pacing_seconds_flag_exposed tests/test_claude_code_openclaw.py::test_client_forwards_rate_limiter_to_chat_anthropic -v
```

### Step 3: Add the `--pacing-seconds` flag

In `cli/research.py:build_parser()`, after `--risk-rounds`:

```python
    p.add_argument(
        "--pacing-seconds", type=float, default=3.0,
        help=(
            "Minimum seconds between LLM calls (shared across deep+quick "
            "clients). Spreads burst to stay under output-tokens-per-minute "
            "rate limits on subscription auth. 0 disables pacing."
        ),
    )
```

In `_build_config()`:

```python
    config["pacing_seconds"] = args.pacing_seconds
```

### Step 4: Forward `rate_limiter` through `ClaudeCodeClient`

In `tradingagents/llm_clients/claude_code_client.py`:

Add `"rate_limiter"` to `_PASSTHROUGH_KWARGS`:

```python
    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "max_tokens",
        "callbacks", "http_client", "http_async_client", "effort",
        "rate_limiter",
    )
```

Set a sensible default for `max_retries`:

```python
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": "claude-code-oauth",
            "max_tokens": 8192,
            "max_retries": 3,  # Phase 5: absorb transient 429s
        }
```

(If the user passed `max_retries` explicitly via kwargs, the `_PASSTHROUGH_KWARGS` loop overwrites this. The default is `3` for unconfigured callers.)

### Step 5: Build `InMemoryRateLimiter` in `trading_graph.py`

In `tradingagents/graph/trading_graph.py`, inside `_get_provider_kwargs()`'s `claude_code` branch:

```python
            if provider == "claude_code":
                kwargs["token_source"] = self.config.get(
                    "claude_code_token_source", "keychain"
                )
                kwargs["openclaw_profile_path"] = self.config.get(
                    "claude_code_openclaw_profile_path"
                )
                kwargs["openclaw_profile_name"] = self.config.get(
                    "claude_code_openclaw_profile_name", "anthropic:default"
                )

                pacing = self.config.get("pacing_seconds", 0)
                if pacing and pacing > 0:
                    from langchain_core.rate_limiters import InMemoryRateLimiter
                    kwargs["rate_limiter"] = InMemoryRateLimiter(
                        requests_per_second=1.0 / float(pacing),
                        check_every_n_seconds=0.1,
                        max_bucket_size=2,
                    )
```

The same `rate_limiter` instance is passed into BOTH `create_llm_client` calls (deep and quick) because `_get_provider_kwargs` returns one dict that is reused. So pacing is account-wide as designed.

### Step 6: Add `pacing_seconds` to `default_config.py`

Add after the `claude_code_openclaw_profile_name` line:

```python
    # Phase 5: minimum seconds between LLM calls. Shared between deep + quick
    # clients via a langchain_core InMemoryRateLimiter. 0 disables pacing.
    # Subscription-OAuth users on Anthropic typically need 3-5s to stay under
    # output-tokens-per-minute burst limits.
    "pacing_seconds": 3,

    # WARNING: do not set both deep_think_llm and quick_think_llm to the same
    # Sonnet variant on subscription auth — that doubles burst risk on the 5
    # heavy turns (4 analysts often + the 2 deep judges). Keep haiku for quick.
```

### Step 7: Run all tests

```bash
.venv/bin/python -m pytest tests/ -q --tb=no
```

Expected: all pass (existing + 3 new).

### Step 8: Smoke test on MacBook (regression)

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py
```

Expected: PONG.

### Step 9: Commit

```bash
git add cli/research.py tradingagents/llm_clients/claude_code_client.py tradingagents/graph/trading_graph.py tradingagents/default_config.py tests/test_research_cli.py tests/test_claude_code_openclaw.py
git commit -m "feat(rate-limit): pacing flag + shared InMemoryRateLimiter; default max_retries=3"
```

---

## Task 3: Simplified SKILL.md (no wrapper, direct binary)

**Files:**
- Modify: deployment artifact (write the file in `/tmp/skill-phase5.md` for scp during deployment)

This task is documentation+config only — no code, no tests.

### Step 1: Write the new SKILL.md content to a local staging file

```bash
cat > /tmp/skill-phase5.md <<'SKILL_EOF'
---
name: trading-research
description: |
  CANONICAL handler for any "research <TICKER>" / "look at <TICKER>" / "deep dive <TICKER>" / "research <TICKER> [for <date>]" request. Multi-agent LLM equity research returning BUY/SELL/HOLD plus full analyst reports (fundamentals, technicals, sentiment, news) and bull/bear + risk-team debates. Research only — never opens or modifies positions. **When the user asks to research a ticker, ALWAYS invoke this skill. Do NOT generate a market summary in place of running the skill.** Distinct from `ibkr-trader` (live IBKR execution) and `ibkr-fund` (fund mgmt).
---

# Trading Research

## Trigger and primacy

Invoke this skill (do not generate a substitute analysis) whenever the user message matches any of:

- `research <TICKER>` (with or without a date)
- `look at <TICKER>`
- `deep dive <TICKER>`
- `do a research on <TICKER>`
- `research <TICKER> for <YYYY-MM-DD>`

If the user wants live market data instead of historical research, route to `ibkr-trader`. If they want to act on the research result, hand off to `ibkr-trader` with explicit user confirmation.

## How to invoke

The binary self-detaches when `--telegram-notify` is passed, so a single Bash call returns immediately and the long-running graph runs in the background. Call it directly — no wrapper needed.

```bash
TICKER="<TICKER>"
DATE="<YYYY-MM-DD>"  # default: $(date +%Y-%m-%d)
OUTDIR="/Users/trueknot/.openclaw/data/research/${DATE}-${TICKER}"
TRADINGRESEARCH_BOT_TOKEN=$(jq -r .channels.telegram.accounts.default.botToken /Users/trueknot/.openclaw/openclaw.json) \
  /Users/trueknot/local/bin/tradingresearch \
    --ticker "${TICKER}" \
    --date "${DATE}" \
    --output-dir "${OUTDIR}" \
    --telegram-notify "-1003753140043"
```

Expected output: a single line `started pid=<N>` and the call returns within ~1 second. Reply to the user with: *"Kicking off research on `<TICKER>` for `<DATE>` — usually 6-10 minutes. I'll post the BUY/SELL/HOLD decision back here when it's done."* Then end your turn.

The binary will post to the Telegram chat itself when it completes (success or failure). Do not wait. Do not poll.

## Concurrency

The binary refuses to start a second concurrent run. If the user asks for another while one is in flight, tell them: *"A research run is already in progress; one ticker at a time."* If a run is stuck, kill: `pkill -f tradingresearch`.

## Output

On-disk reports persist at `/Users/trueknot/.openclaw/data/research/<DATE>-<TICKER>/`. If the user asks for a specific report after the chat post (*"show the fundamentals analysis"*), `cat` and post it.

## Failure handling

The binary's exit code controls the Telegram message it posts:

| Exit | Telegram message | What you tell the user (if asked) |
|---|---|---|
| 0 | ✅ Decision: ... | (no action — user already saw it) |
| 1 | ❌ auth error | Token expired. Run `/Users/trueknot/.nvm/versions/node/v24.14.1/bin/claude -p hi` or wait for the cron at :17 of the next 6h slot. |
| 2 | ❌ runtime error | Per-run log at `/Users/trueknot/.openclaw/logs/tradingresearch-<DATE>-<TICKER>.log`. Offer to `tail -30` it. |

## Rules

1. **Always include `--telegram-notify`.** That's what triggers the self-detach. Without it the binary runs synchronously and your session will be killed by the watchdog.
2. **One run at a time.** Refuse a second within 5h or warn the user about quota.
3. **Research is research.** Hand-off to `ibkr-trader` for execution must be explicit.
SKILL_EOF
echo "wrote /tmp/skill-phase5.md ($(wc -l < /tmp/skill-phase5.md) lines)"
```

This file gets deployed during the post-merge runbook. No commit needed for this step — it's a deployment artifact, not source.

---

## Task 4: Final regression sweep + push

**Files:** none (verification only).

### Step 1

```bash
cd "/Users/songkeat/Documents/Python/Trading Agent/TradingAgents"
.venv/bin/python -m pytest tests/ -q --tb=no
```

Expected: all pass.

### Step 2

```bash
.venv/bin/tradingresearch --help | grep -E "pacing-seconds|no-daemonize|telegram-notify"
```

Expected: all three flags listed.

### Step 3

```bash
git push origin main
```

### Step 4

```bash
git tag phase5-rate-limit-redesign
```

(Local only — push the tag once Phase 5 deploy on trueknot is verified.)

---

## Phase 5 deploy runbook (after merge)

```bash
# from MacBook
ssh macmini-trueknot
cd ~/tradingagents
git pull origin main
.venv/bin/pip install -e . --quiet  # only if pyproject changed
~/local/bin/tradingresearch --help | grep -E "pacing-seconds|no-daemonize"
ln -sf ~/tradingagents/.venv/bin/tradingresearch ~/local/bin/tradingresearch
rm -f ~/local/bin/tradingresearch-bg
exit

# back on MacBook — write skill content (Task 3 step 1 above) to /tmp/skill-phase5.md, then:
scp /tmp/skill-phase5.md macmini-trueknot:.openclaw/workspace/skills/trading-research/SKILL.md
ssh macmini-superqsp "sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway"

# Verify daemon back up:
until curl -sf -m 3 http://192.168.10.20:18790/health 2>/dev/null | grep -q live; do sleep 3; done

# Smoke test from agent CLI on trueknot:
ssh macmini-trueknot 'zsh -ilc "openclaw agent -m \"research SPY 2024-05-10\" --agent trader"'
# Expected: agent replies "kicking off"; tradingresearch process running detached;
# ~6-10 min later, Telegram receives ✅ Decision: ... message.
```

---

## Self-review notes

- All 5 design points from the spec map to plan tasks: daemonize=Task 1, pacing=Task 2, retries=Task 2 (max_retries=3 default), model lock-down=Task 2 (comment), SKILL simplification=Task 3.
- Type/method names are consistent: `_daemonize`, `pacing_seconds`, `rate_limiter` (kwarg name matches the langchain ChatAnthropic param).
- No placeholders or TBDs.
- Phase 5 deploy runbook is non-TDD (deployment ops); kept as appendix not a TDD task per the same pattern as Phase 1's runbook.
