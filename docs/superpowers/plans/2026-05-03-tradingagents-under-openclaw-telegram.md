# TradingAgents under OpenClaw (Phase 1 — fork-side dev) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the TradingAgents fork so it can run as a headless OpenClaw skill on the trueknot Mac Mini (`192.168.10.20`, user `trueknot`) under the trader agent, by adding (a) an `openclaw_profile` token source for the `claude_code` provider — alongside the existing keychain source — and (b) a non-interactive `tradingresearch` CLI that emits a final decision JSON plus per-analyst report files.

**Architecture:** Reuses everything in `tradingagents/` unchanged (graph, agents, dataflows). Extends `tradingagents/llm_clients/claude_code_client.py` to read OAuth from OpenClaw's `auth-profiles.json` instead of (or alongside) the macOS keychain. Adds `cli/research.py` as a second console script (parallel to the existing interactive `cli/main.py`), which builds a `TradingAgentsGraph` with the right config, runs `propagate()`, streams node-completion lines to stdout, and writes report files to a caller-provided output directory.

**Tech Stack:** Python 3.13 (repo requires-python ≥3.10), LangChain (`langchain_anthropic`), LangGraph (existing), Anthropic SDK (auth_token bearer), pytest with the existing `unit`/`integration`/`smoke` markers. Tests follow the flat `tests/test_<topic>.py` layout already in this repo.

**Spec:** `docs/superpowers/specs/2026-05-03-tradingagents-under-openclaw-telegram-design.md`

---

## File Structure

**Files to create:**
- `cli/research.py` — argparse + main() for `tradingresearch` console script
- `cli/research_writer.py` — pure functions that write `decision.md`, `analyst_*.md`, `debate_*.md`, `state.json` from a final `AgentState`
- `cli/research_progress.py` — LangGraph callback that prints `[node] start|done` lines to stdout
- `tests/test_claude_code_openclaw.py` — token-source unit tests
- `tests/test_research_cli.py` — CLI integration tests against a stubbed graph
- `tests/test_research_writer.py` — writer unit tests

**Files to modify:**
- `tradingagents/llm_clients/claude_code_client.py` — add `_read_openclaw_profile()` + branch in `get_oauth_token()`
- `tradingagents/default_config.py` — add three new keys with safe defaults
- `tradingagents/graph/trading_graph.py` — forward the three new keys into client kwargs when provider is `claude_code`
- `pyproject.toml` — register `tradingresearch = "cli.research:main"`
- `scripts/smoke_test_oauth.py` — add `--source openclaw_profile` flag for manual smoke

---

## Task 1: OpenClaw `auth-profiles.json` reader

**Files:**
- Create: `tests/test_claude_code_openclaw.py`
- Modify: `tradingagents/llm_clients/claude_code_client.py`

The OpenClaw `auth-profiles.json` format (from `OpenClaw/docs/tokens.md`):
```json
{
  "version": 1,
  "profiles": {
    "anthropic:default": {
      "type": "token",
      "provider": "anthropic",
      "token": "sk-ant-oat01-..."
    }
  }
}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claude_code_openclaw.py`:

```python
"""Tests for the openclaw_profile token source in claude_code_client."""

import json

import pytest

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    _read_openclaw_profile,
)


pytestmark = pytest.mark.unit


def _write_profile(path, content):
    path.write_text(json.dumps(content))


def test_reads_token_from_profile(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {
            "anthropic:default": {
                "type": "token",
                "provider": "anthropic",
                "token": "sk-ant-oat01-abc123",
            },
        },
    })
    assert _read_openclaw_profile(str(p), "anthropic:default") == "sk-ant-oat01-abc123"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ClaudeCodeAuthError, match="not found"):
        _read_openclaw_profile(str(tmp_path / "nope.json"), "anthropic:default")


def test_missing_profile_name_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {"version": 1, "profiles": {}})
    with pytest.raises(ClaudeCodeAuthError, match="anthropic:default"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_malformed_token_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "not-an-oauth-token"},
        },
    })
    with pytest.raises(ClaudeCodeAuthError, match="sk-ant-oat01"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    p.write_text("{not json")
    with pytest.raises(ClaudeCodeAuthError):
        _read_openclaw_profile(str(p), "anthropic:default")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/songkeat/Documents/Python/Trading Agent/TradingAgents"
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py -v
```
Expected: ImportError on `_read_openclaw_profile` (function does not exist yet).

- [ ] **Step 3: Implement `_read_openclaw_profile`**

In `tradingagents/llm_clients/claude_code_client.py`, add this near the existing `_read_macos_keychain()`:

```python
def _read_openclaw_profile(path: str, profile_name: str) -> str:
    """Return the access token for an OpenClaw auth-profiles.json profile."""
    p = Path(path)
    if not p.exists():
        raise ClaudeCodeAuthError(
            f"OpenClaw auth-profiles.json not found at {path}. "
            f"Verify the path or run OpenClaw's update-tokens.sh from the MacBook."
        )
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ClaudeCodeAuthError(
            f"OpenClaw auth-profiles.json at {path} is not valid JSON: {e}"
        ) from e

    profiles = data.get("profiles", {})
    profile = profiles.get(profile_name)
    if not profile:
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' not in {path}. "
            f"Available: {sorted(profiles.keys())}"
        )

    token = profile.get("token", "")
    if not token.startswith("sk-ant-oat01-"):
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' token does not look like an Anthropic "
            f"OAuth token (expected sk-ant-oat01- prefix). Rotate via OpenClaw."
        )
    return token
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_claude_code_openclaw.py tradingagents/llm_clients/claude_code_client.py
git commit -m "feat(claude_code): add openclaw auth-profiles.json reader"
```

---

## Task 2: Wire openclaw_profile into `get_oauth_token()`

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tests/test_claude_code_openclaw.py`

Currently `get_oauth_token()` always uses keychain on macOS. It needs to dispatch on a source argument.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_claude_code_openclaw.py`:

```python
def test_get_oauth_token_openclaw_source(tmp_path):
    from tradingagents.llm_clients.claude_code_client import get_oauth_token

    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "sk-ant-oat01-xyz"},
        },
    }))
    token = get_oauth_token(
        source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
    )
    assert token == "sk-ant-oat01-xyz"


def test_get_oauth_token_unknown_source_raises():
    from tradingagents.llm_clients.claude_code_client import get_oauth_token

    with pytest.raises(ClaudeCodeAuthError, match="Unknown token source"):
        get_oauth_token(source="bogus")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py -v
```
Expected: TypeError on unexpected `source` kwarg.

- [ ] **Step 3: Update `get_oauth_token()` signature**

Replace the existing function in `tradingagents/llm_clients/claude_code_client.py`:

```python
def get_oauth_token(
    source: str = "keychain",
    openclaw_profile_path: str | None = None,
    openclaw_profile_name: str | None = None,
) -> str:
    """Return a non-expired Claude Code OAuth access token, or raise.

    source:
      - "keychain"          (default) — macOS keychain via `security` (Linux
                            falls back to ~/.claude/.credentials.json).
      - "openclaw_profile"  — read from an OpenClaw auth-profiles.json file
                            on the same host. Requires openclaw_profile_path
                            and openclaw_profile_name.
    """
    if source == "keychain":
        if platform.system() == "Darwin":
            creds = _read_macos_keychain()
        else:
            creds = _read_linux_creds()
        oauth = creds.get("claudeAiOauth") or creds
        access_token = oauth.get("accessToken")
        expires_at = oauth.get("expiresAt")

        if not access_token:
            raise ClaudeCodeAuthError(
                "No accessToken in Claude Code credentials. Run `claude /login`."
            )
        if expires_at is not None:
            now_ms = int(time.time() * 1000)
            if now_ms >= expires_at:
                raise ClaudeCodeAuthError(
                    "Claude Code access token expired. Run any Claude Code "
                    "command (e.g. `claude /status`) to refresh, then retry."
                )
        return access_token

    if source == "openclaw_profile":
        if not openclaw_profile_path or not openclaw_profile_name:
            raise ClaudeCodeAuthError(
                "openclaw_profile source requires openclaw_profile_path and "
                "openclaw_profile_name."
            )
        return _read_openclaw_profile(openclaw_profile_path, openclaw_profile_name)

    raise ClaudeCodeAuthError(
        f"Unknown token source: {source!r}. Use 'keychain' or 'openclaw_profile'."
    )
```

- [ ] **Step 4: Run all tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Re-run smoke test to confirm keychain path still works**

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py
```
Expected: prints `[response] content='PONG'` (regression check).

- [ ] **Step 6: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tests/test_claude_code_openclaw.py
git commit -m "feat(claude_code): get_oauth_token dispatches on source param"
```

---

## Task 3: Plumb token-source kwargs through `ClaudeCodeClient` and the graph

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `tests/test_claude_code_openclaw.py`

The client constructor must accept the new kwargs, store them, and pass them to `get_oauth_token()` inside `get_llm()`. The graph must read the new config keys and pass them when constructing the client.

- [ ] **Step 1: Add the failing integration test**

Append to `tests/test_claude_code_openclaw.py`:

```python
def test_client_uses_openclaw_source(tmp_path, monkeypatch):
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient

    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "sk-ant-oat01-zzz"},
        },
    }))

    captured = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return object()  # don't actually network

    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.Anthropic", fake_anthropic)
    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.AsyncAnthropic", fake_anthropic)
    # _OAuthChatAnthropic does network on first call but not on construction;
    # we just need the constructor to succeed.
    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client._OAuthChatAnthropic",
        lambda **kw: type("Stub", (), {"_client": None, "_async_client": None, **kw})(),
    )

    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
    )
    client.get_llm()
    assert captured["auth_token"] == "sk-ant-oat01-zzz"
```

- [ ] **Step 2: Run test to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py::test_client_uses_openclaw_source -v
```
Expected: client ignores token_source kwarg, falls back to keychain (test fails because captured token won't match).

- [ ] **Step 3: Update `ClaudeCodeClient.get_llm()` to use the new kwargs**

In `tradingagents/llm_clients/claude_code_client.py`, find the `get_llm` method and replace the `token = get_oauth_token()` line:

```python
    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        token = get_oauth_token(
            source=self.kwargs.get("token_source", "keychain"),
            openclaw_profile_path=self.kwargs.get("openclaw_profile_path"),
            openclaw_profile_name=self.kwargs.get("openclaw_profile_name"),
        )
        # ... rest unchanged
```

- [ ] **Step 4: Add config keys to `default_config.py`**

After the existing `"anthropic_effort": None,` line in `tradingagents/default_config.py`, add:

```python
    # Claude Code OAuth: where to read the access token from.
    # "keychain" — macOS keychain (default; for local dev on the user's MacBook).
    # "openclaw_profile" — read from an OpenClaw auth-profiles.json file
    #   (used when running as a TrueKnot OpenClaw skill on trueknot@192.168.10.20).
    "claude_code_token_source": "keychain",
    "claude_code_openclaw_profile_path": None,
    "claude_code_openclaw_profile_name": "anthropic:default",
```

- [ ] **Step 5: Forward the keys in `_get_provider_kwargs()`**

In `tradingagents/graph/trading_graph.py`, update the `claude_code` branch (the `elif provider in ("anthropic", "claude_code"):` you already added in earlier work):

```python
        elif provider in ("anthropic", "claude_code"):
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

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
```

- [ ] **Step 6: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py -v
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py
```
Expected: 8 unit tests pass; smoke test still prints PONG.

- [ ] **Step 7: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tradingagents/default_config.py tradingagents/graph/trading_graph.py tests/test_claude_code_openclaw.py
git commit -m "feat(claude_code): config-driven token source through graph"
```

---

## Task 4: CLI argparse skeleton and main() entry

**Files:**
- Create: `cli/research.py`
- Create: `tests/test_research_cli.py`
- Modify: `pyproject.toml`

The CLI is independent of the heavy graph imports until it actually runs. Argparse first, behavior later.

- [ ] **Step 1: Write the failing test**

Create `tests/test_research_cli.py`:

```python
"""Tests for the headless tradingresearch CLI."""

import pytest

pytestmark = pytest.mark.unit


def test_help_includes_required_flags(capsys):
    from cli.research import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
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
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py -v
```
Expected: ModuleNotFoundError on `cli.research`.

- [ ] **Step 3: Create the parser**

Create `cli/research.py`:

```python
"""Headless CLI: run TradingAgents end-to-end and emit decision JSON + report files.

Designed to be invoked by an OpenClaw skill (TrueKnot trading-research). For
interactive use, see cli/main.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tradingresearch",
        description="Run a multi-agent equity research workflow on a ticker for a date.",
    )
    p.add_argument("--ticker", required=True, help="US-listed ticker symbol, e.g. NVDA.")
    p.add_argument("--date", required=True, help="Trade date YYYY-MM-DD (historical).")
    p.add_argument("--output-dir", required=True, help="Directory to write report files into.")

    p.add_argument("--deep", default="claude-sonnet-4-6", help="Deep-think model id.")
    p.add_argument("--quick", default="claude-haiku-4-5", help="Quick-think model id.")
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)

    p.add_argument(
        "--token-source", choices=("keychain", "openclaw_profile"), default="keychain",
        help="Where the claude_code provider reads the OAuth token from.",
    )
    p.add_argument(
        "--openclaw-profile-path",
        help="Path to OpenClaw auth-profiles.json (only when --token-source=openclaw_profile).",
    )
    p.add_argument(
        "--openclaw-profile-name", default="anthropic:default",
        help="Profile key inside auth-profiles.json (default: anthropic:default).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Behavior in subsequent tasks.
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Register the entrypoint**

In `pyproject.toml`, find the `[project.scripts]` section and add a second line:

```toml
[project.scripts]
tradingagents = "cli.main:app"
tradingresearch = "cli.research:main"
```

- [ ] **Step 6: Reinstall and verify the binary**

```bash
.venv/bin/pip install -e . --quiet
.venv/bin/tradingresearch --help
```
Expected: usage banner with all flags listed.

- [ ] **Step 7: Commit**

```bash
git add cli/research.py tests/test_research_cli.py pyproject.toml
git commit -m "feat(cli): tradingresearch headless argparse skeleton"
```

---

## Task 5: Streaming progress callback

**Files:**
- Create: `cli/research_progress.py`
- Create: `tests/test_research_progress.py`

LangGraph supports callbacks via the `BaseCallbackHandler` interface from `langchain_core.callbacks`. We hook node-start and node-end events.

- [ ] **Step 1: Write the failing test**

Create `tests/test_research_progress.py`:

```python
"""Tests for the progress callback used by tradingresearch CLI."""

from io import StringIO

import pytest

pytestmark = pytest.mark.unit


def test_callback_emits_done_lines():
    from cli.research_progress import ProgressCallback

    out = StringIO()
    cb = ProgressCallback(stream=out)
    cb.on_node_start("Market Analyst")
    cb.on_node_done("Market Analyst", duration_s=2.34)

    lines = out.getvalue().strip().splitlines()
    assert any("[Market Analyst] start" in line for line in lines)
    assert any("[Market Analyst] done" in line for line in lines)
    assert any("2.3" in line for line in lines)  # duration formatted to 1 dp


def test_callback_handles_unicode_node_names():
    from cli.research_progress import ProgressCallback

    out = StringIO()
    cb = ProgressCallback(stream=out)
    cb.on_node_done("Risk-Manager", duration_s=0.05)
    assert "[Risk-Manager] done" in out.getvalue()
```

- [ ] **Step 2: Run test to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_research_progress.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

Create `cli/research_progress.py`:

```python
"""Stdout progress reporter for tradingresearch.

Each node-completion line is consumed by an OpenClaw skill agent and
forwarded to Telegram (rate-limited on the OpenClaw side).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import IO


@dataclass
class ProgressCallback:
    stream: IO[str] = field(default_factory=lambda: sys.stdout)
    _node_starts: dict[str, float] = field(default_factory=dict)

    def on_node_start(self, node: str) -> None:
        self._node_starts[node] = time.monotonic()
        print(f"[{node}] start", file=self.stream, flush=True)

    def on_node_done(self, node: str, duration_s: float | None = None) -> None:
        if duration_s is None:
            started = self._node_starts.get(node, time.monotonic())
            duration_s = time.monotonic() - started
        print(f"[{node}] done ({duration_s:.1f}s)", file=self.stream, flush=True)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_research_progress.py -v
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add cli/research_progress.py tests/test_research_progress.py
git commit -m "feat(cli): progress callback emits per-node start/done lines"
```

---

## Task 6: Report file writers

**Files:**
- Create: `cli/research_writer.py`
- Create: `tests/test_research_writer.py`

Six markdown files plus one JSON dump per run. Pure functions over a final `AgentState` dict — no graph dependency, easy to test.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research_writer.py`:

```python
"""Tests for cli.research_writer.write_research_outputs."""

import json

import pytest

pytestmark = pytest.mark.unit


def _stub_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2024-05-10",
        "market_report": "## Market\n- price up 3%",
        "sentiment_report": "## Sentiment\n- bullish on social",
        "news_report": "## News\n- earnings beat",
        "fundamentals_report": "## Fundamentals\n- P/E elevated",
        "investment_debate_state": {
            "bull_history": "Bull: strong momentum",
            "bear_history": "Bear: valuation stretched",
            "judge_decision": "Manager: lean bull",
        },
        "risk_debate_state": {
            "aggressive_history": "Agg: max long",
            "neutral_history": "Neu: half size",
            "conservative_history": "Cons: skip",
            "judge_decision": "PM: BUY at half size — strong fundamentals, manageable risk",
        },
        "final_trade_decision": "BUY",
    }


def test_writes_all_expected_files(tmp_path):
    from cli.research_writer import write_research_outputs

    written = write_research_outputs(_stub_state(), str(tmp_path))

    expected = {
        "decision.md", "analyst_market.md", "analyst_social.md",
        "analyst_news.md", "analyst_fundamentals.md",
        "debate_bull_bear.md", "debate_risk.md", "state.json",
    }
    assert {p.name for p in written} == expected
    for p in written:
        assert p.read_text(), f"{p.name} is empty"


def test_decision_md_contains_action_and_pm_judgement(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    decision = (tmp_path / "decision.md").read_text()
    assert "BUY" in decision
    assert "PM:" in decision  # PM rationale carried forward
    assert "NVDA" in decision
    assert "2024-05-10" in decision


def test_state_json_round_trips(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert loaded["company_of_interest"] == "NVDA"
    assert loaded["final_trade_decision"] == "BUY"


def test_creates_output_dir_if_missing(tmp_path):
    from cli.research_writer import write_research_outputs

    out = tmp_path / "deep" / "nested" / "dir"
    write_research_outputs(_stub_state(), str(out))
    assert (out / "decision.md").exists()
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_research_writer.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the writer**

Create `cli/research_writer.py`:

```python
"""Write research outputs from a final AgentState dict to disk.

Output layout per run (under output_dir):
  decision.md
  analyst_market.md
  analyst_social.md
  analyst_news.md
  analyst_fundamentals.md
  debate_bull_bear.md
  debate_risk.md
  state.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _decision_md(state: dict[str, Any]) -> str:
    ticker = state.get("company_of_interest", "?")
    date = state.get("trade_date", "?")
    action = state.get("final_trade_decision", "UNKNOWN")
    pm_judgement = (
        state.get("risk_debate_state", {}).get("judge_decision", "").strip()
    )
    return (
        f"# {ticker} — {date}\n\n"
        f"**Decision:** {action}\n\n"
        f"## Portfolio Manager Rationale\n\n{pm_judgement}\n"
    )


def _analyst_md(title: str, body: str) -> str:
    return f"# {title}\n\n{body or '_(no report)_'}\n"


def _bull_bear_md(state: dict[str, Any]) -> str:
    debate = state.get("investment_debate_state", {})
    return (
        "# Bull vs Bear Debate\n\n"
        f"## Bull\n\n{debate.get('bull_history', '_(empty)_')}\n\n"
        f"## Bear\n\n{debate.get('bear_history', '_(empty)_')}\n\n"
        f"## Research Manager Decision\n\n{debate.get('judge_decision', '_(empty)_')}\n"
    )


def _risk_md(state: dict[str, Any]) -> str:
    risk = state.get("risk_debate_state", {})
    return (
        "# Risk Team Debate\n\n"
        f"## Aggressive\n\n{risk.get('aggressive_history', '_(empty)_')}\n\n"
        f"## Neutral\n\n{risk.get('neutral_history', '_(empty)_')}\n\n"
        f"## Conservative\n\n{risk.get('conservative_history', '_(empty)_')}\n\n"
        f"## Portfolio Manager Decision\n\n{risk.get('judge_decision', '_(empty)_')}\n"
    )


def write_research_outputs(state: dict[str, Any], output_dir: str) -> list[Path]:
    """Write all report files. Returns the list of paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, str]] = [
        ("decision.md", _decision_md(state)),
        ("analyst_market.md", _analyst_md("Market Analyst", state.get("market_report", ""))),
        ("analyst_social.md", _analyst_md("Social Sentiment Analyst", state.get("sentiment_report", ""))),
        ("analyst_news.md", _analyst_md("News Analyst", state.get("news_report", ""))),
        ("analyst_fundamentals.md", _analyst_md("Fundamentals Analyst", state.get("fundamentals_report", ""))),
        ("debate_bull_bear.md", _bull_bear_md(state)),
        ("debate_risk.md", _risk_md(state)),
    ]

    written: list[Path] = []
    for name, content in files:
        p = out / name
        p.write_text(content)
        written.append(p)

    state_path = out / "state.json"
    state_path.write_text(json.dumps(state, indent=2, default=str))
    written.append(state_path)
    return written
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_research_writer.py -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add cli/research_writer.py tests/test_research_writer.py
git commit -m "feat(cli): research_writer emits decision.md + analyst/debate reports"
```

---

## Task 7: Wire CLI to graph + writer + final JSON (happy path)

**Files:**
- Modify: `cli/research.py`
- Modify: `tests/test_research_cli.py`

Build the config from CLI args, run the graph (mocked in tests), call the writer, print final-decision JSON to stdout. The progress callback can be plumbed later — the writer is the deliverable.

- [ ] **Step 1: Add the failing integration test**

Append to `tests/test_research_cli.py`:

```python
def test_main_runs_graph_writes_files_prints_json(tmp_path, monkeypatch, capsys):
    """CLI wires args → config → graph.propagate → writer → stdout JSON."""
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

    # All 8 files written
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
```

(Add `import json` at the top of the test file if not already present.)

- [ ] **Step 2: Run test to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py::test_main_runs_graph_writes_files_prints_json -v
```
Expected: AttributeError on `TradingAgentsGraph` (not yet imported in `cli.research`).

- [ ] **Step 3: Update `cli/research.py` main()**

Replace the contents of `cli/research.py` with:

```python
"""Headless CLI: run TradingAgents end-to-end and emit decision JSON + report files."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.research_progress import ProgressCallback
from cli.research_writer import write_research_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tradingresearch",
        description="Run a multi-agent equity research workflow on a ticker for a date.",
    )
    p.add_argument("--ticker", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--deep", default="claude-sonnet-4-6")
    p.add_argument("--quick", default="claude-haiku-4-5")
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)

    p.add_argument(
        "--token-source", choices=("keychain", "openclaw_profile"), default="keychain",
    )
    p.add_argument("--openclaw-profile-path")
    p.add_argument("--openclaw-profile-name", default="anthropic:default")
    return p


def _build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "claude_code"
    config["deep_think_llm"] = args.deep
    config["quick_think_llm"] = args.quick
    config["max_debate_rounds"] = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds
    config["claude_code_token_source"] = args.token_source
    config["claude_code_openclaw_profile_path"] = args.openclaw_profile_path
    config["claude_code_openclaw_profile_name"] = args.openclaw_profile_name
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    progress = ProgressCallback()

    progress.on_node_start("research")
    started = time.monotonic()
    graph = TradingAgentsGraph(debug=False, config=config)
    final_state, _decision = graph.propagate(args.ticker, args.date)
    write_research_outputs(final_state, args.output_dir)
    duration = time.monotonic() - started
    progress.on_node_done("research", duration_s=duration)

    payload = {
        "decision": final_state.get("final_trade_decision", "UNKNOWN"),
        "ticker": args.ticker,
        "date": args.date,
        "output_dir": args.output_dir,
        "duration_s": round(duration, 1),
    }
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> Per-LangGraph-node progress (e.g. `[Market Analyst] done`) is intentionally deferred — wiring requires probing LangGraph's callback API and is outside Phase 1 scope. The ProgressCallback class is structured to support per-node events later without further refactoring.

- [ ] **Step 4: Run all CLI tests**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py tests/test_research_writer.py tests/test_research_progress.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cli/research.py tests/test_research_cli.py
git commit -m "feat(cli): wire tradingresearch main() to graph + writer + decision json"
```

---

## Task 8: CLI error handling and exit codes

**Files:**
- Modify: `cli/research.py`
- Modify: `tests/test_research_cli.py`

Per spec error-handling table: `ClaudeCodeAuthError` → exit 1, anything else → exit 2 with traceback to stderr.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_research_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py::test_auth_error_exits_1 tests/test_research_cli.py::test_unexpected_error_exits_2 -v
```
Expected: both fail (exception bubbles, no exit code).

- [ ] **Step 3: Wrap `main()` body in error handling**

In `cli/research.py`, replace the body of `main()`:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    progress = ProgressCallback()

    from tradingagents.llm_clients.claude_code_client import ClaudeCodeAuthError

    progress.on_node_start("research")
    started = time.monotonic()
    try:
        graph = TradingAgentsGraph(debug=False, config=config)
        final_state, _decision = graph.propagate(args.ticker, args.date)
        write_research_outputs(final_state, args.output_dir)
    except ClaudeCodeAuthError as e:
        print(f"auth error: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - top-level CLI catch
        traceback.print_exc(file=sys.stderr)
        return 2

    duration = time.monotonic() - started
    progress.on_node_done("research", duration_s=duration)
    payload = {
        "decision": final_state.get("final_trade_decision", "UNKNOWN"),
        "ticker": args.ticker,
        "date": args.date,
        "output_dir": args.output_dir,
        "duration_s": round(duration, 1),
    }
    print(json.dumps(payload), flush=True)
    return 0
```

- [ ] **Step 4: Run all CLI tests**

```bash
.venv/bin/python -m pytest tests/test_research_cli.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cli/research.py tests/test_research_cli.py
git commit -m "feat(cli): exit 1 on auth error, exit 2 on graph error"
```

---

## Task 9: Extend smoke_test_oauth.py for openclaw_profile

**Files:**
- Modify: `scripts/smoke_test_oauth.py`

Existing smoke proves the keychain path works. Add a `--source` flag so the same script can verify the openclaw_profile path against any auth-profiles.json the user can copy onto the MacBook for testing.

- [ ] **Step 1: Update the script**

Replace the contents of `scripts/smoke_test_oauth.py` with:

```python
"""Smoke test: confirm Claude Code OAuth credentials work end-to-end.

Default reads the macOS keychain. Pass --source openclaw_profile to instead
read an OpenClaw auth-profiles.json file (useful for verifying the
production path locally before deploying to mini).

Run from the repo root:
    .venv/bin/python scripts/smoke_test_oauth.py
    .venv/bin/python scripts/smoke_test_oauth.py \\
        --source openclaw_profile \\
        --path /tmp/auth-profiles.json --name anthropic:default
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    ClaudeCodeClient,
    get_oauth_token,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("keychain", "openclaw_profile"), default="keychain")
    parser.add_argument("--path", help="auth-profiles.json path (openclaw_profile only)")
    parser.add_argument("--name", default="anthropic:default")
    args = parser.parse_args()

    try:
        token = get_oauth_token(
            source=args.source,
            openclaw_profile_path=args.path,
            openclaw_profile_name=args.name,
        )
    except ClaudeCodeAuthError as e:
        print(f"[auth] {e}", file=sys.stderr)
        return 2
    print(f"[auth] {args.source} access_token loaded ({len(token)} chars)")

    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source=args.source,
        openclaw_profile_path=args.path,
        openclaw_profile_name=args.name,
    )
    llm = client.get_llm()
    print(f"[client] {type(llm).__name__} ready, model={client.model}")

    response = llm.invoke("Reply with just the single word: PONG")
    print(f"[response] content={response.content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify keychain path still works**

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py
```
Expected: prints PONG.

- [ ] **Step 3: Verify openclaw_profile path against a synthetic file**

```bash
cat > /tmp/auth-profiles.json <<'JSON'
{"version": 1, "profiles": {"anthropic:default": {"type": "token", "provider": "anthropic", "token": "sk-ant-oat01-NOT-REAL"}}}
JSON
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py --source openclaw_profile --path /tmp/auth-profiles.json
rm /tmp/auth-profiles.json
```
Expected: `[auth]` line confirms token loaded; the actual API call will 401 because the token is fake — that's fine, it proves the source plumbing works without needing a real production token.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test_oauth.py
git commit -m "test(smoke): smoke_test_oauth supports --source openclaw_profile"
```

---

## Task 10: Final regression sweep + push

**Files:**
- (No file changes; verification only.)

- [ ] **Step 1: Run the full new test suite**

```bash
.venv/bin/python -m pytest tests/test_claude_code_openclaw.py tests/test_research_cli.py tests/test_research_writer.py tests/test_research_progress.py -v
```
Expected: all pass (around 20 tests across 4 files).

- [ ] **Step 2: Confirm `tradingresearch --help` is wired**

```bash
.venv/bin/tradingresearch --help
```
Expected: usage banner with all flags.

- [ ] **Step 3: Push branch to fork**

```bash
git push origin feat/claude-code-oauth
```
Expected: branch updated on `https://github.com/SongKeat2901/TradingAgents/tree/feat/claude-code-oauth`.

- [ ] **Step 4: Tag the Phase 1 completion locally**

```bash
git tag phase1-openclaw-cli
```
(No push of the tag — keep it local until Phase 2 deploy verifies.)

---

## Phase 2-4 Deployment Runbook (not TDD tasks)

These steps happen on the trueknot Mac Mini (`192.168.10.20`, user `trueknot`) and in the trader agent's workspace. They aren't testable in the TDD sense — they are SSH + filesystem operations. Execute manually after Phase 1 ships and the smoke test passes.

### Phase 2 — Deploy CLI to trueknot@10.20

```bash
# from MacBook
ssh macmini-trueknot

# on the Mac Mini, as user trueknot
git clone https://github.com/SongKeat2901/TradingAgents.git ~/tradingagents
cd ~/tradingagents
python3.13 -m venv .venv
.venv/bin/pip install -e .

# Symlink the binary into PATH
mkdir -p ~/local/bin
ln -sf ~/tradingagents/.venv/bin/tradingresearch ~/local/bin/tradingresearch

# Verify
~/local/bin/tradingresearch --help

# Smoke against the host's keychain-written ~/.claude/.credentials.json
# (default --token-source=keychain; the Linux-fallback reader handles it).
~/local/bin/tradingresearch --ticker SPY --date 2024-05-10 \
    --output-dir /tmp/smoke-spy
cat /tmp/smoke-spy/decision.md
```

If the keychain path fails inside a daemon-spawned subprocess (Lesson #2 territory in OpenClawOps), fall back to:

```bash
~/local/bin/tradingresearch --ticker SPY --date 2024-05-10 \
    --output-dir /tmp/smoke-spy \
    --token-source openclaw_profile \
    --openclaw-profile-path ~/.openclaw/auth-profiles.json
```

If `~/.openclaw/auth-profiles.json` doesn't exist, look under `~/.openclaw/agents/<agent>/agent/auth-profiles.json` (per-agent layout).

### Phase 3 — Trader agent skill scaffolding

On the same host, create the new skill in the trader agent's workspace (NOT the admin's):

```bash
mkdir -p ~/.openclaw/workspace/skills/trading-research
cat > ~/.openclaw/workspace/skills/trading-research/SKILL.md <<'MD'
# Trading Research
Emoji: 📊
... (copy from spec section "Component 3") ...
MD
```

Append a `## Trading Research` block to `~/.openclaw/workspace/TOOLS.md` next to the existing `## IBKR Trader` and `## IBKR Fund` entries. **Append only — do not overwrite TOOLS.md**. Do not touch `~/.openclaw/workspace-admin/` (admin agent territory).

### Phase 4 — Telegram trigger

The trader agent is already on Telegram (`@TrueKnotBot`, supergroup `-1003753140043`). Send a message in that group:

```
research SPY 2024-05-10
```

The trader agent reads the new SKILL.md (may need a daemon kickstart on first deploy if the workspace cache is stale), runs `tradingresearch`, posts decision.md inline + attaches the analyst .md files.

Daemon kickstart (if the new skill isn't picked up):

```bash
ssh macmini-superqsp "sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway"
```

Tune progress-line cadence and decision.md format if needed; both are pure-data changes (no recompile of the daemon).

---

## Self-review notes

- All spec requirements are covered: token source (Tasks 1-3), CLI binary (Tasks 4-8), error handling (Task 8), output files (Task 6), smoke (Task 9). Phase 2-4 covered in runbook.
- Type/method names consistent across tasks: `_read_openclaw_profile`, `get_oauth_token(source=...)`, `ClaudeCodeClient(token_source=...)`, `write_research_outputs(state, output_dir)`, `ProgressCallback.on_node_start/done`.
- Spec sections deferred to future plans (reflection / outcome feedback, IBKR handoff, multi-ticker batch) are *not* in this plan, by design.
