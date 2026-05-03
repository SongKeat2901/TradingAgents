# TradingAgents under OpenClaw, triggered from Telegram

**Date:** 2026-05-03
**Status:** Draft
**Author:** songkeat (with Claude)

## Goal

Run the TradingAgents multi-agent research pipeline as a skill of an OpenClaw profile, triggered by Telegram messages, sharing OpenClaw's existing Anthropic OAuth so the user does not need a separate API key. The internal multi-agent debate (analysts → bull/bear → trader → risk team → portfolio manager) is preserved unchanged; only the trigger, auth source, and output surface change.

## Non-goals

- Replacing TradingAgents' multi-agent graph with a single OpenClaw conversation (rejected: collapses the adversarial debate dynamic).
- Per-agent sub-process spawning (rejected: per-spawn import overhead × 20-40 nodes).
- A standalone LLM proxy daemon (rejected: no behavioural gain over reading `OpenClaw/.env` directly for a single-user setup).
- Real-time per-token streaming to Telegram (defer: chunked progress markers are sufficient).
- Multi-user concurrency, queueing, fairness across requesters (defer: solo researcher use case).

## Architecture

```
Telegram message ─→ OpenClaw `trading` profile (Telegram channel, daemon)
                       │
                       │  parent agent reads
                       │  ~/.openclaw-trading/workspace/skills/trading-research/SKILL.md
                       │
                       ▼
                   Bash: tradingresearch \
                           --ticker NVDA --date 2024-05-10 \
                           --output-dir ~/.openclaw-trading/data/research/...
                       │
                       ▼
              ┌─ tradingresearch CLI (this fork) ─┐
              │   1. resolve OAuth from OpenClaw  │
              │      .env (TOKEN_TRUEKNOT or QSP) │
              │   2. build TradingAgentsGraph     │
              │      with llm_provider=claude_code│
              │   3. graph.propagate(ticker, date)│
              │   4. stream node-completion lines │
              │      to stdout                    │
              │   5. write report files to disk   │
              │   6. print final decision JSON    │
              └────────────────────────────────────┘
                       │
                       ▼
            stdout consumed by OpenClaw agent
                       │
                       ▼
            agent posts to Telegram:
              - inline: BUY/SELL/HOLD + 3-5 line rationale
              - attachments: full analyst .md files
```

## Components

### 1. `claude_code` provider — env source flag (TradingAgents fork)

The existing `tradingagents/llm_clients/claude_code_client.py` reads OAuth from the macOS keychain. Add a config-driven alternative source:

```python
# default_config.py
"claude_code_token_source": "keychain",   # "keychain" | "openclaw_env"
"claude_code_openclaw_env_path": "/Users/songkeat/Documents/Python/OpenClaw/.env",
"claude_code_openclaw_token_var": "TOKEN_TRUEKNOT",
```

`get_oauth_token()` branches on `claude_code_token_source`. The OpenClaw path reads the named env var from the file (without invoking the shell), validates it starts with `sk-ant-oat01-`, and returns it. No expiry check — OpenClaw owns the rotation cadence; if expired, the API call fails and the CLI surfaces the remediation hint ("rotate via OpenClaw token tooling").

**~30 lines added to claude_code_client.py.** Existing keychain path remains the default for non-OpenClaw use.

### 2. `tradingresearch` CLI (TradingAgents fork)

New entry point: `cli/research.py`, registered in `pyproject.toml` as `tradingresearch`. Single command, no interactive prompts (the existing `cli.main` is interactive; this one is for headless skill use).

```
tradingresearch --ticker SYM --date YYYY-MM-DD \
                --output-dir PATH \
                [--profile NAME] [--deep MODEL] [--quick MODEL] \
                [--debate-rounds N] [--risk-rounds N] \
                [--token-source openclaw_env|keychain]
                [--openclaw-env PATH] [--token-var NAME]
```

Behaviour:
1. Build `DEFAULT_CONFIG` overrides from flags. Force `llm_provider=claude_code`.
2. Construct `TradingAgentsGraph(config=...)`.
3. Subscribe a LangGraph callback that emits one stdout line per node entry/exit:
   `[market-analyst] start` / `[market-analyst] done (3.2s, 4 LLM calls)`.
4. Call `graph.propagate(ticker, date)`.
5. After the run, write to `<output-dir>/`:
   - `decision.md` — final BUY/SELL/HOLD + 5-line rationale (parsed from PM output)
   - `analyst_market.md`, `analyst_social.md`, `analyst_news.md`, `analyst_fundamentals.md`
   - `debate_bull_bear.md`, `debate_risk.md`
   - `state.json` — full final `AgentState` dump for inspection
6. Print final JSON to stdout: `{"decision": "BUY", "output_dir": "...", "duration_s": 412}`.
7. Exit code 0 on success, 1 on `ClaudeCodeAuthError`, 2 on graph runtime error.

Reuses existing `TradingAgentsGraph.propagate()` and result-writing helpers. New code is ~150 lines for the CLI plus the streaming callback.

### 3. OpenClaw `trading-research` skill

A new OpenClaw profile `trading` (parallel to `hht`, `lostnoob`) bound to a Telegram channel. Workspace at `~/.openclaw-trading/workspace/`.

**`skills/trading-research/SKILL.md`** — instructs the OpenClaw parent agent how to invoke the CLI, parse stdout progress lines for chat updates, and route the output files. Roughly:

```
# Trading Research
Emoji: 📊

## Purpose
Run a multi-agent equity research workflow on a ticker for a historical date,
returning a BUY/SELL/HOLD decision with the full analyst reports.

## Trigger
User says: "research <TICKER> [for <DATE>]" or "look at <TICKER>".
DATE defaults to today; ticker is uppercase US-listed symbol.

## Tool
Binary: /Users/gs/local/bin/tradingresearch
Invocation:
  tradingresearch --ticker $T --date $D \
    --output-dir ~/.openclaw-trading/data/research/$D-$T \
    --token-source openclaw_env

## Output handling
Stream stdout lines to chat as they arrive (rate-limit to one update per 30s).
On exit:
  - Send the contents of decision.md as a Telegram message.
  - Attach analyst_*.md and debate_*.md as documents.
  - On exit code 1: tell user to check Anthropic token rotation in OpenClaw.
  - On exit code 2: send last 30 lines of stderr.
```

**`workspace/TOOLS.md`** — appends an entry for `trading-research` per OpenClaw convention.

### 4. Output rendering rules

- `decision.md` ≤ 800 chars to fit comfortably in a single Telegram message.
- Each `analyst_*.md` is sent as a `.md` document attachment (Telegram permits any size up to 50 MB).
- Progress lines: only post `[node] done` lines to Telegram, suppress `start`. Reduces noise; user sees pipeline advancing.

### 5. Data layout

```
~/.openclaw-trading/
├── workspace/
│   ├── TOOLS.md
│   ├── IDENTITY.md / SOUL.md / USER.md   (OpenClaw conventions)
│   └── skills/trading-research/SKILL.md
├── data/
│   └── research/
│       └── 2024-05-10-NVDA/
│           ├── decision.md
│           ├── analyst_*.md
│           ├── debate_*.md
│           └── state.json
└── logs/
    └── tradingresearch-2026-05-03.log
```

`memory_log_path` for the reflection feature points at `~/.openclaw-trading/data/memory/trading_memory.md`. This is per-profile, isolated from any other OpenClaw instance.

## Data flow (one Telegram request)

1. User: `research NVDA 2024-05-10`
2. OpenClaw `trading` daemon receives Telegram message, routes to its agent.
3. Agent reads `SKILL.md`, decides this is a `trading-research` invocation.
4. Agent posts `📊 Running research on NVDA for 2024-05-10... (~5-15 min)` to chat.
5. Agent shells out to `tradingresearch ...`.
6. CLI loads OAuth from `OpenClaw/.env` (TOKEN_TRUEKNOT), builds graph, propagates.
7. Each LangGraph node-done event prints to stdout; agent forwards (rate-limited) to chat.
8. CLI exits with JSON; agent reads decision.md, attaches reports.
9. Final Telegram thread: status updates → decision summary → attached `.md` files.

## Error handling

| Failure | CLI behaviour | Skill behaviour |
|---|---|---|
| `OpenClaw/.env` missing or unreadable | Exit 1, `ClaudeCodeAuthError` with path | Tell user to check OpenClaw .env |
| Token var absent or wrong format | Exit 1 | Same |
| Anthropic 401 (token rotated/expired) | Exit 1 | Tell user to run OpenClaw token rotation, then retry |
| Anthropic 429 (rate limit) | CLI retries with backoff up to 3× per call; if still failing, exit 2 | Tell user "quota exhausted, try later" |
| Graph runtime error | Exit 2 with traceback to stderr | Send last 30 lines of stderr to chat |
| Telegram message > 4096 chars | n/a | Always send decision.md as both inline (truncated) AND attachment |

## Testing strategy

1. **Unit:** `tests/test_claude_code_client.py` adds cases for `openclaw_env` token source — env file present/absent, var present/absent, malformed token.
2. **CLI smoke:** `tests/test_cli_research_smoke.py` runs `tradingresearch` against a mocked graph that emits 3 fake nodes and a stub PM decision; asserts output files + JSON match.
3. **Live integration:** existing `scripts/smoke_test_oauth.py` extends to optionally pull token from OpenClaw env (`--source openclaw_env`).
4. **End-to-end:** manual — send a Telegram message to the `trading` profile with a small ticker (e.g. SPY for a recent date), verify the full chain.

No automated end-to-end test of the OpenClaw side; that lives in OpenClaw's own test setup.

## Implementation order

1. **Phase 1 — CLI + provider (TradingAgents fork only).** Add OpenClaw-env token source to claude_code client. Build `tradingresearch` CLI. Smoke test offline. Merge.
2. **Phase 2 — OpenClaw skill scaffolding.** Create `~/.openclaw-trading/` workspace. Write SKILL.md, TOOLS.md. Test by manually invoking `tradingresearch` from OpenClaw daemon shell.
3. **Phase 3 — Telegram wiring.** Configure the OpenClaw `trading` profile against Telegram channel, allowlist user. Run a real ticker.
4. **Phase 4 — Output polish.** Tune progress-line rate, decision.md format, attachment behaviour based on Telegram UX.

Each phase ships independently; phases 2-4 live in OpenClaw, not this fork.

## Open questions

- **Profile name:** `trading` proposed. Could share with HHT instead — but TradingAgents output is research, not chat ops, so a dedicated profile keeps message routing clean.
- **Date defaulting:** if user omits date in Telegram, default to *today's date* or *most recent trading day*? Latter is more useful for research.
- **Reflection persistence:** `ta.reflect_and_remember(returns)` writes to `memory_log_path`. Calling it requires a realized P&L number that the bot doesn't know yet. Defer to a separate "report outcome" Telegram command in a future spec.

## Deferred sub-projects (separate specs)

These were identified during brainstorm but are out of scope here:

- Multi-ticker batch research (e.g. portfolio sweeps).
- Scheduled cron triggers (daily morning watchlist research).
- Outcome-feedback loop for reflection memory.
- IBKR / Backtest project handoff (decision → paper-trade signal).
