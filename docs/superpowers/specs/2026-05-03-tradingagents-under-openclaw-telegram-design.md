# TradingAgents under OpenClaw, triggered from Telegram

**Date:** 2026-05-03
**Status:** Draft
**Author:** songkeat (with Claude)

## Goal

Run the TradingAgents multi-agent research pipeline as a skill of the existing **TrueKnot** OpenClaw profile (Farm 1 Mac Mini, `/Users/gs/.openclaw-trueknot/`), triggered by Telegram messages, sharing OpenClaw's existing Anthropic OAuth so the user does not need a separate API key. The internal multi-agent debate (analysts → bull/bear → trader → risk team → portfolio manager) is preserved unchanged; only the trigger, auth source, and output surface change.

The new skill coexists with TrueKnot's existing scheduled tasks (`trueknot-daily-brief`, `trueknot-weekly-review`, `trueknot-monday-brief`); it does not replace them.

## Non-goals

- Replacing TradingAgents' multi-agent graph with a single OpenClaw conversation (rejected: collapses the adversarial debate dynamic).
- Per-agent sub-process spawning (rejected: per-spawn import overhead × 20-40 nodes).
- A standalone LLM proxy daemon (rejected: no behavioural gain over reading `OpenClaw/.env` directly for a single-user setup).
- Real-time per-token streaming to Telegram (defer: chunked progress markers are sufficient).
- Multi-user concurrency, queueing, fairness across requesters (defer: solo researcher use case).

## Architecture

All paths below are on **Farm 1 Mac Mini** (`mini`, user `gs`) where the TrueKnot daemon runs. The CLI, the workspace skill files, and the data dir all live on mini. The user's MacBook is the management terminal; SSH/rsync deploys updates to mini.

```
Telegram message ─→ OpenClaw `trueknot` profile (Telegram channel, daemon on mini)
                       │
                       │  parent agent (Opus 4.6) reads
                       │  /Users/gs/.openclaw-trueknot/workspace/skills/trading-research/SKILL.md
                       │
                       ▼
                   Bash: tradingresearch \
                           --ticker NVDA --date 2024-05-10 \
                           --output-dir /Users/gs/.openclaw-trueknot/data/research/...
                       │
                       ▼
              ┌─ tradingresearch CLI (this fork, deployed on mini) ─┐
              │   1. resolve OAuth from                              │
              │      /Users/gs/.openclaw-trueknot/auth-profiles.json │
              │      (profile "anthropic:default")                   │
              │   2. build TradingAgentsGraph                        │
              │      with llm_provider=claude_code                   │
              │   3. graph.propagate(ticker, date)                   │
              │   4. stream node-completion lines to stdout          │
              │   5. write report files to disk                      │
              │   6. print final decision JSON                       │
              └──────────────────────────────────────────────────────┘
                       │
                       ▼
            stdout consumed by TrueKnot parent agent
                       │
                       ▼
            agent posts to Telegram:
              - inline: BUY/SELL/HOLD + 3-5 line rationale
              - attachments: full analyst .md files
```

The internal TradingAgents quick/deep model split (Sonnet 4.6 / Haiku 4.5) is independent of the TrueKnot parent agent's model (Opus 4.6); both share the same OAuth, just different model IDs.

## Components

### 1. `claude_code` provider — OpenClaw profile source (TradingAgents fork)

The existing `tradingagents/llm_clients/claude_code_client.py` reads OAuth from the macOS keychain. Add a config-driven alternative that reads from OpenClaw's `auth-profiles.json` (the native token store on each OpenClaw host):

```python
# default_config.py
"claude_code_token_source": "keychain",   # "keychain" | "openclaw_profile"
"claude_code_openclaw_profile_path":
    "/Users/gs/.openclaw-trueknot/auth-profiles.json",
"claude_code_openclaw_profile_name": "anthropic:default",
```

`get_oauth_token()` branches on `claude_code_token_source`. The `openclaw_profile` path:

1. Reads the JSON file (format documented in `OpenClaw/docs/tokens.md` — `{"profiles": {"anthropic:default": {"token": "sk-ant-oat01-..."}}}`).
2. Looks up `profiles[<profile_name>].token`.
3. Validates the token starts with `sk-ant-oat01-`.
4. Returns it.

No expiry check — OpenClaw's `update-tokens.sh` owns rotation cadence (one cron away on the user's MacBook). If the token is stale, the Anthropic call returns 401 and the CLI surfaces a remediation hint ("rotate via `OpenClaw/update-tokens.sh`").

**~30 lines added to claude_code_client.py.** Existing keychain path remains the default for local non-OpenClaw use.

### 2. `tradingresearch` CLI (TradingAgents fork)

New entry point: `cli/research.py`, registered in `pyproject.toml` as `tradingresearch`. Single command, no interactive prompts (the existing `cli.main` is interactive; this one is for headless skill use).

```
tradingresearch --ticker SYM --date YYYY-MM-DD \
                --output-dir PATH \
                [--deep MODEL] [--quick MODEL] \
                [--debate-rounds N] [--risk-rounds N] \
                [--token-source openclaw_profile|keychain] \
                [--openclaw-profile-path PATH] \
                [--openclaw-profile-name NAME]
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

### 3. OpenClaw `trading-research` skill (under TrueKnot)

Adds a new skill directory under the existing TrueKnot workspace at `/Users/gs/.openclaw-trueknot/workspace/skills/trading-research/`. The TrueKnot daemon already runs and is bound to its Telegram chat — no new profile or daemon needed.

**`skills/trading-research/SKILL.md`** — instructs the TrueKnot parent agent how to invoke the CLI, parse stdout progress lines for chat updates, and route the output files. Roughly:

```
# Trading Research
Emoji: 📊

## Purpose
Run a multi-agent equity research workflow on a ticker for a historical date,
returning a BUY/SELL/HOLD decision with the full analyst reports.

## Trigger
User says: "research <TICKER> [for <DATE>]" or "look at <TICKER>".
DATE defaults to most recent trading day; ticker is uppercase US-listed symbol.

## Tool
Binary: /Users/gs/local/bin/tradingresearch
Invocation:
  tradingresearch --ticker $T --date $D \
    --output-dir /Users/gs/.openclaw-trueknot/data/research/$D-$T \
    --token-source openclaw_profile \
    --openclaw-profile-path /Users/gs/.openclaw-trueknot/auth-profiles.json

## Output handling
Stream stdout lines to chat as they arrive (rate-limit to one update per 30s).
On exit:
  - Send the contents of decision.md as a Telegram message.
  - Attach analyst_*.md and debate_*.md as documents.
  - On exit code 1: tell user to rotate Anthropic token via update-tokens.sh.
  - On exit code 2: send last 30 lines of stderr.
```

**`workspace/TOOLS.md`** — appends an entry for `trading-research` (alongside the existing TrueKnot scheduled-task entries). This is the registry the parent agent reads on startup.

### 4. Output rendering rules

- `decision.md` ≤ 800 chars to fit comfortably in a single Telegram message.
- Each `analyst_*.md` is sent as a `.md` document attachment (Telegram permits any size up to 50 MB).
- Progress lines: only post `[node] done` lines to Telegram, suppress `start`. Reduces noise; user sees pipeline advancing.

### 5. Data layout (on Farm 1 mini)

```
/Users/gs/.openclaw-trueknot/
├── auth-profiles.json                    (existing — read by our CLI)
├── workspace/                            (existing TrueKnot workspace)
│   ├── TOOLS.md                          (existing — we append a section)
│   ├── IDENTITY.md / SOUL.md / USER.md   (existing)
│   ├── memory/                           (existing)
│   └── skills/
│       ├── trading-research/             ← new
│       │   └── SKILL.md
│       └── ...                           (existing trueknot-* skills coexist)
├── data/
│   └── research/                         ← new
│       └── 2024-05-10-NVDA/
│           ├── decision.md
│           ├── analyst_*.md
│           ├── debate_*.md
│           └── state.json
└── logs/
    └── tradingresearch-2026-05-03.log    ← new
```

`memory_log_path` for the reflection feature points at `/Users/gs/.openclaw-trueknot/data/memory/trading_memory.md`. Isolated from other OpenClaw instances.

**Deployment from MacBook to mini:** `rsync` or git-pull the TradingAgents fork to mini (e.g. `/Users/gs/local/src/TradingAgents`), `pip install .` into a venv there, symlink the `tradingresearch` entry-point into `/Users/gs/local/bin/`. Workspace files (SKILL.md, TOOLS.md edit) are deployed separately into `~/.openclaw-trueknot/workspace/`.

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
| `auth-profiles.json` missing or unreadable | Exit 1, `ClaudeCodeAuthError` with path | Tell user to check OpenClaw token deploy |
| Profile name absent or token malformed | Exit 1 | Same |
| Anthropic 401 (token rotated/expired) | Exit 1 | Tell user to run `update-tokens.sh` from MacBook, then retry |
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

1. **Phase 1 — CLI + provider (TradingAgents fork only, work on MacBook).** Add `openclaw_profile` token source to claude_code client. Build `tradingresearch` CLI. Smoke test on MacBook against the local Claude Code keychain. Merge to fork.
2. **Phase 2 — Deploy to mini.** SSH to mini, clone the fork to `/Users/gs/local/src/TradingAgents`, create venv, `pip install .`, symlink `tradingresearch` into `/Users/gs/local/bin/`. Verify the binary runs and reads `auth-profiles.json` correctly.
3. **Phase 3 — TrueKnot skill scaffolding.** On mini, create `/Users/gs/.openclaw-trueknot/workspace/skills/trading-research/SKILL.md`, append a section to existing `TOOLS.md`. Test by manually invoking `tradingresearch` and confirming output files land in the expected place.
4. **Phase 4 — Telegram trigger.** No daemon changes needed (TrueKnot's Telegram bot is already running). Send a Telegram message like `research SPY 2024-05-10`; verify the parent agent picks up SKILL.md, runs the CLI, and posts results back. Tune progress-line cadence and decision.md format from real chat behaviour.

Each phase ships independently; Phase 1 is fork-only, Phases 2-4 are deployment + OpenClaw configuration.

## Open questions

- **Auth-profiles.json location on mini:** docs note it can live at `~/.openclaw-trueknot/auth-profiles.json` (top-level, Farm 2 style) OR `~/.openclaw-trueknot/agents/<agent>/agent/auth-profiles.json` (per-agent). Verify the actual path on Farm 1 mini in Phase 2; CLI flag `--openclaw-profile-path` lets us point at either.
- **Date defaulting:** confirmed — most recent trading day when omitted (more useful for research than today's date, which may be a non-trading day).
- **Reflection persistence:** `ta.reflect_and_remember(returns)` writes to `memory_log_path`. Calling it requires a realized P&L number that the bot doesn't know yet. Defer to a separate "report outcome" Telegram command in a future spec.

## Deferred sub-projects (separate specs)

These were identified during brainstorm but are out of scope here:

- Multi-ticker batch research (e.g. portfolio sweeps).
- Scheduled cron triggers (daily morning watchlist research).
- Outcome-feedback loop for reflection memory.
- IBKR / Backtest project handoff (decision → paper-trade signal).
