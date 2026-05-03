# TradingAgents under OpenClaw, triggered from Telegram

**Date:** 2026-05-03
**Status:** Draft
**Author:** songkeat (with Claude)

## Goal

Run the TradingAgents multi-agent research pipeline as a skill of the existing **TrueKnot** trader agent (Mac Mini at `192.168.10.20`, macOS user `trueknot`, daemon `com.trueknot.openclaw.gateway` on port 18790), triggered by Telegram messages to `@TrueKnotBot`, sharing the host's existing Anthropic OAuth so the user does not need a separate API key. The internal multi-agent debate (analysts → bull/bear → trader → risk team → portfolio manager) is preserved unchanged; only the trigger, auth source, and output surface change.

The trueknot daemon currently hosts two agents on one process: the **trader** agent (Telegram, IBKR Auto-Trader — workspace `~/.openclaw/workspace/`) and the **admin** agent (WhatsApp — workspace `~/.openclaw/workspace-admin/`). The new `trading-research` skill is added to the **trader** agent's workspace. Existing trader skills (`ibkr-trader`, `ibkr-fund`) remain enabled; the rest of the openclaw skill bundle is intentionally trimmed and stays disabled.

## Non-goals

- Replacing TradingAgents' multi-agent graph with a single OpenClaw conversation (rejected: collapses the adversarial debate dynamic).
- Per-agent sub-process spawning (rejected: per-spawn import overhead × 20-40 nodes).
- A standalone LLM proxy daemon (rejected: no behavioural gain over reading `OpenClaw/.env` directly for a single-user setup).
- Real-time per-token streaming to Telegram (defer: chunked progress markers are sufficient).
- Multi-user concurrency, queueing, fairness across requesters (defer: solo researcher use case).

## Architecture

All paths below are on the **trueknot Mac Mini** (`192.168.10.20`, SSH alias `macmini-trueknot`, macOS user `trueknot`) where the openclaw daemon runs. The CLI, the workspace skill files, and the data dir all live there. The user's MacBook is the management terminal; SSH/git deploys updates.

```
Telegram message ─→ @TrueKnotBot in supergroup -1003753140043
                       │
                       │  trader agent reads
                       │  /Users/trueknot/.openclaw/workspace/skills/trading-research/SKILL.md
                       │
                       ▼
                   Bash: tradingresearch \
                           --ticker NVDA --date 2024-05-10 \
                           --output-dir /Users/trueknot/.openclaw/data/research/...
                       │
                       ▼
              ┌─ tradingresearch CLI (this fork, deployed on 10.20) ─┐
              │   1. resolve OAuth (see "Component 1" — keychain      │
              │      via ~/.claude/.credentials.json OR openclaw      │
              │      auth-profiles.json)                              │
              │   2. build TradingAgentsGraph                         │
              │      with llm_provider=claude_code                    │
              │   3. graph.propagate(ticker, date)                    │
              │   4. stream node-completion lines to stdout           │
              │   5. write report files to disk                       │
              │   6. print final decision JSON                        │
              └───────────────────────────────────────────────────────┘
                       │
                       ▼
            stdout consumed by the trader agent
                       │
                       ▼
            agent posts to Telegram:
              - inline: BUY/SELL/HOLD + 3-5 line rationale
              - attachments: full analyst .md files
```

The internal TradingAgents quick/deep model split (Sonnet 4.6 / Haiku 4.5) is independent of the trader agent's parent model; both share the same Anthropic OAuth, just different model IDs.

## Components

### 1. `claude_code` provider — OpenClaw profile source (TradingAgents fork)

The existing `tradingagents/llm_clients/claude_code_client.py` reads OAuth from two sources already (macOS keychain on Darwin, `~/.claude/.credentials.json` on Linux). The trueknot Mac Mini stores the host's Anthropic OAuth at `/Users/trueknot/.claude/.credentials.json` (per `OpenClawOps/trueknot-runbook.md` — written from keychain at boot via Lesson #2). The existing keychain code path therefore Just Works on this host: on macOS it tries `security`, on the off-chance the keychain path fails, the Linux fallback reads `.credentials.json`. So **for the trueknot deploy, no config changes are strictly required** — the default `token_source: "keychain"` already covers it.

A second supported source — OpenClaw's native `auth-profiles.json` — is provided for OpenClaw deployments that prefer to drive token rotation from the daemon's own credential store. Three config keys (already shipped in Phase 1):

```python
# default_config.py
"claude_code_token_source": "keychain",   # "keychain" | "openclaw_profile"
"claude_code_openclaw_profile_path": None,                # default: None
"claude_code_openclaw_profile_name": "anthropic:default",
```

`get_oauth_token()` branches on `claude_code_token_source`. The `openclaw_profile` path:

1. Reads the JSON file (format `{"profiles": {"anthropic:default": {"token": "sk-ant-oat01-..."}}}`).
2. Looks up `profiles[<profile_name>].token`.
3. Validates the token starts with `sk-ant-oat01-`.
4. Returns it.

No expiry check. If the token is stale, the Anthropic call returns 401 and the CLI surfaces a remediation hint.

**Already implemented in Phase 1.** Phase 2 deployment can pick either source — keychain default is fine for trueknot@10.20. If the keychain unlock isn't reliably alive in the daemon-spawned subprocess (Lesson #2 territory), switch to `openclaw_profile` pointing at `/Users/trueknot/.openclaw/auth-profiles.json`.

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

### 3. OpenClaw `trading-research` skill (under the trader agent)

Adds a new skill directory under the trader agent's workspace at `/Users/trueknot/.openclaw/workspace/skills/trading-research/`. The trueknot daemon already runs (`com.trueknot.openclaw.gateway` on port 18790) and the trader agent is already bound to Telegram (`@TrueKnotBot`, group `-1003753140043`) — no new daemon, no new agent needed.

**`skills/trading-research/SKILL.md`** — instructs the trader agent how to invoke the CLI, parse stdout progress lines for chat updates, and route the output files. Roughly:

```
# Trading Research
Emoji: 📊

## Purpose
Run a multi-agent equity research workflow on a ticker for a historical date,
returning a BUY/SELL/HOLD decision with the full analyst reports. Distinct
from the existing ibkr-trader skill (which talks to the live IBKR backend).

## Trigger
User says: "research <TICKER> [for <DATE>]" or "look at <TICKER>".
DATE defaults to most recent trading day; ticker is uppercase US-listed symbol.

## Tool
Binary: /Users/trueknot/local/bin/tradingresearch
Invocation (keychain auth — default):
  tradingresearch --ticker $T --date $D \
    --output-dir /Users/trueknot/.openclaw/data/research/$D-$T

(If keychain isn't reliably accessible from the daemon subprocess, add:
  --token-source openclaw_profile \
  --openclaw-profile-path /Users/trueknot/.openclaw/auth-profiles.json)

## Output handling
Stream stdout lines to chat as they arrive (rate-limit to one update per 30s).
On exit:
  - Send the contents of decision.md as a Telegram message.
  - Attach analyst_*.md and debate_*.md as documents.
  - On exit code 1: tell user the OAuth needs refresh.
  - On exit code 2: send last 30 lines of stderr.

## Coexistence with ibkr-trader
This skill is research-only — it never opens or modifies positions. If the
user asks to act on the decision (buy/sell), hand off to the ibkr-trader skill.
```

**`workspace/TOOLS.md`** — appends an entry for `trading-research` (alongside the existing `ibkr-trader` and `ibkr-fund` entries). The other ~52 bundled openclaw skills are explicitly disabled per the trueknot skill-trim pattern; this addition does not change that.

### 4. Output rendering rules

- `decision.md` ≤ 800 chars to fit comfortably in a single Telegram message.
- Each `analyst_*.md` is sent as a `.md` document attachment (Telegram permits any size up to 50 MB).
- Progress lines: only post `[node] done` lines to Telegram, suppress `start`. Reduces noise; user sees pipeline advancing.

### 5. Data layout (on `trueknot@192.168.10.20`)

```
/Users/trueknot/
├── .claude/
│   └── .credentials.json                 (existing — host OAuth, written from keychain)
├── .openclaw/
│   ├── auth-profiles.json                (existing — alternative token source)
│   ├── workspace/                        (existing — trader agent's workspace)
│   │   ├── TOOLS.md                      (existing — we append a section)
│   │   ├── IDENTITY.md / SOUL.md / USER.md / AGENTS.md / HEARTBEAT.md / MEMORY.md
│   │   ├── memory/                       (existing)
│   │   └── skills/
│   │       ├── ibkr-trader/              (existing — keep)
│   │       ├── ibkr-fund/                (existing — keep)
│   │       └── trading-research/         ← new
│   │           └── SKILL.md
│   ├── workspace-admin/                  (existing — admin agent on WhatsApp; NOT touched)
│   ├── data/
│   │   └── research/                     ← new
│   │       └── 2024-05-10-NVDA/
│   │           ├── decision.md
│   │           ├── analyst_*.md
│   │           ├── debate_*.md
│   │           └── state.json
│   └── logs/
│       ├── gateway.log                   (existing daemon log)
│       └── tradingresearch-2026-05-03.log ← new
├── tradingagents/                        ← new — fork clone target
│   ├── .venv/
│   └── ...
└── local/bin/tradingresearch             ← new — symlink to .venv entrypoint
```

`memory_log_path` for the reflection feature points at `/Users/trueknot/.openclaw/data/memory/trading_memory.md`.

**Deployment from MacBook to trueknot@10.20:** SSH via `macmini-trueknot`, `git clone https://github.com/SongKeat2901/TradingAgents.git ~/tradingagents`, `python3.13 -m venv ~/tradingagents/.venv && ~/tradingagents/.venv/bin/pip install -e ~/tradingagents`, `mkdir -p ~/local/bin && ln -sf ~/tradingagents/.venv/bin/tradingresearch ~/local/bin/tradingresearch`. The skill files (SKILL.md, TOOLS.md append) are deployed separately into `~/.openclaw/workspace/`. No system-domain LaunchDaemon changes — the trader agent picks up the new skill on next message-handling pass (or restart via `sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway` from `macmini-superqsp` if needed).

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
2. **Phase 2 — Deploy to trueknot@10.20.** SSH via `macmini-trueknot`, `git clone https://github.com/SongKeat2901/TradingAgents.git ~/tradingagents`, build venv, `pip install -e .`, symlink `~/local/bin/tradingresearch`. Verify with `tradingresearch --help` and a small smoke run against `~/.claude/.credentials.json` (default keychain mode — no flag changes needed). Fall back to `--token-source openclaw_profile` only if keychain access from the daemon subprocess fails.
3. **Phase 3 — Trader agent skill scaffolding.** On 10.20, create `/Users/trueknot/.openclaw/workspace/skills/trading-research/SKILL.md`, append a section to existing `TOOLS.md` (which already lists `ibkr-trader` and `ibkr-fund`). Test by manually invoking `tradingresearch` and confirming output files land in `~/.openclaw/data/research/`.
4. **Phase 4 — Telegram trigger.** No daemon changes needed (`@TrueKnotBot` is already running). Send a message like `research SPY 2024-05-10` to the supergroup; verify the trader agent picks up SKILL.md, runs the CLI, and posts results back. Tune progress-line cadence and decision.md format from real chat behaviour. If the agent doesn't pick up the new skill on message receipt, kickstart the daemon: `ssh macmini-superqsp "sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway"`.

Each phase ships independently; Phase 1 is fork-only, Phases 2-4 are deployment + OpenClaw configuration on the trueknot host.

## Open questions

- **Auth source choice:** keychain default (via `~/.claude/.credentials.json`) is expected to work since OpenClawOps Lesson #2 already writes that file at boot. If empirically it doesn't, switch to `--token-source openclaw_profile` pointing at `~/.openclaw/auth-profiles.json` (top-level) or, if the trader agent's tokens live per-agent, `~/.openclaw/agents/main/agent/auth-profiles.json`. Decide during Phase 2 smoke.
- **Date defaulting:** confirmed — most recent trading day when omitted (more useful for research than today's date, which may be a non-trading day).
- **Reflection persistence:** `ta.reflect_and_remember(returns)` writes to `memory_log_path`. Calling it requires a realized P&L number that the bot doesn't know yet. Defer to a separate "report outcome" Telegram command in a future spec.

## Deferred sub-projects (separate specs)

These were identified during brainstorm but are out of scope here:

- Multi-ticker batch research (e.g. portfolio sweeps).
- Scheduled cron triggers (daily morning watchlist research).
- Outcome-feedback loop for reflection memory.
- IBKR / Backtest project handoff (decision → paper-trade signal).
