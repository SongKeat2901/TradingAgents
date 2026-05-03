# Phase 5: Rate-limit-tolerant redesign

**Date:** 2026-05-03
**Status:** Draft
**Builds on:** Phase 1 spec/plan in this directory; Phase 2-4 deployment to `trueknot@192.168.10.20`.

## Goal

Make the `tradingresearch` binary survive subscription-OAuth rate-limit pressure on its own, without depending on a wrapper script that the trader agent might bypass and without contending with the OpenClaw parent agent's `claude-cli` session for output-tokens-per-minute on the same Anthropic account.

## Background — what failed in Phase 4 testing

A single end-to-end run hit Anthropic 429 (`rate_limit_error`, no detail in the body) at the **Research Manager** node — the first deep `claude-sonnet-4-6` call after ~12 quick `claude-haiku-4-5` analyst calls. Two factors combined:

1. **Concurrent Sonnet sessions on one OAuth account.** While the `tradingresearch` binary was running, the OpenClaw trader agent's own `claude-cli` session was also alive (waiting for the bash subprocess to return). Both used `claude-sonnet-4-6`. The output-tokens-per-minute burst from two simultaneous Sonnet streams exceeded the burst ceiling for `default_claude_max_20x`.
2. **Bash-synthesis bypass of the wrapper.** The trader agent rewrote our `tradingresearch-bg` invocation into its own variant (`tradingresearch ... 2>"$OUTDIR/stderr.log"`), dropping `--telegram-notify`, `nohup`, `&`, and `disown`. Without those, the binary ran synchronously, the parent agent's claude-cli stayed alive throughout the run, and concurrency-on-one-account was the unavoidable result.

## Non-goals

- Reducing the total number of LLM calls (~20 per ticker). Pacing them, not removing them.
- Switching to an Anthropic API key. Subscription-OAuth stays the auth path.
- Replacing analysts with local models (Ollama). Available as a future option if quota becomes a recurring blocker even with pacing.
- Reducing graph complexity (dropping analysts or risk team). Keeps the existing 60K-star agent design.
- Multi-tenant safety. Single-user, one account, one host.

## Architecture changes

### 1. Self-daemonizing binary (`--telegram-notify` triggers fork)

When `--telegram-notify CHAT_ID` is passed and `TRADINGRESEARCH_BOT_TOKEN` is in the environment, the binary calls `os.fork()` + `os.setsid()` immediately on entry. The parent prints `started pid=<N>` to stdout and exits 0. The child redirects stdin/stdout/stderr to a per-run log file under `~/.openclaw/logs/tradingresearch-<DATE>-<TICKER>.log` and continues execution.

Effect: any caller — wrapper, agent's synth-bash, cron, manual SSH — gets fire-and-forget for free. The bash-synthesis vulnerability is eliminated because there's nothing to bypass; the binary self-detaches no matter how it's invoked.

A `--no-daemonize` flag exists for unit tests and direct foreground use.

### 2. Inter-call pacing via shared `InMemoryRateLimiter`

A `--pacing-seconds N` CLI flag (default `3`) configures a `langchain_core.rate_limiters.InMemoryRateLimiter` shared between the deep and quick `ChatAnthropic` instances:

```python
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1.0 / args.pacing_seconds,
    check_every_n_seconds=0.1,
    max_bucket_size=2,
)
```

The same instance is passed as the `rate_limiter` kwarg to both `_OAuthChatAnthropic` constructions inside `ClaudeCodeClient.get_llm()`. Because both LLM instances share the same limiter object, pacing is account-wide.

Effect: at default 3s pacing, ~20 calls become ~60s of forced idle in addition to natural call duration → calls spread across ~5-7 minutes. Output-tokens-per-minute stays well under burst limits.

### 3. 429 retry with exponential backoff

`ClaudeCodeClient.get_llm()` already passes `max_retries` from `_PASSTHROUGH_KWARGS` to `ChatAnthropic`. The default has been `None` (single attempt). Phase 5 sets a sensible default of `3` retries when not explicitly overridden. The Anthropic SDK's built-in backoff handles 429s with exponential delay (~5s → ~30s → ~120s).

Effect: a transient burst that slips past pacing recovers on its own without the run failing.

### 4. Model defaults locked down with a comment

`default_config.py` and the CLI flags already default to:
- `deep_think_llm=claude-sonnet-4-6` (only the 2 judges)
- `quick_think_llm=claude-haiku-4-5` (everything else)

Add a comment in `default_config.py` warning future tinkerers that mixing two Sonnet tiers (or upgrading quick to Sonnet) materially raises burst risk and should only be done on Pro+ tiers if at all. No code change beyond the comment.

### 5. SKILL.md simplification, wrapper retirement

Because the binary self-detaches, the `tradingresearch-bg` wrapper script is redundant. Phase 5:
- Removes `~/local/bin/tradingresearch-bg` from the trueknot host.
- Restores `~/local/bin/tradingresearch` symlinking to the venv entrypoint.
- Rewrites `~/.openclaw/workspace/skills/trading-research/SKILL.md` to instruct the agent to run `tradingresearch --ticker $T --date $D --output-dir ... --telegram-notify -1003753140043` directly, with `TRADINGRESEARCH_BOT_TOKEN` exported via `jq` from `openclaw.json`.

The agent can still synthesize its own bash variation; the binary self-detaches regardless.

## Files to change

**Created:**
- `tests/test_research_daemonize.py` — fork-mock tests for the daemonize path.

**Modified:**
- `cli/research.py` — add `--pacing-seconds`, `--no-daemonize` flags; daemonize-on-`--telegram-notify` early in `main()`.
- `tradingagents/llm_clients/claude_code_client.py` — accept a `rate_limiter` kwarg, forward it to both `_OAuthChatAnthropic` constructors.
- `tradingagents/graph/trading_graph.py` — when `claude_code` provider, build a shared `InMemoryRateLimiter` from `pacing_seconds` config and pass to `create_llm_client`.
- `tradingagents/default_config.py` — add `pacing_seconds: 3` and a model-locking comment.
- `tests/test_research_cli.py` — add tests for the new flags.
- `tests/test_claude_code_openclaw.py` — add a test that the rate_limiter kwarg flows through.

**Deployed (Phase 5 deployment runbook):**
- Replace `~/.openclaw/workspace/skills/trading-research/SKILL.md` with the simplified version.
- Remove `~/local/bin/tradingresearch-bg`. Restore `~/local/bin/tradingresearch` symlink to `~/tradingagents/.venv/bin/tradingresearch`.

## Test strategy

| Concern | Test |
|---|---|
| Daemonize forks on `--telegram-notify` | Monkeypatch `os.fork` to return `(parent_path, child_path)` controlled values; assert parent prints `started pid=` and exits before reaching graph construction. |
| `--no-daemonize` skips fork | Monkeypatch `os.fork` to assert it is never called. |
| Pacing flag wires `InMemoryRateLimiter` | Construct `ClaudeCodeClient` with `rate_limiter=<sentinel>`; assert it lands on the underlying `ChatAnthropic`. |
| `pacing_seconds` default = 3 | Read from `DEFAULT_CONFIG`. |
| `max_retries` default = 3 | Read from client kwargs after construction. |
| 429 retry path | Skipped (would require Anthropic SDK internal mocking that's brittle). The Anthropic SDK's own tests cover backoff; we just trust it works once we set `max_retries`. |

## Deployment runbook (Phase 5 deploy on trueknot)

```bash
# from MacBook
ssh macmini-trueknot
cd ~/tradingagents
git pull origin main
.venv/bin/pip install -e . --quiet  # only if pyproject changed; usually no

# Restore bare binary symlink, retire the wrapper
ln -sf ~/tradingagents/.venv/bin/tradingresearch ~/local/bin/tradingresearch
rm -f ~/local/bin/tradingresearch-bg

# Help reflects new flags
~/local/bin/tradingresearch --help | grep -E "pacing-seconds|no-daemonize"
```

Then deploy the new SKILL.md (scp from MacBook with the simplified contents) and kickstart the daemon to refresh the skill cache:

```bash
# from MacBook
scp /tmp/skill-phase5.md macmini-trueknot:.openclaw/workspace/skills/trading-research/SKILL.md
ssh macmini-superqsp "sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway"
```

## Verification (after deploy, with quota fresh)

1. Manual smoke from SSH on trueknot:
   ```bash
   TRADINGRESEARCH_BOT_TOKEN=$(jq -r .channels.telegram.accounts.default.botToken ~/.openclaw/openclaw.json) \
     ~/local/bin/tradingresearch \
       --ticker SPY --date 2024-05-10 \
       --output-dir ~/.openclaw/data/research/2024-05-10-SPY \
       --telegram-notify -1003753140043
   ```
   Expected: prints `started pid=<N>`, returns immediately. Tail the log for progress; ~6-8 min later, decision posts to Telegram.

2. Agent test via CLI:
   ```bash
   openclaw agent -m "research SPY 2024-05-10" --agent trader
   ```
   Expected: agent replies "kicking off" within seconds; binary runs in the background regardless of how the agent crafted the bash; Telegram receives the result.

3. End-to-end via real Telegram (you sending the trigger from your phone): same outcome.

## Open questions

- **Whether 3s pacing is actually low enough.** Empirical — may need tuning to 5s or 10s on this account. The `--pacing-seconds` flag makes this trivial to retune.
- **Concurrent admin agent activity.** The trueknot daemon also runs an `admin` agent on WhatsApp using the same OAuth account. If admin happens to make a Sonnet call mid-research, contention returns. Phase 5 doesn't try to coordinate across agents on one account; if this becomes a problem, the eventual fix is per-agent quota partitioning, which Anthropic does not currently expose for subscription auth.
- **Worker isolation.** The daemonized child runs as the trueknot user, no LaunchDaemon supervision. If the host reboots mid-run, the run is lost. Acceptable for research workflows; not for trading execution.

## Deferred to future phases

- Local-model analysts (Ollama) for further quota relief.
- Multi-ticker batch / scheduled cron triggers.
- Reflection-loop integration (`ta.reflect_and_remember(returns)` after a position closes).
- IBKR-trader handoff for the BUY/SELL/HOLD signal.
