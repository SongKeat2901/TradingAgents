# Temperature Pin Experiment — Design

**Date:** 2026-05-04
**Status:** approved (user approved 2026-05-04)
**Predecessor:** [TA Judge Transparency](2026-05-04-ta-judge-transparency-design.md)

## Goal

Determine whether LLM sampling variance is the cause of inter-run divergence on the same ticker/date input. Run #5 of MSFT 2026-05-01 produced "reversal" technical reads; Run #6 produced "breakdown" reads. Same code, same data, opposite directional conclusions. The first hypothesis is that this is sampling noise (the LLM uses non-zero temperature by default). Pin `temperature=0` on the analyst tier and re-run to test.

This is an **experiment**, not a feature. The success metric is convergence: two runs of the same ticker/date should produce identical (or near-identical) outputs through the analyst + TA layers. The PM Final tier (claude CLI subprocess) is intentionally out of scope.

## Motivation

Run-#5 and run-#6 audit comparison:

| Topic | Run #5 | Run #6 |
|---|---|---|
| TA v2 classification | Post-Earnings Washout + Short-Term Reversal Setup | Confirmed Early-Stage Capitulation |
| PM verdict on v2 | Partially adopt | Adopt |
| Final rating | UNDERWEIGHT (with reversal hedge) | UNDERWEIGHT (with breakdown trim plan) |

Both reach UNDERWEIGHT but for opposite technical reasons. The variance starts upstream — TA v1 emits different setup classifications across runs — and cascades through analysts → TA v2 → debate → PM. If upstream variance disappears with `temperature=0`, downstream divergence should largely follow.

## Architecture

No architectural change. One config key plus one passthrough hookup so the existing `_OAuthChatAnthropic` (the analyst tier's LLM wrapper) forwards `temperature` to its parent `ChatAnthropic` constructor. The deep-judge tier (`ClaudeCliChatModel` → `claude -p` subprocess) is unaffected — that path doesn't expose a `--temperature` flag in the standard `claude` CLI, and bypassing it would revert the Phase-5 rate-limit fix.

```
                   ┌─ ChatAnthropic (Sonnet/Haiku, OAuth)  ← temperature=0 (NEW)
LLM tier ─────────┤
                   └─ ClaudeCliChatModel (Opus, claude -p)  ← unchanged (out of scope)
```

## Components

### `tradingagents/default_config.py`

Add a config key:

```python
"temperature": 0.0,
```

### `tradingagents/llm_clients/claude_code_client.py`

The class `_OAuthChatAnthropic(ChatAnthropic)` currently forwards a fixed list of kwargs (`_PASSTHROUGH_KWARGS`) from the constructor. Add `temperature` to that list (or to the kwargs dict directly) so it reaches the parent `ChatAnthropic.__init__`.

### `tradingagents/graph/trading_graph.py`

The `_get_provider_kwargs` helper for the `claude_code` provider reads config keys and builds the LLM constructor kwargs. Add `temperature` to that dict.

### `cli/research.py`

`_build_config(args)` is the place where CLI flags merge with `DEFAULT_CONFIG`. No change needed unless we want to expose `--temperature` as a flag. Skip — the experiment doesn't need a flag, and the default in `default_config.py` is sufficient.

## Data flow

1. CLI starts with default config (`temperature: 0.0`)
2. `_build_config` produces a dict including `temperature`
3. `TradingAgentsGraph` reads it via `_get_provider_kwargs`
4. `ClaudeCodeClient.get_llm()` instantiates `_OAuthChatAnthropic(temperature=0.0, ...)` for both deep + quick tiers (when not via CLI)
5. Every analyst, TA agent v1/v2, bull/bear, RM, trader, risk debater, and QC agent that runs on the ChatAnthropic path now uses `temperature=0`
6. PM Final still routes through `claude -p` subprocess — temperature uncontrolled there (intentional)

## Tests

### Unit

`tests/test_temperature_pin.py` — new file:

```python
def test_oauth_chat_anthropic_forwards_temperature():
    """temperature=0 in config must land on the ChatAnthropic init."""
    # Mock parent __init__; instantiate _OAuthChatAnthropic(temperature=0);
    # assert temperature=0 was passed to parent.

def test_default_config_has_temperature_zero():
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG.get("temperature") == 0.0
```

### Experiment

Manual e2e on macmini-trueknot:

1. Pin temperature, deploy at HEAD
2. Run MSFT 2026-05-01 → archive `~/.openclaw/data/research/2026-05-01-MSFT-A/`
3. Clean original dir; run MSFT 2026-05-01 again → archive `~/.openclaw/data/research/2026-05-01-MSFT-B/`
4. Diff side-by-side:
   - `raw/technicals.md` (TA v1 — first stochastic source)
   - 4 analyst reports (`market_report`, `fundamentals_report`, `news_report`, `sentiment_report`)
   - `raw/technicals_v2.md` (TA v2 — depends on TA v1 + analyst reports)
   - `debate_bull_bear.md` and `debate_risk.md`
   - `decision.md` final rating + EV

### Pass criteria

- TA v1 outputs are byte-identical or word-identical between A and B (definitional success)
- Four analyst reports converge — same setup classification, same key flags, comparable peer matrices
- TA v2 outputs converge on the same setup classification (e.g., both runs say "breakdown" or both say "reversal")
- Bull/bear/RM/Trader/Risk team transcripts may differ in word choice but converge on the same direction
- PM Final may still diverge slightly (CLI tier remains stochastic) but the technical-setup-adopted subsection should reference the same v2 classification

If the above holds → design succeeds and the prior divergence was sampling-driven.

### Failure criteria + next step

If TA v1 still emits divergent classifications between A and B → temperature pin didn't take effect, OR there's another source of variance (date/time injection, prompt drift, provider-side noise even at temp=0). In that case:

1. Add logging to confirm `temperature=0` actually reached the ChatAnthropic call
2. Diff the prompts byte-by-byte (could there be a timestamp injected?)
3. If everything is identical and outputs still differ → escalate to design Option B (deterministic classification rules in TA prompt) per the prior brainstorm

## Out of scope

- **CLI tier temperature pinning** — `claude -p` doesn't expose `--temperature`. Bypassing would revert Phase-5 rate-limit fix.
- **Rule-based classification logic** in TA prompts (option B from prior brainstorm) — only pursued if A fails
- **Multi-run consensus** with dispersion surfaced (option C) — separate workstream
- **Empirical historical anchoring** (option D) — separate workstream
- **Bull/Bear deliberate diversity** — pinning these too is fine because the bull/bear roles produce divergence-by-prompt, not divergence-by-sampling

## Trade-offs accepted

- Pinning temperature MAY produce more "formulaic" output. For an analyst tier whose job is structured numerical analysis, this is a feature, not a bug. The bull/bear debate already gets diversity from the role-prompt, not the sampling.
- The PM remains slightly stochastic via the CLI subprocess. That's acceptable — the audit found that divergence STARTS upstream and cascades. Pinning the upstream should be sufficient signal for the experiment.
- Convergence at temp=0 doesn't guarantee good output quality — only consistent output quality. If runs converge on a wrong reading, that's a separate problem (rule-based classification or empirical anchoring would address it).

## Estimated effort

- Code: 5-10 lines (config key + kwargs passthrough)
- Tests: 1 file with 2 unit tests
- E2e runs: 2 × ~14 min wall clock = ~30 min total
- Diff analysis + summary: ~10 min

Total: ~1-1.5 hours including report-back.

## Rollback path

Revert one commit. The change is a single config key + a kwargs passthrough; reverting puts the system back to the current stochastic state. No state migrations, no schema changes.
