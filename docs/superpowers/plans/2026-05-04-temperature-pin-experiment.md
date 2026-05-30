# Temperature Pin Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin `temperature=0` on the analyst-tier LLM (`_OAuthChatAnthropic`) so we can test whether LLM sampling variance is the cause of inter-run divergence on the same MSFT 2026-05-01 input. Run the e2e twice; if outputs converge, hypothesis confirmed.

**Architecture:** One config key (`temperature: 0.0` in `default_config.py`) flows through `_get_provider_kwargs` (in `trading_graph.py`) into the kwargs dict that `ClaudeCodeClient.get_llm()` builds, then into `_OAuthChatAnthropic(**llm_kwargs)` (which forwards to its parent `ChatAnthropic`). The CLI deep-judge tier (`ClaudeCliChatModel` → `claude -p`) is intentionally untouched — that path doesn't expose `--temperature` and bypassing would revert the Phase-5 rate-limit fix.

**Tech Stack:** Python 3.13, langchain_anthropic ChatAnthropic (Pydantic-based), pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-04-temperature-pin-experiment-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tests/test_temperature_pin.py` | Unit tests — config has temperature=0, kwargs forward it to ChatAnthropic init |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/default_config.py` | Add `"temperature": 0.0` config key |
| `tradingagents/llm_clients/claude_code_client.py` | Add `"temperature"` to `_PASSTHROUGH_KWARGS` so it forwards to `_OAuthChatAnthropic.__init__` |
| `tradingagents/graph/trading_graph.py` | Add `temperature` to the kwargs dict in `_get_provider_kwargs()` for the `claude_code` provider |

---

## Task 1: Pin temperature config key + passthrough

**Files:**
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Create: `tests/test_temperature_pin.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_temperature_pin.py`:

```python
"""Tests for the temperature=0 pin on the analyst-tier LLM (Phase-6 stochasticity experiment)."""
import pytest

pytestmark = pytest.mark.unit


def test_default_config_has_temperature_zero():
    """The pin's default value lives in DEFAULT_CONFIG so any pipeline run
    that doesn't explicitly override it gets temperature=0."""
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG.get("temperature") == 0.0


def test_oauth_chat_anthropic_accepts_temperature_kwarg():
    """_OAuthChatAnthropic must accept temperature in its constructor and
    expose it on the resulting model instance."""
    from tradingagents.llm_clients.claude_code_client import _OAuthChatAnthropic
    chat = _OAuthChatAnthropic(model="claude-sonnet-4-6", api_key="test", temperature=0.0)
    assert chat.temperature == 0.0


def test_passthrough_kwargs_includes_temperature():
    """The kwargs allowlist must include temperature so user-provided values
    flow into _OAuthChatAnthropic instead of being dropped."""
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient
    assert "temperature" in ClaudeCodeClient._PASSTHROUGH_KWARGS


def test_provider_kwargs_includes_temperature_for_claude_code():
    """trading_graph._get_provider_kwargs must surface the config's temperature
    value into the LLM client kwargs for the claude_code provider."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    config = {
        "llm_provider": "claude_code",
        "deep_think_llm": "claude-opus-4-6",
        "quick_think_llm": "claude-sonnet-4-6",
        "claude_code_token_source": "keychain",
        "temperature": 0.0,
    }
    # Use object.__new__ to avoid triggering full __init__ (which builds the graph)
    g = object.__new__(TradingAgentsGraph)
    g.config = config
    kwargs = g._get_provider_kwargs()
    assert kwargs.get("temperature") == 0.0
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_temperature_pin.py -v
```

Expected: 4 failures.
- `test_default_config_has_temperature_zero` — KeyError or `is None` (config doesn't have the key yet).
- `test_oauth_chat_anthropic_accepts_temperature_kwarg` — should pass already (ChatAnthropic accepts temperature natively); if it fails, investigate.
- `test_passthrough_kwargs_includes_temperature` — assertion error (`"temperature"` not in tuple).
- `test_provider_kwargs_includes_temperature_for_claude_code` — assertion error (kwargs dict has no `temperature` key).

- [ ] **Step 3: Add `temperature` to default config**

Open `tradingagents/default_config.py`. Find the existing `DEFAULT_CONFIG` dict. Add a new key (placement near other LLM tuning keys like `max_tokens`, `pacing_seconds` is fine — match style):

```python
    # Phase-6 stochasticity experiment: pin sampling temperature to 0 so
    # the analyst tier (TA agents, 4 analysts, bull/bear, RM, trader, risk
    # debaters, QC agent) produces deterministic output for identical
    # input. Reduces inter-run divergence (Run #5 reversal vs Run #6
    # breakdown on the same MSFT 2026-05-01 data).
    "temperature": 0.0,
```

- [ ] **Step 4: Add `temperature` to `_PASSTHROUGH_KWARGS`**

Open `tradingagents/llm_clients/claude_code_client.py`. Find:

```python
    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "max_tokens",
        "callbacks", "http_client", "http_async_client", "effort",
        "rate_limiter", "pre_invoke_sleep_seconds",
    )
```

Replace with:

```python
    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "max_tokens",
        "callbacks", "http_client", "http_async_client", "effort",
        "rate_limiter", "pre_invoke_sleep_seconds",
        "temperature",
    )
```

- [ ] **Step 5: Surface `temperature` in `_get_provider_kwargs`**

Open `tradingagents/graph/trading_graph.py`. Find the `_get_provider_kwargs` method, specifically the `claude_code` block (around line 173). After the existing `max_tokens` passthrough block:

```python
                # Per-call output-token cap. Lower values reduce per-minute
                # token spike (helps avoid 429s on Sonnet/Opus tiers).
                if "max_tokens" in self.config:
                    kwargs["max_tokens"] = self.config["max_tokens"]
```

Add immediately after:

```python
                # Phase-6 stochasticity experiment: forward the temperature
                # config to the analyst-tier ChatAnthropic so identical
                # inputs produce identical outputs. The CLI deep-judge tier
                # (ClaudeCliChatModel) is unaffected — `claude -p` doesn't
                # expose --temperature.
                if "temperature" in self.config:
                    kwargs["temperature"] = self.config["temperature"]
```

- [ ] **Step 6: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_temperature_pin.py -v
```

Expected: 4 passed.

Then full unit suite to confirm no regressions:

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 142 + 4 = 146 tests pass.

- [ ] **Step 7: Commit**

```bash
git add tradingagents/default_config.py tradingagents/llm_clients/claude_code_client.py tradingagents/graph/trading_graph.py tests/test_temperature_pin.py
git commit -m "feat(llm): pin temperature=0 for analyst tier (Phase-6 stochasticity experiment)"
```

---

## Task 2: E2E experiment — run twice on macmini, diff outputs

**Files:** none (operator step, output captured for the user-facing report)

- [ ] **Step 1: Push the branch + redeploy on macmini**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main && .venv/bin/pip install -e . --quiet'
ssh macmini-trueknot 'cd ~/tradingagents && git rev-parse --short HEAD'
```

Expected: HEAD matches the just-pushed SHA.

- [ ] **Step 2: Refresh OAuth token (avoid the 8h-TTL issue we hit twice this session)**

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
```

Expected: a brief greeting back. The credentials.json mtime should advance.

- [ ] **Step 3: Run A**

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min for completion. Verify done:

```bash
ssh macmini-trueknot 'pgrep -fl tradingresearch | head -3'  # empty
ssh macmini-trueknot 'grep -E "^\[research\]" ~/.openclaw/data/logs/tradingresearch-2026-05-01-MSFT.log | tail -2'  # done line
```

Archive the outputs as run-A:

```bash
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-tempA'
```

- [ ] **Step 4: Run B**

Refresh the token again (token expires after ~8h regardless of usage):

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
```

Then run again:

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min for completion. Archive:

```bash
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-tempB'
```

- [ ] **Step 5: Pull both runs locally and diff**

```bash
mkdir -p /tmp/tempA /tmp/tempB
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempA/raw/technicals.md' /tmp/tempA/technicals_v1.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempA/raw/technicals_v2.md' /tmp/tempA/technicals_v2.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempA/raw/pm_brief.md' /tmp/tempA/pm_brief.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempA/*.md' /tmp/tempA/

scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempB/raw/technicals.md' /tmp/tempB/technicals_v1.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempB/raw/technicals_v2.md' /tmp/tempB/technicals_v2.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempB/raw/pm_brief.md' /tmp/tempB/pm_brief.md
scp -q macmini-trueknot:'.openclaw/data/research/2026-05-01-MSFT-tempB/*.md' /tmp/tempB/
```

Then diff:

```bash
echo "=== TA v1 (the upstream stochasticity source) ==="
diff -q /tmp/tempA/technicals_v1.md /tmp/tempB/technicals_v1.md && echo "BYTE-IDENTICAL ✓" || diff /tmp/tempA/technicals_v1.md /tmp/tempB/technicals_v1.md | head -30

echo "=== PM Pre-flight Brief ==="
diff -q /tmp/tempA/pm_brief.md /tmp/tempB/pm_brief.md && echo "BYTE-IDENTICAL ✓" || echo "DIFFERS"

echo "=== Analyst reports ==="
for analyst in analyst_market analyst_fundamentals analyst_news analyst_social; do
  diff -q /tmp/tempA/${analyst}.md /tmp/tempB/${analyst}.md && echo "${analyst}: BYTE-IDENTICAL ✓" || echo "${analyst}: DIFFERS"
done

echo "=== TA v2 ==="
diff -q /tmp/tempA/technicals_v2.md /tmp/tempB/technicals_v2.md && echo "BYTE-IDENTICAL ✓" || echo "DIFFERS"

echo "=== Bull/Bear + Risk debates ==="
for f in debate_bull_bear debate_risk; do
  diff -q /tmp/tempA/${f}.md /tmp/tempB/${f}.md && echo "${f}: BYTE-IDENTICAL ✓" || echo "${f}: DIFFERS"
done

echo "=== PM Final ==="
diff -q /tmp/tempA/decision.md /tmp/tempB/decision.md && echo "decision.md: BYTE-IDENTICAL ✓" || echo "decision.md: DIFFERS (expected — CLI tier is stochastic)"
```

- [ ] **Step 6: Interpret + report**

Pass criteria:

- TA v1 byte-identical → temperature pin took effect at the analyst-tier LLM ✅ (definitional success of the experiment)
- 4 analyst reports byte-identical → upstream cascade also pinned ✅
- TA v2 byte-identical → second-pass review also pinned ✅
- Bull/bear + risk debate byte-identical → all ChatAnthropic-tier output pinned ✅
- PM Final may differ (claude CLI subprocess remains stochastic) — note as expected

Report findings to user:

- **All-converge**: hypothesis confirmed; the upstream divergence was sampling-driven; ship temperature=0 as the new default. No further action needed unless PM divergence becomes problematic.
- **Partial-converge** (e.g., TA v1 identical but TA v2 differs): something downstream re-introduces variance — investigate.
- **No-converge** (TA v1 still differs): pin didn't take effect, OR there's another variance source. Add logging to confirm `temperature=0` is on the actual SDK call. If confirmed and outputs still differ, escalate to design Option B (deterministic classification rules in TA prompt) per the prior brainstorm.

- [ ] **Step 7: Cleanup the archived dirs (optional)**

After reporting:

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-tempA
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-tempB
'
```

(Or keep them around if you want to manually re-inspect later.)

---

## Self-review notes

**Spec coverage:**
- ✅ default_config.py adds `"temperature": 0.0` (Task 1, Step 3)
- ✅ `_OAuthChatAnthropic` forwards temperature via `_PASSTHROUGH_KWARGS` (Task 1, Step 4)
- ✅ `_get_provider_kwargs` surfaces it for claude_code provider (Task 1, Step 5)
- ✅ Unit tests (4 tests in tests/test_temperature_pin.py) per the spec's "Tests / Unit" section (Task 1, Step 1)
- ✅ E2e experiment with two runs, diff, interpretation (Task 2)
- ✅ Out-of-scope (CLI tier) explicitly noted in commit message and as expected difference in PM Final
- ✅ Failure path documented (Step 6) with escalation to design Option B

**Type / signature consistency:**
- `temperature` is consistently typed as `float` (default `0.0`) — Pydantic on `ChatAnthropic` will validate it as `Optional[float]`.
- `_PASSTHROUGH_KWARGS` is a `tuple[str, ...]`; adding a string to it preserves the type.
- `kwargs` in `_get_provider_kwargs` is `Dict[str, Any]`; adding `kwargs["temperature"] = float` is consistent.

**Placeholder scan:** No TBDs / TODOs / "implement later" / "similar to Task N" patterns. Each step shows exact code to add.

**Out-of-scope confirmation:** The plan explicitly does NOT touch the CLI subprocess tier. PM Final divergence in Step 6 is expected, not a regression.

**Rollback path:** Single commit; revert returns the system to pre-pin behavior.
