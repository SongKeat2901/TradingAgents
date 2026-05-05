# TradingAgents — fork at SongKeat2901/TradingAgents

Multi-agent equity-research pipeline. Personal Claude subscription, single-user research,
deployed to `macmini-trueknot` via SSH. Production reports delivered as TrueKnot-branded
PDFs to Telegram chat `-1003753140043`.

## Quick Start

```bash
# Unit tests (216 tests, ~2s)
.venv/bin/python -m pytest -q -m unit --tb=line

# Single-test debug
.venv/bin/python -m pytest tests/test_pm_preflight.py::test_X -v

# Local e2e (don't — too slow; use macmini)
~/local/bin/tradingresearch --ticker MSFT --date 2026-05-05 \
  --output-dir ~/.openclaw/data/research/2026-05-05-MSFT \
  --telegram-notify=-1003753140043
```

## E2E run on macmini-trueknot (canonical)

```bash
# 1. Push, deploy
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet'

# 2. Refresh OAuth (8h TTL)
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'

# 3. Archive prior run + kick fresh
ssh macmini-trueknot 'mv ~/.openclaw/data/research/<DATE>-<TICKER> ...run-<SHA> 2>/dev/null'
ssh macmini-trueknot '~/local/bin/tradingresearch --ticker <T> --date <D> \
  --output-dir <PATH> --telegram-notify=-1003753140043'
# Daemon detaches; takes ~22-35 min.

# 4. Watch for completion
ssh macmini-trueknot 'until ! pgrep -f "tradingresearch --ticker <T>" >/dev/null; do sleep 30; done'
```

## Architecture (HEAD bcf4301, Phase 6.6)

```
PM Pre-flight (Opus 4.7 / CLI) — appends calendar block + SEC filing footer to pm_brief.md
  → Researcher (Python, deterministic data + classifier + peer-ratios block)
  → TA v1 → 4 analysts → TA v2 → Bull/Bear → RM → Trader → Risk team → PM → QC
  → write_research_outputs → research_pdf (TrueKnot-branded)
```

**Both LLM tiers route through `claude -p` subprocess** (`deep_via_cli=True`,
`quick_via_cli=True` defaults). Direct ChatAnthropic 429s on Sonnet/Opus on the personal
subscription. See memory: `project_phase5_rate_limit_finding.md`.

**Deterministic Python blocks** (Phase 6.2/6.3/6.4) are appended to `pm_brief.md` after
the PM Pre-flight LLM call so downstream agents see authoritative ground truth they
can't paraphrase. See memory: `project_phase6_deterministic_blocks_pattern.md`.

## Key Files

| File | Role |
|---|---|
| `cli/research.py` | Main CLI entry; default `--deep claude-opus-4-7`, `--quick claude-sonnet-4-6` |
| `cli/research_pdf.py` | TrueKnot-branded PDF generator (executive summary + appendix + vocab cleanup) |
| `cli/research_writer.py` | Writes per-run markdown + state.json with `_meta` block |
| `tradingagents/graph/trading_graph.py` | Graph wiring (`deep_via_cli`/`quick_via_cli` routing) |
| `tradingagents/agents/managers/pm_preflight.py` | Calendar (6.2) + SEC filing (6.3) block appends |
| `tradingagents/agents/researcher.py` | Peer-ratios (6.4) block append + classification |
| `tradingagents/agents/managers/qc_agent.py` | 16-item audit (item 15 filing-anchor, 16 numerical-trace) |
| `tradingagents/agents/utils/{calendar,sec_edgar,peer_ratios,classifier}.py` | Pure-Python deterministic helpers |
| `tradingagents/llm_clients/claude_cli_chat_model.py` | `claude -p` subprocess wrapper + model alias map |

## Commit conventions

- Commits go directly on `main` (no feature branches for this project).
- Footer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` —
  **no** "Generated with [Claude Code]" line.
- Use `feat()` / `fix()` / `docs()` / `test()` Conventional-Commits prefixes.
- Each commit covers one slice (don't bundle e.g. `peer_ratios.py` + `pm_preflight.py`
  wiring together).

## Gotchas

- **Don't add `bind_tools()` to new analysts.** Phase 6 dropped tool calls; analysts
  read pre-fetched data from `raw/` via `format_for_prompt`. Adding tools forces the
  analyst back onto ChatAnthropic direct-API which 429s.
- **PM Pre-flight runs BEFORE Researcher.** If a deterministic block needs `peers.json`
  (written by Researcher), wire it inside `researcher.py` AFTER the `peers.json` write.
  If it only needs `reference.json` + `classification.json`, wire it in `pm_preflight.py`.
  Calendar lives in PM Pre-flight; SEC filing in PM Pre-flight (fetcher); peer ratios
  in Researcher.
- **yfinance data anomalies.** ORCL Q1 capex/revenue can come back as 108% (capex >
  revenue). The deterministic block reports it faithfully; analyst flags as "anomalous"
  and excludes from peer comparison. Don't auto-clamp.
- **CLI subprocess doesn't support `with_structured_output`.** Judges fall back to
  free-text via `agents/utils/structured.py:invoke_structured_or_freetext`. Expected
  log line: `"... provider does not support with_structured_output (...); falling back
  to free-text generation"`.
- **OAuth token has 8h TTL.** Refresh before each e2e run with `claude -p hi`.
- **`max_tokens` defaults to 8192.** Lower it (`--max-tokens 4096`) only if the run
  hits per-call token-spike pressure; not needed for the current Sonnet/Opus-via-CLI
  setup.

## PDF cover branding

TrueKnot brand pack at `~/Documents/Python/TrueKnot Web Setup/docs/brand/`. Color
tokens live in `cli/research_pdf.py` `_CSS`. Cover SVG is inline; corporate footer
text is hardcoded ("TrueKnot Pte. Ltd. · UEN 202608241M · 1 Bukit Batok Cres,
#05-15, Singapore 658064 · trueknot.sg"). See memory:
`reference_trueknot_brand_pack.md`.

## Pointers

- Memory: `~/.claude/projects/-Users-songkeat-Documents-Python-Trading-Agent/memory/`
- Specs: `docs/superpowers/specs/`
- Plans: `docs/superpowers/plans/`
- Past runs on macmini: `~/.openclaw/data/research/<date>-<ticker>.run-<sha>/`
- Telegram delivery: `cli/research_telegram.py` (auto-discovers bot token from
  `~/.openclaw/openclaw.json` on the OpenClaw host)
