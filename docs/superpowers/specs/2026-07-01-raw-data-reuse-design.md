# Phase A — Raw-Data Reuse on Rerun (Rerun-Reduction, Phase 2 of 3)

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Make a cadence rerun cheaper by reusing the reproducible, expensive-to-fetch data from the prior attempt's `raw/*.json` instead of re-hitting yfinance. Phase 2 of the three-phase rerun-reduction effort (C done → **A** → B). Phase B (LLM-stage reuse) is out of scope here.

---

## Background

A cadence run that fails the post-hoc Phase-7 validators must re-run the whole ~20–35 min pipeline. Phase C reduced how *often* that happens; Phase A makes the reruns that remain cheaper — specifically the network-fetch cost, which is the 429-prone part.

**Corrected finding (this changes the earlier assumption):** the existing LangGraph `SqliteSaver` checkpointer does NOT help the validation-fail case. `clear_checkpoint` fires the instant the graph reaches `END` (`trading_graph.py:406-410`), which is *before* the post-graph validators run (`cli/research.py:561-567`) — so any run that gets far enough to fail validation has already had its checkpoint deleted. There is also no `update_state`/`get_state` anywhere to rewind to a specific node. The checkpointer only helps mid-run *crashes*, not validation failures. So Phase A does NOT touch the checkpointer.

**What IS reusable (verified):** the Researcher runs as a graph node (`fetch_research_pack`, wired at `setup.py:96-99`); it fetches first (network) then computes 10 deterministic blocks that are pure-Python and cheap. So the expensive part is purely the network fetches. Reusing the fetch outputs from disk and recomputing the blocks unchanged is a clean short-circuit. **LLM-stage reuse is NOT feasible with what exists and belongs to Phase B.**

## Design principles

- **Opt-in, default OFF.** Behavior with the flag absent is byte-identical to today — zero risk to first runs or to anyone not asking for reuse.
- **Reuse only the raw *fetch* outputs; always recompute the blocks.** The 10 deterministic-block JSONs (`net_debt`, `peer_ratios`, `intrinsic_value`, `accounting_ratios`, `relative_multiples`, `recent_closes`, `calendar`, `classification`, `volume_profile`, `latest_session`) are recomputed every run from the (reused) raw data. This makes reuse inherently robust to block-logic changes — only a change to the raw *fetch* schema could break it (low risk; noted below).
- **Re-fetch the date-sensitive artifacts.** `news.json` and `social.json` are 30-day rolling windows that change continuously; they are ALWAYS re-fetched fresh, never reused.
- **Identity = the output dir itself.** `raw_dir` is `preaudit/<date>-<ticker>`, so its `raw/*.json` are by construction this exact ticker+date run's files. Existence in that dir is the identity signal; a cheap ticker/trade_date sanity check is applied where the field exists. A missing/garbled/mismatched file falls back to fetching that one (partial reuse never fails the run).
- **Free-data honesty / observability:** log exactly what was reused vs fetched so a run is never silently stale.

## Trigger plumbing

`--reuse-raw` (store_true, default False) on `tradingresearch`:
- `cli/research.py`: add the arg; `_build_config` sets `config["reuse_raw"] = args.reuse_raw`.
- `tradingagents/graph/propagation.py` `create_initial_state`: add `"reuse_raw": <config value>` to the state dict (mirrors how `raw_dir` is already threaded, `propagation.py:24,59`).
- `tradingagents/agents/utils/agent_states.py` `AgentState`: add optional `reuse_raw: bool`.
- `tradingagents/agents/researcher.py` `fetch_research_pack`: read `reuse = state.get("reuse_raw", False)`.

The `cadence-run` skill's rerun step adds `--reuse-raw`; the first run omits it (and has no `raw/` anyway).

## What gets reused vs re-fetched

| Artifact | Action under `--reuse-raw` | Why |
|---|---|---|
| `financials.json` | REUSE if present | fixed for ticker+date |
| `prices.json` | REUSE if present | heaviest 5-yr OHLCV pull; fixed once session settled |
| `peers.json` | REUSE only if its keys == current `peers` list | biggest 429 multiplier (N× financials); guard peer-set change |
| `insider.json` | REUSE if present | ticker-only, rarely changes |
| `reference.json` | REUSE if present | skips the 7 indicator round-trips; holds the reference price (determinism) |
| `news.json` | ALWAYS re-fetch | 30-day rolling window, date-sensitive |
| `social.json` | ALWAYS re-fetch | same rolling-window staleness |
| all block `*.json` | ALWAYS recompute | cheap; keeps block-logic changes effective |

## The reuse-aware loader

A small helper in `researcher.py` (or a new `agents/utils/raw_reuse.py`):

```
_reuse_or_fetch(raw_dir, filename, fetch_fn, reuse, sanity=None) -> (data, reused: bool)
    if reuse and (raw_dir/filename).exists():
        try: data = json.load(...)
        except: return fetch_fn(), False        # garbled -> fetch
        if sanity is None or sanity(data):       # optional ticker/trade_date check
            return data, True                    # REUSED (fetch_fn NOT called)
    return fetch_fn(), False                      # missing / mismatch / reuse off -> fetch
```

`fetch_research_pack` routes the five reusable fetches through it; `news`/`social` bypass it. `peers.json` uses a `sanity` that checks the key set matches `peers`. At the end, log e.g. `raw-reuse: reused 5/5 fetch artifacts (skipped yfinance); re-fetched news, social`.

## pm_brief.md idempotency

No new handling needed: a full-graph rerun recreates `pm_brief.md` fresh via PM Pre-flight (the node before Researcher), then Researcher appends its blocks once — reuse only swaps *fetch* for *load*, so the append flow is unchanged. (Invariant to confirm in the plan: PM Pre-flight writes `pm_brief.md` fresh, not append — true today, else existing same-dir reruns would already double-append.)

## Testing

- **Unit (`tests/test_raw_reuse.py`):** `_reuse_or_fetch` — (a) reuse=True + file present + sanity ok → returns loaded data, `fetch_fn` NOT called (assert via a mock/counter); (b) reuse=True + file missing → calls `fetch_fn`; (c) reuse=True + present but sanity fails (ticker/trade_date or peer-set mismatch) → calls `fetch_fn`; (d) reuse=True + garbled JSON → calls `fetch_fn`, no exception; (e) reuse=False → always calls `fetch_fn`. Plus a test that the `peers.json` sanity rejects a changed peer set.
- **Prompt/plumbing test:** `--reuse-raw` sets `config["reuse_raw"]` and `create_initial_state` propagates it to `state["reuse_raw"]` (default False when absent).
- **Real success measure (not a unit test):** a rerun with `--reuse-raw` logs "reused …", makes no peer/price/financials yfinance calls, and produces an identical reference price + recent-closes pins to the original. Note in the run log.

## Out of scope (later / not this phase)

- LLM-stage reuse / targeted re-run of the failing stage — **Phase B**.
- The LangGraph checkpointer (dead end for validation fails; only helps crashes).
- Code-version stamping of raw files (a `raw/fetch_meta.json` with a git SHA to refuse reuse across a fetch-schema change) — future hardening; low risk today given reuse is opt-in, same-dir, and blocks are always recomputed.
- Auto-reuse / `--no-reuse` — rejected in favor of explicit opt-in.
