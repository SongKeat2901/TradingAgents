# Raw-Data Reuse on Rerun — Implementation Plan (Rerun-Reduction Phase A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--reuse-raw` path so a rerun loads the reproducible fetch outputs (financials/prices/peers/insider/reference) from the prior attempt's `raw/*.json` instead of re-hitting yfinance — cutting rerun time and 429 exposure while keeping reruns deterministic.

**Architecture:** A small reuse-aware loader (`raw_reuse.py`) wraps each reusable fetch: load-from-disk when reuse is on and the file exists+passes a cheap sanity check, else fetch. `fetch_research_pack` routes the five reproducible fetches through a testable `_gather_raw` helper; `news`/`social` always re-fetch; the deterministic blocks always recompute (unchanged). The flag threads `--reuse-raw` → `config["reuse_raw"]` → `create_initial_state` → `state["reuse_raw"]`, mirroring how `output_dir` already flows.

**Tech Stack:** Python 3, pytest (`unit` marker), no new deps, no new network calls.

## Global Constraints

- **Opt-in, default OFF.** With `--reuse-raw` absent, behavior is byte-identical to today. `state.get("reuse_raw", False)` everywhere.
- **Reuse only raw *fetch* outputs; always recompute the 10 deterministic blocks.** Reuse set: `financials.json`, `prices.json`, `peers.json`, `insider.json`, `reference.json`. Never reuse a block `*.json`.
- **Always re-fetch `news.json` and `social.json`** (date-sensitive 30-day rolling window).
- **Identity = the output dir.** `raw_dir` is `preaudit/<date>-<ticker>`; existence in it is the identity signal, plus a cheap ticker/trade_date sanity check where the field exists (`financials`, `reference`) and a peer-key-set check for `peers.json`. Missing/garbled/mismatch → fetch that one (partial reuse never fails the run).
- **Observability:** the run logs what was reused vs fetched.
- **No new fetch on reuse; no renamed existing variables in the graph plumbing.**
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root `/Users/songkeat/Documents/Python/Trading Agent/TradingAgents` (baseline **745** — do not regress).

---

## File Structure

- Create: `tradingagents/agents/utils/raw_reuse.py` — `reuse_or_fetch` + `reuse_or_fetch_peers`.
- Modify: `cli/research.py` — add `--reuse-raw` arg; `_build_config` sets `config["reuse_raw"]`.
- Modify: `tradingagents/graph/trading_graph.py` (~line 369-373) — pass `reuse_raw=self.config.get("reuse_raw", False)` to `create_initial_state`.
- Modify: `tradingagents/graph/propagation.py` (`create_initial_state`) — add `reuse_raw: bool = False` param + `"reuse_raw": reuse_raw` in the returned dict.
- Modify: `tradingagents/agents/utils/agent_states.py` (`AgentState`) — add `reuse_raw: bool`.
- Modify: `tradingagents/agents/researcher.py` (`fetch_research_pack`, lines ~201-250) — extract `_build_reference`, add `_gather_raw`, route fetches, log line.
- Create tests: `tests/test_raw_reuse.py`, `tests/test_reuse_raw_plumbing.py`, `tests/test_gather_raw.py`.

---

### Task 1: reuse-aware loader (`raw_reuse.py`)

**Files:**
- Create: `tradingagents/agents/utils/raw_reuse.py`
- Test: `tests/test_raw_reuse.py`

**Interfaces:**
- Produces: `reuse_or_fetch(raw_dir, filename, fetch_fn, reuse, sanity=None) -> (data, reused: bool)` and `reuse_or_fetch_peers(raw_dir, peers, fetch_all_fn, reuse) -> (data, reused: bool)`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_raw_reuse.py
import json
import pytest
from tradingagents.agents.utils.raw_reuse import reuse_or_fetch, reuse_or_fetch_peers

pytestmark = pytest.mark.unit


def _write(dirpath, name, obj):
    (dirpath / name).write_text(json.dumps(obj), encoding="utf-8")


def test_reuse_hit_skips_fetch(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "MSFT", "trade_date": "2026-06-30", "x": 1})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True,
                                  sanity=lambda d: d.get("ticker") == "MSFT" and d.get("trade_date") == "2026-06-30")
    assert reused is True and data["x"] == 1 and calls["n"] == 0


def test_reuse_miss_fetches(tmp_path):
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True)
    assert reused is False and data == {"fetched": True} and calls["n"] == 1


def test_reuse_sanity_fail_fetches(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "AAPL", "trade_date": "2026-06-30"})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True,
                                  sanity=lambda d: d.get("ticker") == "MSFT")
    assert reused is False and calls["n"] == 1  # wrong ticker -> fetch


def test_reuse_garbled_json_fetches(tmp_path):
    (tmp_path / "financials.json").write_text("{not json", encoding="utf-8")
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True)
    assert reused is False and calls["n"] == 1  # no exception, fetched


def test_reuse_off_always_fetches(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "MSFT", "trade_date": "2026-06-30"})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=False)
    assert reused is False and calls["n"] == 1


def test_peers_reuse_hit(tmp_path):
    _write(tmp_path, "peers.json", {"AAPL": {}, "GOOGL": {}})
    calls = {"n": 0}
    def fetch_all():
        calls["n"] += 1
        return {"AAPL": {"fresh": 1}, "GOOGL": {"fresh": 1}}
    data, reused = reuse_or_fetch_peers(tmp_path, ["AAPL", "GOOGL"], fetch_all, reuse=True)
    assert reused is True and set(data.keys()) == {"AAPL", "GOOGL"} and calls["n"] == 0


def test_peers_keyset_mismatch_fetches(tmp_path):
    _write(tmp_path, "peers.json", {"AAPL": {}, "GOOGL": {}})
    calls = {"n": 0}
    def fetch_all():
        calls["n"] += 1
        return {"AAPL": {}, "META": {}}
    data, reused = reuse_or_fetch_peers(tmp_path, ["AAPL", "META"], fetch_all, reuse=True)
    assert reused is False and calls["n"] == 1  # peer set changed -> refetch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_raw_reuse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.agents.utils.raw_reuse'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/raw_reuse.py
"""Opt-in raw-artifact reuse for cheap reruns (rerun-reduction Phase A).

On a rerun with reuse enabled, load a prior attempt's raw/<file>.json from the
same run dir instead of re-fetching from yfinance. Only the raw FETCH outputs
are reused; the deterministic blocks are always recomputed by the caller.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def reuse_or_fetch(raw_dir, filename: str, fetch_fn: Callable[[], Any],
                   reuse: bool, sanity: Callable[[Any], bool] | None = None):
    """Return (data, reused). Load raw_dir/filename when reuse is on and it
    exists, parses, and passes sanity(); otherwise call fetch_fn()."""
    if reuse:
        p = Path(raw_dir) / filename
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if data is not None and (sanity is None or sanity(data)):
                return data, True
    return fetch_fn(), False


def reuse_or_fetch_peers(raw_dir, peers, fetch_all_fn: Callable[[], Any], reuse: bool):
    """peers.json is a dict keyed by peer symbol; reuse only if the key set
    exactly matches the current peer list (else the peer set changed → refetch)."""
    if reuse:
        p = Path(raw_dir) / "peers.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict) and set(data.keys()) == set(peers):
                return data, True
    return fetch_all_fn(), False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_raw_reuse.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/raw_reuse.py tests/test_raw_reuse.py
git commit -m "feat: add opt-in raw-artifact reuse loader"
```

---

### Task 2: thread `--reuse-raw` flag → state

**Files:**
- Modify: `cli/research.py` (arg parser + `_build_config`)
- Modify: `tradingagents/graph/trading_graph.py` (~line 369-373, the `create_initial_state` call)
- Modify: `tradingagents/graph/propagation.py` (`create_initial_state`)
- Modify: `tradingagents/agents/utils/agent_states.py` (`AgentState`)
- Test: `tests/test_reuse_raw_plumbing.py`

**Interfaces:**
- Produces: `state["reuse_raw"]` (bool, default False) available to `fetch_research_pack` in Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reuse_raw_plumbing.py
import pytest
from tradingagents.graph.propagation import Propagator

pytestmark = pytest.mark.unit


def test_create_initial_state_default_reuse_false():
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x")
    assert st["reuse_raw"] is False


def test_create_initial_state_reuse_true():
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x", reuse_raw=True)
    assert st["reuse_raw"] is True


def test_build_config_sets_reuse_raw():
    import argparse
    from cli.research import _build_config
    ns = argparse.Namespace()
    # minimal namespace mirroring the parser's fields used by _build_config
    for k, v in {
        "deep": "claude-opus-4-8", "quick": "claude-sonnet-4-6", "debate_rounds": 1,
        "risk_rounds": 1, "token_source": "auto", "openclaw_profile_path": None,
        "openclaw_profile_name": None, "pacing_seconds": 0, "max_tokens": 8192,
        "deep_cooldown_seconds": 0, "output_dir": "/tmp/x", "reuse_raw": True,
    }.items():
        setattr(ns, k, v)
    cfg = _build_config(ns)
    assert cfg["reuse_raw"] is True
```

Note: if `_build_config` reads other `args` fields not listed above, add them to the namespace so the test constructs cleanly — READ `_build_config` (`cli/research.py:298-312`) and include every `args.<x>` it touches.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reuse_raw_plumbing.py -q`
Expected: FAIL — `create_initial_state` has no `reuse_raw` param / `KeyError: 'reuse_raw'` / `_build_config` lacks the key.

- [ ] **Step 3: Implement the plumbing**

In `tradingagents/graph/propagation.py`, `create_initial_state` — add the param and the state key:

```python
    def create_initial_state(
        self, company_name: str, trade_date: str, past_context: str = "",
        output_dir: str = "/tmp", reuse_raw: bool = False,
    ) -> Dict[str, Any]:
        raw_dir = str(Path(output_dir) / "raw")
        return {
            # ... existing keys unchanged ...
            "raw_dir": raw_dir,
            "reuse_raw": reuse_raw,
            # ... rest unchanged ...
        }
```

In `tradingagents/graph/trading_graph.py` (~line 370-373), pass it from config:

```python
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context,
            output_dir=output_dir,
            reuse_raw=self.config.get("reuse_raw", False),
        )
```

In `tradingagents/agents/utils/agent_states.py`, add to the `AgentState` TypedDict (place near `raw_dir`):

```python
    reuse_raw: bool  # opt-in: load reproducible raw/*.json instead of re-fetching
```

In `cli/research.py`, add the argument (near the other `p.add_argument` calls) and set it in `_build_config`:

```python
    p.add_argument("--reuse-raw", action="store_true",
                   help="On a rerun, reuse reproducible raw/*.json (financials/prices/"
                        "peers/insider/reference) instead of re-fetching; news/social "
                        "are still fetched fresh.")
```
```python
    config["reuse_raw"] = args.reuse_raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reuse_raw_plumbing.py -q` and `.venv/bin/python -c "import cli.research, tradingagents.graph.trading_graph"`
Expected: PASS (3 tests); imports clean.

- [ ] **Step 5: Commit**

```bash
git add cli/research.py tradingagents/graph/trading_graph.py tradingagents/graph/propagation.py tradingagents/agents/utils/agent_states.py tests/test_reuse_raw_plumbing.py
git commit -m "feat: thread --reuse-raw flag through config into agent state"
```

---

### Task 3: route fetches through the loader in `fetch_research_pack`

**Files:**
- Modify: `tradingagents/agents/researcher.py` (`fetch_research_pack`, lines ~201-250)
- Test: `tests/test_gather_raw.py`

**Interfaces:**
- Consumes: `reuse_or_fetch` / `reuse_or_fetch_peers` (Task 1); `state["reuse_raw"]` (Task 2).
- Produces: `_build_reference(ticker, date, prices, indicators) -> dict` and `_gather_raw(ticker, date, peers, raw, reuse) -> (bundle: dict, reused: dict[str,bool])`, both module-level in `researcher.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gather_raw.py
import json
import pytest
import tradingagents.agents.researcher as R

pytestmark = pytest.mark.unit

_OHLCV = ("Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
          "2026-06-29,360.0,370.0,359.0,368.57,1,0.0,0.0\n"
          "2026-06-30,368.0,375.0,367.0,373.02,1,0.0,0.0\n")


def _seed(raw):
    (raw / "financials.json").write_text(json.dumps({"ticker": "MSFT", "trade_date": "2026-06-30"}), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps({"ohlcv": _OHLCV}), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps({"transactions": []}), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps({"AAPL": {}, "GOOGL": {}}), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps({"ticker": "MSFT", "trade_date": "2026-06-30", "reference_price": 373.02}), encoding="utf-8")


def test_reuse_skips_reproducible_fetches_but_refetches_news_social(tmp_path, monkeypatch):
    raw = tmp_path
    _seed(raw)
    counts = {k: 0 for k in ("fin", "news", "insider", "social", "prices", "ind")}
    monkeypatch.setattr(R, "_fetch_financials", lambda t, d: counts.__setitem__("fin", counts["fin"] + 1) or {"ticker": t, "trade_date": d})
    monkeypatch.setattr(R, "_fetch_news", lambda t, d: counts.__setitem__("news", counts["news"] + 1) or {"n": 1})
    monkeypatch.setattr(R, "_fetch_insider", lambda t, d: counts.__setitem__("insider", counts["insider"] + 1) or {"transactions": []})
    monkeypatch.setattr(R, "_fetch_social", lambda t, d: counts.__setitem__("social", counts["social"] + 1) or {"s": 1})
    monkeypatch.setattr(R, "_fetch_prices", lambda t, d: counts.__setitem__("prices", counts["prices"] + 1) or {"ohlcv": _OHLCV})
    monkeypatch.setattr(R, "_fetch_indicators", lambda t, d: counts.__setitem__("ind", counts["ind"] + 1) or {})

    bundle, reused = R._gather_raw("MSFT", "2026-06-30", ["AAPL", "GOOGL"], raw, reuse=True)

    # reproducible fetches skipped
    assert counts["fin"] == 0 and counts["prices"] == 0 and counts["insider"] == 0 and counts["ind"] == 0
    # peers reused (no per-peer financials fetch beyond the 0 above)
    assert reused["financials"] and reused["prices"] and reused["insider"] and reused["reference"] and reused["peers"]
    # news/social always fresh
    assert counts["news"] == 1 and counts["social"] == 1
    assert bundle["reference"]["reference_price"] == 373.02


def test_reuse_off_fetches_everything(tmp_path, monkeypatch):
    raw = tmp_path
    _seed(raw)  # files present, but reuse off => ignored
    counts = {k: 0 for k in ("fin", "prices", "ind", "insider", "news", "social")}
    monkeypatch.setattr(R, "_fetch_financials", lambda t, d: counts.__setitem__("fin", counts["fin"] + 1) or {"ticker": t, "trade_date": d})
    monkeypatch.setattr(R, "_fetch_news", lambda t, d: counts.__setitem__("news", counts["news"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_insider", lambda t, d: counts.__setitem__("insider", counts["insider"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_social", lambda t, d: counts.__setitem__("social", counts["social"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_prices", lambda t, d: counts.__setitem__("prices", counts["prices"] + 1) or {"ohlcv": _OHLCV})
    monkeypatch.setattr(R, "_fetch_indicators", lambda t, d: counts.__setitem__("ind", counts["ind"] + 1) or {})

    bundle, reused = R._gather_raw("MSFT", "2026-06-30", ["AAPL", "GOOGL"], raw, reuse=False)

    assert counts["fin"] >= 1 and counts["prices"] == 1 and counts["ind"] == 1
    assert not any(reused.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gather_raw.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_gather_raw'`.

- [ ] **Step 3: Refactor `fetch_research_pack`**

READ `researcher.py:201-250` first. Extract the reference-build (lines 214-241) into a module-level helper, add `_gather_raw`, and replace the inline fetch/reference/section with calls to them. Concretely:

Add near the other module helpers:

```python
def _build_reference(ticker, date, prices, indicators):
    """Build the reference dict (was inline in fetch_research_pack)."""
    rows = _parse_ohlcv_rows(prices.get("ohlcv", ""))
    close_date, close_on_date = _close_with_date_on_or_before(rows, date)
    ytd_high, ytd_low = _ytd_high_low(rows, date)
    if close_date and close_date == date:
        _ref_source = f"yfinance close of {close_date}"
    elif close_date:
        _ref_source = (f"yfinance close of {close_date} (latest available on/before "
                       f"trade_date {date}; {date}'s session has not closed/indexed)")
    else:
        _ref_source = f"yfinance close on or before {date}"
    return {
        "ticker": ticker, "trade_date": date,
        "reference_price": close_on_date, "reference_close_date": close_date,
        "reference_price_source": _ref_source,
        "spot_50dma": _latest_indicator_value(indicators.get("close_50_sma", "")),
        "spot_200dma": _latest_indicator_value(indicators.get("close_200_sma", "")),
        "ytd_high": ytd_high, "ytd_low": ytd_low,
        "atr_14": _latest_indicator_value(indicators.get("atr", "")),
    }


def _gather_raw(ticker, date, peers, raw, reuse):
    """Fetch (or reuse) the raw inputs. Reuses financials/prices/insider/peers/
    reference from raw/*.json when reuse is on; always fetches news/social fresh;
    returns (bundle, reused_map)."""
    from tradingagents.agents.utils.raw_reuse import reuse_or_fetch, reuse_or_fetch_peers
    _id = lambda d: d.get("ticker") == ticker and d.get("trade_date") == date
    financials, r_fin = reuse_or_fetch(raw, "financials.json", lambda: _fetch_financials(ticker, date), reuse, sanity=_id)
    prices, r_px = reuse_or_fetch(raw, "prices.json", lambda: _fetch_prices(ticker, date), reuse)
    insider, r_ins = reuse_or_fetch(raw, "insider.json", lambda: _fetch_insider(ticker, date), reuse)
    peers_data, r_peers = reuse_or_fetch_peers(raw, peers, lambda: {p: _fetch_financials(p, date) for p in peers}, reuse)
    reference, r_ref = reuse_or_fetch(raw, "reference.json",
                                      lambda: _build_reference(ticker, date, prices, _fetch_indicators(ticker, date)),
                                      reuse, sanity=_id)
    news = _fetch_news(ticker, date)     # always fresh (date-sensitive)
    social = _fetch_social(ticker, date)  # always fresh (date-sensitive)
    bundle = {"financials": financials, "prices": prices, "insider": insider,
              "peers_data": peers_data, "reference": reference, "news": news, "social": social}
    reused = {"financials": r_fin, "prices": r_px, "insider": r_ins, "peers": r_peers, "reference": r_ref}
    return bundle, reused
```

Then in `fetch_research_pack`, replace lines ~201-241 (the six `_fetch_*` calls, the `indicators` fetch, and the inline reference build) with:

```python
    reuse = state.get("reuse_raw", False)
    bundle, reused = _gather_raw(ticker, date, peers, raw, reuse)
    financials = bundle["financials"]; prices = bundle["prices"]; insider = bundle["insider"]
    peers_data = bundle["peers_data"]; reference = bundle["reference"]
    news = bundle["news"]; social = bundle["social"]
    if reuse:
        hit = ", ".join(k for k, v in reused.items() if v) or "none"
        print(f"[raw-reuse] reused {sum(reused.values())}/{len(reused)} artifacts "
              f"({hit}); re-fetched fresh: news, social")
    else:
        print("[raw-reuse] off (fetched all fresh)")
```

Leave the `raw/*.json` writes (lines ~244-250) unchanged — re-writing reused data is harmless and keeps the run dir consistent. Confirm no other code in `fetch_research_pack` still references a now-removed local `indicators` variable (it was only used to build `reference`). If `researcher.py` uses a logger rather than `print`, use that logger instead.

- [ ] **Step 4: Run tests + import smoke**

Run: `.venv/bin/python -m pytest tests/test_gather_raw.py -q` and `.venv/bin/python -c "import tradingagents.agents.researcher"`
Then full: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all pass. Note in the report that the full `fetch_research_pack` node isn't unit-tested (network + pm_brief deps), so verification is: `_gather_raw` reuse test + import smoke + full suite; live verification is a `--reuse-raw` rerun.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_gather_raw.py
git commit -m "feat: route researcher fetches through reuse loader (--reuse-raw)"
```

---

### Task 4: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 745 + new raw-reuse/plumbing/gather tests). Investigate any regression.

- [ ] **Step 2: (Optional) live rerun check**

On the mini, run a ticker once (writes `raw/`), then re-run the same ticker+date with `--reuse-raw` and confirm the log shows `[raw-reuse] reused 5/5 ...`, no peer/price yfinance calls occur, and the reference price + recent-closes pins are identical to the first run. Not required for merge.

---

## Out of scope (later / not this plan)

- LLM-stage reuse / targeted re-run of the failing stage — Phase B.
- The LangGraph checkpointer (dead end for validation fails).
- Code-version stamping of raw files (`raw/fetch_meta.json` with a git SHA) — future hardening.

## Self-Review

- **Spec coverage:** opt-in flag default-off (Task 2), reuse set financials/prices/peers/insider/reference (Task 3 `_gather_raw`), always-refetch news/social (Task 3), always-recompute blocks (unchanged — blocks stay after `_gather_raw`), identity sanity incl. peer-key-set (Task 1 `reuse_or_fetch_peers` + `_id` sanity), partial-reuse fallback on missing/garbled/mismatch (Task 1 tests b/c/d), observability log line (Task 3), loader unit-tested (Task 1), plumbing tested (Task 2), reuse-routing tested without network (Task 3 `_gather_raw` monkeypatch test). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step has real code; commands have expected output. The one "READ and include other args" note in Task 2 Step 1 is a concrete instruction (mirror `_build_config`'s field reads), not a placeholder.
- **Type consistency:** `reuse_or_fetch(...) -> (data, reused)` and `reuse_or_fetch_peers(...) -> (data, reused)` unpacked consistently in `_gather_raw`; `_gather_raw(...) -> (bundle, reused_map)` consumed the same way in `fetch_research_pack`; `state["reuse_raw"]` produced in Task 2, read in Task 3; `_build_reference` signature matches its call inside `_gather_raw`.
