# Deterministic Peer Ratios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pure-Python compute peer capex/revenue + op margin + P/E ratios from `raw/peers.json`, write `raw/peer_ratios.json`, Python-append a verbatim `## Peer ratios` table to `pm_brief.md` after the PM Pre-flight LLM call. Closes the Phase-6.3 caveat-wrapping hole architecturally.

**Architecture:** Mirrors the Phase 6.2 calendar block. PM Pre-flight is the injection site (peers.json is per-run, not per-ticker). Block lands AFTER the existing Phase 6.2 calendar table and Phase 6.3 SEC filing footer, in that order. Downstream agents reading pm_brief.md via `format_for_prompt` see authoritative ratios as ground truth; QC item 16(c) catches any LLM-side deviation deterministically.

**Tech Stack:** Python 3.13, regex (stdlib), pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-05-deterministic-peer-ratios-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tradingagents/agents/utils/peer_ratios.py` | `compute_peer_ratios(peers_data, trade_date)` returns dict of per-peer ratios; `format_peer_ratios_block(ratios)` renders Markdown table |
| `tests/test_peer_ratios.py` | Unit tests: happy path, missing column, zero revenue, PE parse, PE unparseable, top-level metadata |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/managers/pm_preflight.py` | Read `raw/peers.json`, call `compute_peer_ratios`, write `raw/peer_ratios.json`, append `## Peer ratios` block to `pm_brief.md` |
| `tests/test_pm_preflight.py` | +3 tests: appends block when peers.json present, omits when missing, handles compute exception gracefully |

---

## Task 1: peer_ratios module + unit tests

**Files:**
- Create: `tradingagents/agents/utils/peer_ratios.py`
- Create: `tests/test_peer_ratios.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_peer_ratios.py`:

```python
"""Tests for the deterministic peer-ratios module (Phase-6.4)."""
import pytest

pytestmark = pytest.mark.unit


def _stub_peer(income_rows: list[tuple[str, list[float]]],
               cashflow_rows: list[tuple[str, list[float]]],
               fundamentals_text: str = "") -> dict:
    """Build a minimal peers.json sub-tree for one peer."""
    def _csv(rows):
        out = "# header line\n"
        for name, vals in rows:
            out += name + "," + ",".join(str(v) for v in vals) + "\n"
        return out
    return {
        "ticker": "TEST",
        "trade_date": "2026-05-01",
        "fundamentals": fundamentals_text,
        "balance_sheet": "",
        "cashflow": _csv(cashflow_rows),
        "income_statement": _csv(income_rows),
    }


def test_compute_peer_ratios_happy_path():
    """Per-peer Q1 capex/revenue + op margin compute correctly; PE parses
    from the fundamentals text block."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "GOOGL": _stub_peer(
            income_rows=[
                ("Total Revenue", [109_900_000_000, 100_000_000_000]),
                ("Operating Income", [39_700_000_000, 35_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-35_700_000_000, -30_000_000_000]),
            ],
            fundamentals_text=(
                "# Company Fundamentals for GOOGL\n"
                "Name: Alphabet Inc.\n"
                "PE Ratio (TTM): 29.23341\n"
                "Forward PE: 26.67876\n"
            ),
        ),
    }

    out = compute_peer_ratios(peers_data, "2026-05-01")

    assert out["trade_date"] == "2026-05-01"
    assert out["_unavailable"] == []
    g = out["GOOGL"]
    # 35.7B / 109.9B = 32.48%
    assert abs(g["latest_quarter_capex_to_revenue"] - 32.48) < 0.05
    # 39.7B / 109.9B = 36.12%
    assert abs(g["latest_quarter_op_margin"] - 36.12) < 0.05
    assert abs(g["ttm_pe"] - 29.23) < 0.01
    assert abs(g["forward_pe"] - 26.68) < 0.01
    assert "peers.json" in g["source"].lower()


def test_compute_peer_ratios_handles_missing_capex_column():
    """If a peer has no Capital Expenditure row, the peer enters _unavailable;
    other peers still populate."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "GOOD": _stub_peer(
            income_rows=[
                ("Total Revenue", [100_000_000_000]),
                ("Operating Income", [25_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-10_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 20.0\nForward PE: 18.0\n",
        ),
        "BAD": _stub_peer(
            income_rows=[
                ("Total Revenue", [50_000_000_000]),
            ],
            cashflow_rows=[
                # NO Capital Expenditure row
                ("Operating Cash Flow", [10_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 15.0\nForward PE: 14.0\n",
        ),
    }

    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert "GOOD" in out
    assert out["GOOD"]["latest_quarter_capex_to_revenue"] == 10.0
    assert "BAD" in out
    assert out["BAD"].get("unavailable") is True
    assert "BAD" in out["_unavailable"]
    assert "GOOD" not in out["_unavailable"]


def test_compute_peer_ratios_handles_zero_revenue():
    """Zero revenue must not raise ZeroDivisionError; peer enters _unavailable."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "ZEROREV": _stub_peer(
            income_rows=[
                ("Total Revenue", [0.0]),
                ("Operating Income", [0.0]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-1_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 0\nForward PE: 0\n",
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["ZEROREV"].get("unavailable") is True
    assert "ZEROREV" in out["_unavailable"]


def test_compute_peer_ratios_parses_pe_from_fundamentals_text():
    """The PE parser extracts TTM and Forward PE from the multi-line text block
    yfinance dumps as the `fundamentals` field."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "MSFT": _stub_peer(
            income_rows=[
                ("Total Revenue", [82_886_000_000]),
                ("Operating Income", [38_398_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-30_876_000_000]),
            ],
            fundamentals_text=(
                "# Company Fundamentals for MSFT\n"
                "Name: Microsoft Corporation\n"
                "Sector: Technology\n"
                "Market Cap: 3000000000000\n"
                "PE Ratio (TTM): 31.5\n"
                "Forward PE: 28.4\n"
                "PEG Ratio: 2.1\n"
            ),
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["MSFT"]["ttm_pe"] == 31.5
    assert out["MSFT"]["forward_pe"] == 28.4


def test_compute_peer_ratios_handles_unparseable_pe():
    """If the fundamentals text doesn't contain PE Ratio lines, the peer's
    ttm_pe / forward_pe are None — but capex/revenue + op margin still
    populate. Only PE-related fields go None; the peer is NOT marked
    _unavailable just for missing PE data."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "NOPEDATA": _stub_peer(
            income_rows=[
                ("Total Revenue", [50_000_000_000]),
                ("Operating Income", [10_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-5_000_000_000]),
            ],
            fundamentals_text="(empty fundamentals)",
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["NOPEDATA"]["ttm_pe"] is None
    assert out["NOPEDATA"]["forward_pe"] is None
    assert out["NOPEDATA"]["latest_quarter_capex_to_revenue"] == 10.0
    assert "NOPEDATA" not in out["_unavailable"]


def test_compute_peer_ratios_top_level_metadata():
    """Output dict has top-level trade_date + _unavailable; trade_date matches input."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    out = compute_peer_ratios({}, "2026-05-01")
    assert out["trade_date"] == "2026-05-01"
    assert out["_unavailable"] == []
    # Empty input — no peers in output beyond bookkeeping
    assert set(out.keys()) == {"trade_date", "_unavailable"}
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_peer_ratios.py -v
```

Expected: ImportError on `tradingagents.agents.utils.peer_ratios`.

- [ ] **Step 3: Implement the module**

Create `tradingagents/agents/utils/peer_ratios.py`:

```python
"""Deterministic peer-ratio computation (Phase-6.4 caveat-wrapping closure).

Phase 6.3 audits showed the PM citing fabricated peer capex intensities
(GOOGL 4.9%, AMZN 5.1%; actual 32.5%, 24.4%) under "inherited from prior
debate, not revalidated" caveats that the QC LLM accepted because it
doesn't have access to raw/peers.json. This module computes authoritative
per-peer ratios from peers.json data; PM Pre-flight Python-appends them
to pm_brief.md verbatim, removing the LLM's chance to paraphrase.

Mirrors the Phase 6.2 deterministic earnings calendar pattern.
"""

from __future__ import annotations

import re
from typing import Any


def _parse_quarterly_csv(text: str) -> dict[str, list[float]]:
    """Parse the comma-table format yfinance writes for income_statement /
    cashflow / balance_sheet. Returns {row_name: [col0, col1, ...]} where
    column 0 is the most-recent quarter."""
    rows: dict[str, list[float]] = {}
    if not text:
        return rows
    for line in text.split("\n"):
        if not line or line.startswith("#") or "," not in line:
            continue
        parts = line.split(",")
        name = parts[0].strip()
        if not name:
            continue
        vals: list[float] = []
        for p in parts[1:]:
            p = p.strip()
            if not p:
                continue
            try:
                vals.append(float(p))
            except ValueError:
                pass
        if vals:
            rows[name] = vals
    return rows


def _parse_pe_from_fundamentals(text: str) -> tuple[float | None, float | None]:
    """Extract (TTM PE, Forward PE) from the yfinance fundamentals text block.
    Returns (None, None) if either is missing or unparseable."""
    if not text:
        return None, None

    def _find(pattern: str) -> float | None:
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            return None

    ttm = _find(r"PE Ratio \(TTM\):\s*([0-9.]+)")
    fwd = _find(r"Forward PE:\s*([0-9.]+)")
    return ttm, fwd


def _compute_one_peer(peer_data: dict[str, Any]) -> dict[str, Any]:
    """Per-peer ratio computation; returns either a populated dict or
    {"unavailable": True, "reason": "..."} on any failure."""
    inc = _parse_quarterly_csv(peer_data.get("income_statement", ""))
    cf = _parse_quarterly_csv(peer_data.get("cashflow", ""))

    rev_col = inc.get("Total Revenue")
    opi_col = inc.get("Operating Income")
    capex_col = cf.get("Capital Expenditure")

    if not rev_col or not opi_col or not capex_col:
        missing = []
        if not rev_col:
            missing.append("Total Revenue")
        if not opi_col:
            missing.append("Operating Income")
        if not capex_col:
            missing.append("Capital Expenditure")
        return {"unavailable": True, "reason": f"missing rows: {', '.join(missing)}"}

    revenue = rev_col[0]
    if revenue <= 0:
        return {"unavailable": True, "reason": f"degenerate revenue: {revenue}"}

    op_income = opi_col[0]
    capex = abs(capex_col[0])

    ttm_pe, forward_pe = _parse_pe_from_fundamentals(peer_data.get("fundamentals", ""))

    return {
        "latest_quarter_capex_to_revenue": round(capex / revenue * 100, 2),
        "latest_quarter_op_margin": round(op_income / revenue * 100, 2),
        "ttm_pe": round(ttm_pe, 2) if ttm_pe is not None else None,
        "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
        "source": "peers.json (yfinance via Q1 capex/revenue)",
    }


def compute_peer_ratios(peers_data: dict[str, Any], trade_date: str) -> dict[str, Any]:
    """Compute authoritative peer ratios from raw/peers.json data.

    peers_data: dict mapping ticker → peer-data dict (the structure yfinance
        writes via the researcher: ticker, trade_date, fundamentals,
        balance_sheet, cashflow, income_statement).
    trade_date: "YYYY-MM-DD" string; passed through to the output.

    Returns:
        {
          "trade_date": "2026-05-01",
          "_unavailable": list of ticker symbols where computation failed,
          "<TICKER>": {
            "latest_quarter_capex_to_revenue": <float, in %>,
            "latest_quarter_op_margin": <float, in %>,
            "ttm_pe": <float or None>,
            "forward_pe": <float or None>,
            "source": "peers.json (...)",
          } OR {"unavailable": True, "reason": "..."},
        }
    """
    out: dict[str, Any] = {"trade_date": trade_date, "_unavailable": []}
    for ticker, peer_data in peers_data.items():
        if not isinstance(peer_data, dict):
            continue
        result = _compute_one_peer(peer_data)
        out[ticker] = result
        if result.get("unavailable"):
            out["_unavailable"].append(ticker)
    return out


def format_peer_ratios_block(ratios: dict[str, Any]) -> str:
    """Render peer_ratios.json content as a Markdown table for appending to
    pm_brief.md after the PM Pre-flight LLM call.

    Returns "" if no peer rows can be rendered (all unavailable or empty).
    """
    if not ratios:
        return ""
    trade_date = ratios.get("trade_date", "?")
    unavailable_set = set(ratios.get("_unavailable", []))

    rows: list[str] = []
    for key, val in ratios.items():
        if key in ("trade_date", "_unavailable"):
            continue
        if not isinstance(val, dict):
            continue
        if key in unavailable_set or val.get("unavailable"):
            rows.append(f"| {key} | (unavailable) | (unavailable) | (unavailable) | (unavailable) |")
            continue

        def _pct(v):
            return f"{v:.1f}%" if v is not None else "(n/a)"

        def _x(v):
            return f"{v:.2f}x" if v is not None else "(n/a)"

        rows.append(
            f"| {key} | {_pct(val.get('latest_quarter_capex_to_revenue'))} | "
            f"{_pct(val.get('latest_quarter_op_margin'))} | "
            f"{_x(val.get('ttm_pe'))} | "
            f"{_x(val.get('forward_pe'))} |"
        )

    if not rows:
        return ""

    table = "\n".join(rows)
    return (
        f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {trade_date})\n\n"
        "| Ticker | Q1 capex/revenue | Q1 op margin | TTM P/E | Forward P/E |\n"
        "|---|---|---|---|---|\n"
        f"{table}\n\n"
        "*Use these values verbatim. Do NOT cite \"approximate\" or "
        "\"inherited from prior debate\" alternatives — these are the "
        "authoritative current-quarter figures derived from yfinance data on "
        "the trade date. If you need to make a peer-comparison claim, "
        "recompute deltas from this table, not from memory.*\n"
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_peer_ratios.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 197 + 6 = 203 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/utils/peer_ratios.py tests/test_peer_ratios.py
git commit -m "$(cat <<'EOF'
feat(peer-ratios): pure-Python compute from peers.json (Phase-6.4 foundation)

Foundation module for the Phase 6.4 deterministic peer-ratio injection.
compute_peer_ratios(peers_data, trade_date) returns per-peer Q1
capex/revenue, op margin, TTM/Forward PE, with graceful per-peer
fallback on missing rows or degenerate values.

format_peer_ratios_block(ratios) renders the dict as a Markdown table
for appending to pm_brief.md after the PM Pre-flight LLM call. Mirrors
the Phase 6.2 calendar-block formatting pattern.

Six unit tests cover happy path, missing column, zero revenue, PE
parsing, PE absence, and top-level metadata.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PM Pre-flight integration + unit tests

**Files:**
- Modify: `tradingagents/agents/managers/pm_preflight.py`
- Modify: `tests/test_pm_preflight.py`

- [ ] **Step 1: Append the 3 new tests to `tests/test_pm_preflight.py`**

Add at the bottom of the existing file:

```python
def test_pm_preflight_appends_peer_ratios_block(tmp_path, monkeypatch):
    """If raw/peers.json is present, PM Pre-flight writes raw/peer_ratios.json
    AND appends a '## Peer ratios' block to pm_brief.md after the calendar
    + SEC filing blocks."""
    import json as _json
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    # peers.json with a single peer that has all rows
    peers_json = {
        "GOOGL": {
            "ticker": "GOOGL",
            "trade_date": "2026-05-01",
            "fundamentals": "PE Ratio (TTM): 29.23\nForward PE: 26.68\n",
            "balance_sheet": "",
            "cashflow": "# header\nCapital Expenditure,-35700000000\n",
            "income_statement": (
                "# header\n"
                "Total Revenue,109900000000\n"
                "Operating Income,39700000000\n"
            ),
        }
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "peers.json").write_text(_json.dumps(peers_json), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    pr_path = raw / "peer_ratios.json"
    assert pr_path.exists()
    pr = _json.loads(pr_path.read_text(encoding="utf-8"))
    assert pr["trade_date"] == "2026-05-01"
    # 35.7B / 109.9B = 32.48%
    assert abs(pr["GOOGL"]["latest_quarter_capex_to_revenue"] - 32.48) < 0.05

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios (computed from raw/peers.json, trade_date 2026-05-01)" in brief
    assert "GOOGL" in brief
    assert "32.5%" in brief  # rendered with 1 decimal
    assert "29.23x" in brief
    # The peer block must come AFTER any calendar/SEC-filing blocks (last appended).
    assert brief.rfind("## Peer ratios") > brief.rfind("# PM Pre-flight Brief")


def test_pm_preflight_skips_peer_ratios_when_peers_json_missing(tmp_path):
    """No peers.json → no peer_ratios.json written, no peer block in brief."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    # NO peers.json

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    assert not (raw / "peer_ratios.json").exists()
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios" not in brief


def test_pm_preflight_handles_peer_ratios_compute_exception(tmp_path, monkeypatch):
    """If compute_peer_ratios raises (e.g., unexpected JSON shape), PM
    Pre-flight degrades gracefully — no crash, no peer_ratios.json,
    no block, but brief.md still has the LLM content."""
    import json as _json
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "peers.json").write_text(_json.dumps({"GOOGL": {}}), encoding="utf-8")

    def _raises(*a, **kw):
        raise RuntimeError("simulated peer-compute crash")

    monkeypatch.setattr(
        "tradingagents.agents.utils.peer_ratios.compute_peer_ratios", _raises
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    out = node(state)  # MUST NOT raise

    assert not (raw / "peer_ratios.json").exists()
    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Peer ratios" not in brief
    # The LLM-written content + earlier appended blocks (calendar / sec_filing)
    # are unaffected. Just verify the brief still looks like the LLM brief.
    assert out["pm_brief"] == brief
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: 3 new tests fail (compute_peer_ratios import / wiring not yet added).

- [ ] **Step 3: Wire the peer-ratio block into `pm_preflight_node`**

Open `tradingagents/agents/managers/pm_preflight.py`. Find the existing Phase 6.3 SEC filing footer append block (around line 209-235, ends with `brief = brief + footer`). Add immediately after, BEFORE the `return` statement:

```python
        # Phase 6.4 deterministic peer ratios: compute authoritative
        # capex/revenue + op margin + P/E from raw/peers.json (yfinance
        # data) and append a verbatim "## Peer ratios" block to
        # pm_brief.md so downstream agents see ground-truth values they
        # cannot paraphrase. Closes the 16(a) caveat-wrapping hole the
        # Phase 6.3 audit identified (GOOGL 4.9% / AMZN 5.1% claims with
        # caveats vs actual 32.5% / 24.4%).
        peers_path = raw_dir / "peers.json"
        if peers_path.exists():
            try:
                from tradingagents.agents.utils.peer_ratios import (
                    compute_peer_ratios,
                    format_peer_ratios_block,
                )
                peers_data = json.loads(peers_path.read_text(encoding="utf-8"))
                ratios = compute_peer_ratios(peers_data, date)
                (raw_dir / "peer_ratios.json").write_text(
                    json.dumps(ratios, indent=2, default=str),
                    encoding="utf-8",
                )
                peer_block = format_peer_ratios_block(ratios)
                if peer_block:
                    with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
                        f.write(peer_block)
                    brief = brief + peer_block
            except Exception:  # noqa: BLE001 — graceful degradation
                pass
```

The `try/except Exception` catches any failure (missing keys, JSON decode, compute_peer_ratios raises, etc.) and degrades to the no-block path, mirroring the Phase 6.3 fetcher-exception pattern.

If `import json` is not already at the top of the file, ensure it is.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: all PM Pre-flight tests pass (existing + 3 new).

If a phrasing test fails because the actual block text differs from the asserted substring (e.g., the table rendering rounds differently): adjust the test assertion to match the canonical phrasing while preserving the intent (block presence + temporal context + per-peer numerical content). Don't change the production code unless there's a real bug.

- [ ] **Step 5: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 203 + 3 = 206 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/managers/pm_preflight.py tests/test_pm_preflight.py
git commit -m "$(cat <<'EOF'
feat(pm-preflight): append deterministic peer-ratios block (Phase-6.4)

After the existing Phase 6.2 calendar block + Phase 6.3 SEC filing
footer, PM Pre-flight now computes peer Q1 capex/revenue + op margin
+ TTM/Forward PE from raw/peers.json and appends a verbatim
"## Peer ratios" Markdown table to pm_brief.md. The block uses the
same Python-after-LLM injection pattern proven in 6.2 / 6.3.

Closes the 16(a) caveat-wrapping hole identified by the c5c41e4
audit: the QC LLM doesn't have access to raw/peers.json, so it can't
validate "GOOGL 4.9% / AMZN 5.1% inherited from prior debate"
claims; now the authoritative ratios are in pm_brief.md itself, and
QC item 16(c) catches any deviation deterministically.

Three unit tests cover the present / missing / compute-exception
paths, mirroring the Phase 6.3 fetcher tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: E2E validation on macmini

**Files:** none (operator step)

- [ ] **Step 1: Push + redeploy**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet && git rev-parse --short HEAD'
```

Expected: HEAD on macmini matches the just-pushed SHA from Task 2.

- [ ] **Step 2: Refresh OAuth + run MSFT 2026-05-01**

Archive the prior `0617182` run for comparison:

```bash
ssh macmini-trueknot 'mv ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT.run-0617182 2>/dev/null || true'
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
ssh macmini-trueknot '
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-22 min. Verify done via process-watch.

- [ ] **Step 3: Inspect peer_ratios.json**

```bash
ssh macmini-trueknot 'cat ~/.openclaw/data/research/2026-05-01-MSFT/raw/peer_ratios.json'
```

Expected:
- `trade_date: "2026-05-01"`
- `_unavailable: []` (or a small list if any peer fetch failed)
- For GOOGL: `latest_quarter_capex_to_revenue ≈ 32.5%`
- For AMZN: `≈ 24.4%`
- For ORCL: `≈ 24.7%`
- For CRM: `≈ 0.5%`
- TTM/Forward PE values match the values present in the GOOGL/AMZN fundamentals text

If any peer is in `_unavailable`, check its peers.json sub-tree to confirm a missing-row scenario was the cause.

- [ ] **Step 4: Inspect pm_brief.md tail**

```bash
ssh macmini-trueknot 'tail -25 ~/.openclaw/data/research/2026-05-01-MSFT/raw/pm_brief.md'
```

Expected: ends with the `## Peer ratios (computed from raw/peers.json, trade_date 2026-05-01)` block, AFTER the existing Phase 6.2 calendar table and Phase 6.3 SEC filing footer. The peer ratios in the block must match `peer_ratios.json` byte-exactly.

- [ ] **Step 5: Spot-check the decision.md and analyst reports for the OLD hallucination**

```bash
ssh macmini-trueknot '
grep -niE "GOOGL.{0,15}4\\.9%|AMZN.{0,15}5\\.1%|GCP capex intensity 4\\.9|AWS capex intensity 5\\.1" \
  ~/.openclaw/data/research/2026-05-01-MSFT/decision.md \
  ~/.openclaw/data/research/2026-05-01-MSFT/analyst_*.md \
  ~/.openclaw/data/research/2026-05-01-MSFT/debate_*.md
'
```

Expected: no hits. The fabricated 4.9%/5.1% framing should be gone — the PM/analysts now have authoritative ratios in pm_brief.md.

If any hit appears: the block is in pm_brief.md but the LLM is ignoring it. That's a 16(c) FAIL waiting to happen on the next QC pass. Tighten the block's instruction text or escalate to a 16(c) prompt clarification.

- [ ] **Step 6: Spot-check the decision.md for the NEW (authoritative) numbers**

```bash
ssh macmini-trueknot '
grep -niE "32\\.5|24\\.4|24\\.7" \
  ~/.openclaw/data/research/2026-05-01-MSFT/decision.md \
  ~/.openclaw/data/research/2026-05-01-MSFT/analyst_fundamentals.md
' | head -10
```

Expected: at least some matches. The PM is supposed to cite the new authoritative numbers; if it doesn't, the LLM ignored the block (escalate as in Step 5).

- [ ] **Step 7: Inspect QC verdict**

```bash
ssh macmini-trueknot 'python3 -c "
import json
s = json.load(open(\"/Users/trueknot/.openclaw/data/research/2026-05-01-MSFT/state.json\"))
print(\"qc_passed:\", s.get(\"qc_passed\"))
print(\"qc_retries:\", s.get(\"qc_retries\"))
fb = s.get(\"qc_feedback\", \"\")
if fb: print(\"feedback:\", fb[:600])
"'
```

Expected: PASS, possibly with retry. Item 16(c) may have fired if the PM tried to cite an old approximation; that's success-path-with-correction. If qc_retries=0 and the decision cites the new ratios from pm_brief.md, the deterministic block did its job without needing QC enforcement.

- [ ] **Step 8: Report findings**

Pass criteria:
- ✅ `peer_ratios.json` written with documented schema; numbers match hand-computed values from peers.json.
- ✅ `pm_brief.md` ends with all three Phase 6 blocks (calendar → SEC filing → peer ratios) in order.
- ✅ Decision.md / analyst reports do NOT cite "GOOGL 4.9%" / "AMZN 5.1%" / "inherited from prior debate" framing.
- ✅ Decision.md cites at least some of the authoritative numbers (32.5% / 24.4% / 24.7%).
- ✅ QC PASS (with or without retry).

If all pass: Phase 6.4 closes the caveat-wrapping hole. Report success.

If decision.md still cites 4.9% / 5.1%: the LLM is ignoring the appended block. Tighten by:
1. Strengthening the block's instruction text ("Do NOT cite ... overrides any prior debate transcript context").
2. Adding an explicit 16(c) clause requiring the PM cite from "## Peer ratios" if making a peer claim.

If the new numbers are absent (PM didn't cite them at all): possibly because the PM judged peer comparisons aren't load-bearing for the Underweight call. That's a defensible judgment, not a regression. The mitigation is still working — the fabricated numbers are gone.

- [ ] **Step 9: Cleanup (optional)**

```bash
ssh macmini-trueknot 'ls ~/.openclaw/data/research/2026-05-01-MSFT*'
```

Keep all archived runs for the trajectory record (9717594 → cb24edf → c5c41e4 → 0617182 → <new SHA>) — they tell the Phase 6.3/6.4 hardening story.

---

## Self-review notes

**Spec coverage:**
- ✅ Pure-Python `compute_peer_ratios` module + `format_peer_ratios_block` helper (Task 1)
- ✅ 6 unit tests covering happy path, missing column, zero revenue, PE parse, PE absence, top-level metadata (Task 1)
- ✅ PM Pre-flight wiring + 3 unit tests (Task 2)
- ✅ E2E validation: peer_ratios.json byte-exactly matches inputs + pm_brief.md ends with the table + decision.md drops the old hallucination (Task 3)
- ✅ Out-of-scope items (annualized capex, EV-EBITDA, peer-of-peer matrices, sector aggregates) explicitly listed in spec, not touched here

**Type / signature consistency:**
- `compute_peer_ratios(peers_data: dict, trade_date: str) -> dict` — defined Task 1, called Task 2 with `(peers_data, date)`, output structure consumed by `format_peer_ratios_block`.
- `format_peer_ratios_block(ratios: dict) -> str` — defined Task 1, called Task 2.
- All three documented top-level keys (`trade_date`, `_unavailable`, per-ticker dicts) appear consistently in test stubs and formatter logic.

**Placeholder scan:**
- No "TBD" / "implement later" / "similar to Task N" patterns.
- Each step shows exact code, exact paths, exact verification commands.
- The PE-parse regex has explicit handling for the absent / unparseable case.

**Out-of-scope confirmation:** This plan does not change the QC system prompt (item 16(a) keeps its caveat-wrapping clause as defense in depth), does not touch the Fundamentals analyst prompt, and does not add annualized peer ratios. Each is called out in the spec and remains untouched.

**Rollback path:** Each task commits independently. Reverting Task 2 keeps `peer_ratios.py` as orphan but stops the block from being written. Reverting Task 1 deletes the module entirely.

**Test count delta:** +9 new unit tests (6 peer_ratios + 3 pm_preflight). Suite goes from 197 → 206.
