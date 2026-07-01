# Annual Data Layer + Beneish M-Score — Implementation Plan (FA-101 WP4b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add annual financial statements to the data bundle and compute a deterministic Beneish M-score earnings-manipulation screen as a `pm_brief.md` block, completing FA-101 audit §10.

**Architecture:** Wire the existing `freq="annual"` fetchers into `_fetch_financials` (3 new bundle keys); parse them in `financials_parser` into a `beneish_inputs` current/prior dict (reusing `_parse_quarterly_csv`); compute the 8-ratio Beneish M-score in `distress_screens.py` beside Altman Z; wire the block into `researcher.py` and mandate analyst citation. Deterministic-block pattern throughout.

**Tech Stack:** Python 3, pytest (`unit` marker), no new deps. New network calls = 3 annual statement fetches per ticker (the fetchers already exist; same `yf_retry` path).

## Global Constraints

- **Annual fetch = `freq="annual"`** on the existing tools: `get_balance_sheet.invoke({"ticker":..., "curr_date":..., "freq":"annual"})` (the tools accept `freq`, default quarterly). Add keys `balance_sheet_annual`, `income_statement_annual`, `cashflow_annual` to the bundle.
- **Beneish M formula:** `M = −4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI`. **Flag `M > −1.78` → "elevated"**, else "normal".
- **Ratio defs (t=current col 0, p=prior col 1):** DSRI=(rec_t/sales_t)/(rec_p/sales_p); GMI=GM_p/GM_t [GM=(sales−cogs)/sales]; AQI=AQ_t/AQ_p [AQ=1−(current_assets+ppe)/total_assets]; SGI=sales_t/sales_p; DEPI=DR_p/DR_t [DR=dep/(dep+ppe)]; SGAI=(sga_t/sales_t)/(sga_p/sales_p); LVGI=Lev_t/Lev_p [Lev=(total_assets−total_equity)/total_assets]; TATA=(net_income_t−cfo_t)/total_assets_t.
- **Skip financials** (`fin.get("sector")` contains "financial") — same gate as Altman.
- **Free-data honesty:** any required input missing / any denominator 0 / prior year absent → `m_score=None` → block renders `n/a`. Never fabricate, never crash. Many small/foreign tickers will be n/a — correct.
- **No new QC-checklist item;** citation mandated in the fundamentals-analyst prompt only.
- **Reuse:** `_parse_quarterly_csv` (format-generic) for annual CSVs; `_div`/`_r` helpers in `distress_screens.py`; the Altman skip-financials pattern.
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root (baseline **779** — do not regress).

---

## File Structure

- Modify: `tradingagents/agents/researcher.py` — `_fetch_financials` (annual keys) + wire the Beneish block.
- Modify: `tradingagents/agents/utils/financials_parser.py` — parse annual CSVs → `beneish_inputs`.
- Modify: `tradingagents/agents/utils/distress_screens.py` — `compute_beneish_m` + `format_beneish_block`.
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` — cite the M-score.
- Create test: `tests/test_beneish.py`.
- Modify tests: `tests/test_financials_parser.py`, `tests/test_fundamentals_prompt.py`, `tests/test_distress_wiring.py`.

---

### Task 1: annual statements in the bundle

**Files:**
- Modify: `tradingagents/agents/researcher.py` (`_fetch_financials`, ~line 120-136)
- Test: `tests/test_beneish.py` (fetch portion)

**Interfaces:**
- Produces: `financials.json` bundle gains `balance_sheet_annual`, `income_statement_annual`, `cashflow_annual` (CSV strings). Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_beneish.py
import pytest

pytestmark = pytest.mark.unit


def test_fetch_financials_includes_annual_keys(monkeypatch):
    import tradingagents.agents.utils.agent_utils as au
    import tradingagents.agents.researcher as R

    class _FakeTool:
        def __init__(self, label): self.label = label
        def invoke(self, args):
            return f"{self.label}:{args.get('freq', 'quarterly')}"

    monkeypatch.setattr(au, "get_fundamentals", _FakeTool("F"))
    monkeypatch.setattr(au, "get_balance_sheet", _FakeTool("BS"))
    monkeypatch.setattr(au, "get_cashflow", _FakeTool("CF"))
    monkeypatch.setattr(au, "get_income_statement", _FakeTool("IS"))
    monkeypatch.setattr(R, "_fetch_financial_currency", lambda t: "USD")

    b = R._fetch_financials("MSFT", "2026-06-30")
    # quarterly (default) + annual variants present
    assert b["balance_sheet"] == "BS:quarterly"
    assert b["balance_sheet_annual"] == "BS:annual"
    assert b["income_statement_annual"] == "IS:annual"
    assert b["cashflow_annual"] == "CF:annual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_beneish.py::test_fetch_financials_includes_annual_keys -q`
Expected: FAIL — `KeyError: 'balance_sheet_annual'`.

- [ ] **Step 3: Implement**

In `_fetch_financials`, add the three annual keys to the returned dict (after the quarterly ones):

```python
        "balance_sheet_annual": get_balance_sheet.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
        "cashflow_annual": get_cashflow.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
        "income_statement_annual": get_income_statement.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
```

(The tools are imported inside the function from `agent_utils`; the test patches those module attributes, which the local import resolves at call time.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_beneish.py -q`
Expected: PASS. Note the researcher node itself isn't run (network); this test verifies the bundle wiring via stubbed tools.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_beneish.py
git commit -m "feat: fetch annual statements into the financials bundle"
```

---

### Task 2: parse `beneish_inputs` (current/prior annual)

**Files:**
- Modify: `tradingagents/agents/utils/financials_parser.py`
- Test: `tests/test_financials_parser.py`

**Interfaces:**
- Consumes: bundle `*_annual` CSV keys (Task 1).
- Produces: `parse_financials(...)["beneish_inputs"] = {"current": {...}, "prior": {...}}` with keys `receivables, sales, cogs, current_assets, ppe, total_assets, sga, depreciation, total_equity` (both), plus `net_income, cfo` (current only). Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_financials_parser.py
_BS_A = ("# BS annual\n\n,2025-12-31,2024-12-31,2023-12-31\n"
         "Total Assets,1000,900,\nCurrent Assets,400,360,\nNet PPE,500,450,\n"
         "Receivables,100,90,\nStockholders Equity,600,540,\n")
_IS_A = ("# IS annual\n\n,2025-12-31,2024-12-31,2023-12-31\n"
         "Total Revenue,1000,900,\nCost Of Revenue,600,540,\n"
         "Selling General And Administration,100,90,\nReconciled Depreciation,50,45,\n"
         "Net Income,120,100,\n")
_CF_A = ("# CF annual\n\n,2025-12-31,2024-12-31,2023-12-31\n"
         "Operating Cash Flow,110,95,\n")


def test_beneish_inputs_current_and_prior():
    from tradingagents.agents.utils.financials_parser import parse_financials
    bundle = {"ticker": "ACME", "trade_date": "2026-01-01", "financial_currency": "USD",
              "fundamentals": "# f\nMarket Cap: 1\n", "balance_sheet": "", "cashflow": "", "income_statement": "",
              "balance_sheet_annual": _BS_A, "income_statement_annual": _IS_A, "cashflow_annual": _CF_A}
    bi = parse_financials(bundle)["beneish_inputs"]
    assert bi["current"]["total_assets"] == 1000 and bi["prior"]["total_assets"] == 900
    assert bi["current"]["sales"] == 1000 and bi["prior"]["sales"] == 900
    assert bi["current"]["depreciation"] == 50 and bi["prior"]["depreciation"] == 45
    assert bi["current"]["sga"] == 100 and bi["prior"]["sga"] == 90
    assert bi["current"]["cfo"] == 110  # cfo current-only
    assert bi["current"]["net_income"] == 120


def test_beneish_inputs_missing_prior_year():
    from tradingagents.agents.utils.financials_parser import parse_financials
    bs1 = "# BS\n\n,2025-12-31\nTotal Assets,1000\n"  # only one year
    bundle = {"ticker": "ACME", "trade_date": "2026-01-01", "financial_currency": "USD",
              "fundamentals": "# f\n", "balance_sheet": "", "cashflow": "", "income_statement": "",
              "balance_sheet_annual": bs1, "income_statement_annual": "", "cashflow_annual": ""}
    bi = parse_financials(bundle)["beneish_inputs"]
    assert bi["prior"]["total_assets"] is None  # no prior column -> None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -k beneish -q`
Expected: FAIL — `KeyError: 'beneish_inputs'`.

- [ ] **Step 3: Implement**

In `parse_financials`, after the existing quarterly parses, add the annual parses and the `beneish_inputs` dict. Add to the imports/top nothing new (reuse `_parse_quarterly_csv`, `_row_at`). Insert before the final `return`:

```python
    _, bs_a = _parse_quarterly_csv(d.get("balance_sheet_annual", ""))
    _, is_a = _parse_quarterly_csv(d.get("income_statement_annual", ""))
    _, cf_a = _parse_quarterly_csv(d.get("cashflow_annual", ""))

    def _dep(rows_is, rows_cf, idx):
        # depreciation: income-statement 'Reconciled Depreciation', else cashflow D&A
        return (_row_at(rows_is, idx, "Reconciled Depreciation")
                or _row_at(rows_cf, idx, "Depreciation And Amortization",
                           "Depreciation Amortization Depletion"))

    def _annual_side(idx):
        return {
            "receivables": _row_at(bs_a, idx, "Receivables", "Accounts Receivable", "Net Receivables"),
            "current_assets": _row_at(bs_a, idx, "Current Assets", "Total Current Assets"),
            "ppe": _row_at(bs_a, idx, "Net PPE", "Net Property Plant And Equipment"),
            "total_assets": _row_at(bs_a, idx, "Total Assets"),
            "total_equity": _row_at(bs_a, idx, "Stockholders Equity", "Common Stock Equity"),
            "sales": _row_at(is_a, idx, "Total Revenue", "Operating Revenue"),
            "cogs": _row_at(is_a, idx, "Cost Of Revenue"),
            "sga": _row_at(is_a, idx, "Selling General And Administration"),
            "depreciation": _dep(is_a, cf_a, idx),
        }

    current_side = _annual_side(0)
    current_side["net_income"] = _row_at(is_a, 0, "Net Income", "Net Income Common Stockholders")
    current_side["cfo"] = _row_at(cf_a, 0, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    beneish_inputs = {"current": current_side, "prior": _annual_side(1)}
```

Add `"beneish_inputs": beneish_inputs,` to the returned dict. (`_row_at` returns None when the column/row is absent — incl. the all-None padding column and missing prior year, so degradation is automatic.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/financials_parser.py tests/test_financials_parser.py
git commit -m "feat: parse annual beneish_inputs (current/prior) from annual statements"
```

---

### Task 3: `compute_beneish_m` + block

**Files:**
- Modify: `tradingagents/agents/utils/distress_screens.py`
- Test: `tests/test_beneish.py`

**Interfaces:**
- Consumes: `parse_financials(...)` output (`beneish_inputs`, `sector`).
- Produces: `compute_beneish_m(fin: dict) -> dict`; `format_beneish_block(result: dict) -> str`. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_beneish.py
from tradingagents.agents.utils.distress_screens import compute_beneish_m, format_beneish_block

def _side(**kw):
    base = dict(receivables=100.0, sales=1000.0, cogs=600.0, current_assets=400.0,
                ppe=500.0, total_assets=1000.0, sga=100.0, depreciation=50.0, total_equity=600.0)
    base.update(kw)
    return base

# clean books: current == prior (all 8 ratios == 1), net_income==cfo (TATA=0) -> M = -2.48
_CLEAN = {"sector": "Technology", "beneish_inputs": {
    "current": dict(_side(), net_income=100.0, cfo=100.0),
    "prior": _side()}}


def test_beneish_clean_books_normal():
    r = compute_beneish_m(_CLEAN)
    assert r["applicable"] is True
    assert r["m_score"] == -2.48
    assert r["flag"] == "normal"


def test_beneish_manipulation_pattern_elevated():
    # spike receivables (DSRI), sales (SGI), and accruals (TATA: NI>>CFO)
    manip = {"sector": "Industrials", "beneish_inputs": {
        "current": dict(_side(sales=1500.0, receivables=300.0, cogs=900.0, total_assets=1200.0),
                        net_income=300.0, cfo=40.0),
        "prior": _side()}}
    r = compute_beneish_m(manip)
    assert r["flag"] == "elevated" and r["m_score"] > -1.78


def test_beneish_financials_skipped():
    r = compute_beneish_m(dict(_CLEAN, sector="Financial Services"))
    assert r["applicable"] is False


def test_beneish_missing_prior_na():
    r = compute_beneish_m({"sector": "Technology",
                           "beneish_inputs": {"current": dict(_side(), net_income=100.0, cfo=100.0),
                                              "prior": {k: None for k in _side()}}})
    assert r["m_score"] is None and r["flag"] is None


def test_beneish_block_render():
    block = format_beneish_block(compute_beneish_m(_CLEAN))
    assert "## Manipulation screen (Beneish M-score)" in block
    assert "normal" in block and "verbatim" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_beneish.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_beneish_m'`.

- [ ] **Step 3: Implement**

Append to `tradingagents/agents/utils/distress_screens.py` (reuse the existing `_div` and `_r` helpers already in the module):

```python
def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _add(a, b):
    return None if (a is None or b is None) else a + b


def compute_beneish_m(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    sector = fin.get("sector") or ""
    if "financial" in sector.lower():
        return {"model": "Beneish M", "applicable": False,
                "skip_reason": f"Beneish not meaningful for financials (sector: {sector})"}
    bi = fin.get("beneish_inputs") or {}
    c = bi.get("current") or {}
    p = bi.get("prior") or {}

    def gm(s):   # gross margin
        return _div(_sub(s.get("sales"), s.get("cogs")), s.get("sales"))

    def aq(s):   # asset quality = 1 - (CA+PPE)/TA
        num = _add(s.get("current_assets"), s.get("ppe"))
        frac = _div(num, s.get("total_assets"))
        return None if frac is None else 1 - frac

    def dr(s):   # depreciation rate = dep/(dep+ppe)
        return _div(s.get("depreciation"), _add(s.get("depreciation"), s.get("ppe")))

    def lev(s):  # leverage = total_liabilities / total_assets
        return _div(_sub(s.get("total_assets"), s.get("total_equity")), s.get("total_assets"))

    dsri = _div(_div(c.get("receivables"), c.get("sales")), _div(p.get("receivables"), p.get("sales")))
    gmi = _div(gm(p), gm(c))
    aqi = _div(aq(c), aq(p))
    sgi = _div(c.get("sales"), p.get("sales"))
    depi = _div(dr(p), dr(c))
    sgai = _div(_div(c.get("sga"), c.get("sales")), _div(p.get("sga"), p.get("sales")))
    lvgi = _div(lev(c), lev(p))
    tata = _div(_sub(c.get("net_income"), c.get("cfo")), c.get("total_assets"))

    ratios = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
              "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata}
    if any(v is None for v in ratios.values()):
        missing = [k for k, v in ratios.items() if v is None]
        return {"model": "Beneish M", "applicable": True, "m_score": None, "flag": None,
                "unavailable_reason": f"missing ratios: {', '.join(missing)}",
                **{k: _r(v) for k, v in ratios.items()}}

    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    flag = "elevated" if m > -1.78 else "normal"
    return {"model": "Beneish M", "applicable": True, "m_score": round(m, 2), "flag": flag,
            **{k: _r(v) for k, v in ratios.items()}}


def format_beneish_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable", True):
        return (f"\n\n## Manipulation screen (Beneish M-score) — not applicable "
                f"({r.get('skip_reason', 'financials')})\n\n"
                "*Beneish is not meaningful for this sector; do not cite an M-score for it.*\n")
    if r.get("m_score") is None:
        return (f"\n\n## Manipulation screen (Beneish M-score) — n/a (data unavailable: "
                f"{r.get('unavailable_reason', 'missing annual inputs')})\n\n"
                "*Do not cite an M-score; required annual inputs were missing.*\n")
    rows = "".join(f"| {k} | {r.get(k)} |\n" for k in ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"))
    return (
        f"\n\n## Manipulation screen (Beneish M-score) (computed from annual statements)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Beneish M** | **{r['m_score']}** |\n"
        f"| **Flag** | **{r['flag']}** (M > -1.78 = elevated manipulation risk) |\n"
        f"{rows}\n"
        "*Use the M-score and flag verbatim; do not recompute. Beneish flags RISK of "
        "earnings manipulation, not proof; it is unreliable for financials, recent IPOs, "
        "and heavy-M&A years.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_beneish.py -q`
Expected: PASS. If `test_beneish_clean_books_normal`'s `-2.48` disagrees, recompute by hand from the all-ratios==1, TATA==0 fixture and align the assertion to the code's rounding — do not change the formula.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/distress_screens.py tests/test_beneish.py
git commit -m "feat: add Beneish M-score manipulation screen"
```

---

### Task 4: wire block into researcher + analyst citation

**Files:**
- Modify: `tradingagents/agents/researcher.py` (after the Altman Z block)
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Test: `tests/test_distress_wiring.py`, `tests/test_fundamentals_prompt.py`

**Interfaces:**
- Consumes: `fin_parsed` (has `beneish_inputs`, `sector`), `raw`, `pm_brief_path`; the existing `raw/distress_screens.json`.
- Produces: Beneish result added to `raw/distress_screens.json` + a `## Manipulation screen` block in `pm_brief.md`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_distress_wiring.py
def test_beneish_block_composes(tmp_path):
    from tradingagents.agents.utils.distress_screens import compute_beneish_m, format_beneish_block
    from tests.test_beneish import _CLEAN
    block = format_beneish_block(compute_beneish_m(_CLEAN))
    pm = tmp_path / "pm_brief.md"; pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    assert "## Manipulation screen (Beneish M-score)" in pm.read_text(encoding="utf-8")
```

```python
# add to tests/test_fundamentals_prompt.py
def test_beneish_citation_mandated():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    assert "beneish" in fa._SYSTEM.lower() or "manipulation screen" in fa._SYSTEM.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distress_wiring.py tests/test_fundamentals_prompt.py::test_beneish_citation_mandated -q`
Expected: composition test PASSES; the prompt test FAILS (mandate not present).

- [ ] **Step 3: Implement**

READ `researcher.py` around the Altman Z block append. Immediately after it, in its own try/except (fail-open), add:

```python
        try:
            from tradingagents.agents.utils.distress_screens import compute_beneish_m, format_beneish_block
            m = compute_beneish_m(fin_parsed)
            _dscreens = {}
            _p = raw / "distress_screens.json"
            if _p.exists():
                try:
                    _dscreens = json.loads(_p.read_text(encoding="utf-8"))
                except Exception:
                    _dscreens = {}
            _dscreens = {"altman_z": _dscreens} if _dscreens and "altman_z" not in _dscreens and "beneish_m" not in _dscreens else _dscreens
            _dscreens["beneish_m"] = m
            _p.write_text(json.dumps(_dscreens, indent=2, default=str), encoding="utf-8")
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(format_beneish_block(m))
        except Exception as exc:
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Manipulation screen (Beneish M-score) — unavailable ({exc})\n\n"
                        "*Do not cite an M-score.*\n")
```

Note: the Altman task wrote `raw/distress_screens.json` as the Altman dict directly. To avoid clobbering, this Beneish step reads the existing file and stores results under a `beneish_m` key (nesting the pre-existing Altman result under `altman_z` if it wasn't already keyed). Confirm against the real Altman write shape in `researcher.py` and adapt so BOTH results persist (simplest robust form: the Altman block should ideally write under an `altman_z` key too — if it currently writes the bare Altman dict, this merge handles it; if you prefer, update the Altman write to `{"altman_z": z}` for symmetry and simplify this merge — either way, both results must survive in the file).

In `fundamentals_analyst.py` `_SYSTEM`, extend the distress-screen citation mandate to also cite the Beneish M-score + flag verbatim when applicable, matching the file's tone. Ensure the prompt-test assertion matches the wording added.

- [ ] **Step 4: Run tests + import smoke**

Run: `.venv/bin/python -m pytest tests/test_distress_wiring.py tests/test_fundamentals_prompt.py -q` and `.venv/bin/python -c "import tradingagents.agents.researcher"`
Then full: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all pass. Researcher node not unit-tested (network); verification = composition test + import smoke + full suite.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tradingagents/agents/analysts/fundamentals_analyst.py tests/test_distress_wiring.py tests/test_fundamentals_prompt.py
git commit -m "feat: wire Beneish M-score block into researcher + mandate analyst citation"
```

---

### Task 5: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 779 + new beneish/parser/wiring tests). Investigate any regression, esp. in `tests/test_distress_screens.py` (Altman) — confirm the `raw/distress_screens.json` shape change didn't break Altman's persistence.

- [ ] **Step 2: (Optional) live check**

On the mini: run one ticker and confirm `raw/pm_brief.md` shows `## Manipulation screen (Beneish M-score)` with a sane M (most legitimate large-caps < −1.78 = normal), a financial ticker shows "not applicable", `raw/distress_screens.json` holds BOTH altman + beneish, and a thin/foreign name degrades to n/a. Not required for merge.

---

## Out of scope (later / not this plan)

- Multi-year CAGRs (now trivial on the annual series) — a small follow-up on the accounting-ratios block.
- WP5 (new free sources), WP6 (qualitative prompts).
- A new QC-checklist item.

## Self-Review

- **Spec coverage:** annual fetch (Task 1), `beneish_inputs` current/prior parse w/ depreciation fallback + missing-year degradation (Task 2), Beneish 8-ratio M + M>−1.78 flag + skip-financials + degrade-to-n/a (Task 3), wiring both results into one JSON + analyst citation (Task 4), full-suite + Altman-persistence regression (Task 5). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step has real code; the Task-4 JSON-merge note is a concrete instruction (both results must persist), not a placeholder.
- **Type consistency:** `beneish_inputs` `{current,prior}` keys produced in Task 2 consumed by `compute_beneish_m` in Task 3; `compute_beneish_m(fin)->dict` / `format_beneish_block(result)->str` consumed identically in Task 4 + tests; reuses `_div`/`_r` from Task WP4a's `distress_screens.py`.
