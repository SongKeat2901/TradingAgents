# Intrinsic Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic intrinsic-value (IV) block to the researcher — a triangulated fair-value range computed in Python from `raw/` data, reconciled against the MC scenario EV, surfaced in the report, leak-scrubbed in the PDF, and audited as Tier 15.

**Architecture:** New pure-Python module `tradingagents/agents/utils/intrinsic_value.py` (mirrors `peer_ratios`/`net_debt`). Wired into `researcher.py` after the net-debt block: fetch risk-free, write `raw/intrinsic_value.json`, append a `## Intrinsic value` block to `pm_brief.md` (auto-rendered in the PDF Investment-Thesis section). Rating engine unchanged.

**Tech Stack:** Python 3.12, pytest (`-m unit`), yfinance (risk-free + FX), openpyxl (unrelated). Spec: `docs/superpowers/specs/2026-06-01-intrinsic-value-design.md`.

---

## File structure
- **Create** `tradingagents/agents/utils/intrinsic_value.py` — all IV logic (parser, classifier, methods, orchestration, formatter).
- **Create** `tests/test_intrinsic_value.py` — unit tests (`pytest.mark.unit`).
- **Modify** `tradingagents/agents/researcher.py` — risk-free fetch + wire block after net-debt.
- **Modify** `cli/research_pdf.py` — leak-scrub `raw/intrinsic_value.json` → "the valuation dataset".
- **Modify** `.claude/agents/report-auditor.md` — add Tier 15.

## Module interface (lock these signatures)
```python
# constants (module-level, echoed into JSON constants_note)
ERP = 0.05; TERMINAL_GROWTH = 0.025; HORIZON_YEARS = 5
NEAR_TERM_GROWTH_CAP = 0.25; DISCOUNT_RATE_FLOOR = 0.08; RISK_FREE_FALLBACK = 0.043
NAV_PROXY_TICKERS = {"MSTR"}

def parse_fundamentals(financials: dict) -> dict        # FCF,EPS,fwd_eps,beta,mktcap,ebitda,ni,sector,shares,tax,revenue,currency
def classify_valuation_profile(fund: dict, ticker: str) -> str   # STANDARD|UNPROFITABLE|FINANCIAL|NAV_PROXY
def cost_of_capital(fund, net_debt, risk_free) -> dict  # {cost_of_equity, wacc, ...}
def dcf_value(fund, wacc, near_g, term_g, horizon, net_debt) -> float | None
def epv_value(fund, wacc, net_debt) -> float | None
def multiples_value(fund, peer_ratios, net_debt) -> dict   # {pe_implied, ev_ebitda_implied}
def reverse_dcf_growth(fund, wacc, term_g, horizon, net_debt, price) -> float | None
def mc_ev_from_forward(forward_probabilities: dict) -> float | None
def compute_intrinsic_value(financials, net_debt, reference, peer_ratios, risk_free, forward_probabilities=None, ticker=None, fx_rate=None) -> dict
def format_intrinsic_value_block(iv: dict) -> str
def fetch_risk_free() -> float    # ^TNX/100, fallback RISK_FREE_FALLBACK
```

---

### Task 1: Fundamentals parser + profile classifier

**Files:** Create `tradingagents/agents/utils/intrinsic_value.py`; Test `tests/test_intrinsic_value.py`

- [ ] **Step 1 — failing tests.** Write `tests/test_intrinsic_value.py`:
```python
import pytest
pytestmark = pytest.mark.unit

_FUND_TXT = (
  "Name: Acme\nSector: Technology\nMarket Cap: 1000000000\n"
  "PE Ratio (TTM): 20\nForward PE: 16\nEPS (TTM): 5.0\nForward EPS: 6.0\n"
  "Beta: 1.2\nEBITDA: 200000000\nNet Income: 100000000\nFree Cash Flow: 90000000\n"
  "Revenue (TTM): 800000000\n")
_INC = "\nTax Rate For Calcs,0.15,0.15\nDiluted Average Shares,50000000,50000000\nEBIT,150000000,140000000\n"

def _fin(fund=_FUND_TXT, inc=_INC):
    return {"ticker":"ACME","financial_currency":"USD","fundamentals":fund,"income_statement":inc,"cashflow":"","balance_sheet":""}

def test_parse_fundamentals():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals
    f = parse_fundamentals(_fin())
    assert f["fcf"] == 90000000 and f["eps"] == 5.0 and f["forward_eps"] == 6.0
    assert f["beta"] == 1.2 and f["net_income"] == 100000000 and f["sector"] == "Technology"
    assert f["diluted_shares"] == 50000000 and abs(f["tax"]-0.15) < 1e-9 and f["currency"]=="USD"

def test_classify_profiles():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_valuation_profile
    assert classify_valuation_profile(parse_fundamentals(_fin()), "ACME") == "STANDARD"
    loss = _FUND_TXT.replace("Net Income: 100000000","Net Income: -50000000")
    assert classify_valuation_profile(parse_fundamentals(_fin(loss)), "X") == "UNPROFITABLE"
    fin = _FUND_TXT.replace("Sector: Technology","Sector: Financial Services")
    assert classify_valuation_profile(parse_fundamentals(_fin(fin)), "X") == "FINANCIAL"
    assert classify_valuation_profile(parse_fundamentals(_fin()), "MSTR") == "NAV_PROXY"
```
- [ ] **Step 2 — run, expect fail** (`ModuleNotFoundError`). `.venv/bin/python -m pytest tests/test_intrinsic_value.py -q`
- [ ] **Step 3 — implement** `parse_fundamentals` (regex `^<Label>:\s*<num>` from the fundamentals block; `Sector:` as text; pull `Diluted Average Shares`/`Tax Rate For Calcs`/`EBIT` col-0 from income_statement CSV via the same `_col0` style as peer_ratios) and `classify_valuation_profile` (NAV_PROXY if ticker in NAV_PROXY_TICKERS; FINANCIAL if sector startswith "Financial"/contains "Bank"; UNPROFITABLE if ni<=0 or fcf<=0; else STANDARD). Define the module constants at top.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit** `feat(intrinsic-value): fundamentals parser + profile classifier`.

### Task 2: Cost of capital (CAPM + WACC)

- [ ] **Step 1 — failing test:**
```python
def test_cost_of_capital():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, cost_of_capital
    cc = cost_of_capital(parse_fundamentals(_fin()), {"net_debt":-50000000}, risk_free=0.04)
    # CoE = 0.04 + 1.2*0.05 = 0.10
    assert abs(cc["cost_of_equity"] - 0.10) < 1e-6
    assert cc["wacc"] >= 0.08  # floor respected
```
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `cost_of_capital`: `coe = max(risk_free + beta*ERP, DISCOUNT_RATE_FLOOR)`. WACC: if net debt > 0 and total debt known, weight after-tax cost of debt (`risk_free + 0.02` spread) by debt/(debt+equity); else WACC = coe. Net cash (net_debt<0) → WACC = coe. Return `{cost_of_equity, cost_of_debt, wacc, weight_debt}`.
- [ ] **Step 4 — run, expect pass.** **Step 5 — commit.**

### Task 3: The four valuation methods

- [ ] **Step 1 — failing tests** for `dcf_value`, `epv_value`, `multiples_value`, `reverse_dcf_growth`:
```python
def test_dcf_known():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, dcf_value
    f = parse_fundamentals(_fin())  # fcf 90M, shares 50M
    v = dcf_value(f, wacc=0.10, near_g=0.10, term_g=0.025, horizon=5, net_debt={"net_debt":-50000000})
    assert v is not None and 30 < v < 120   # sane per-share magnitude
def test_epv_floor():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, epv_value
    v = epv_value(parse_fundamentals(_fin()), wacc=0.10, net_debt={"net_debt":-50000000})
    # EBIT 150M*(1-.15)=127.5M /0.10 = 1.275B EV +50M cash /50M sh
    assert abs(v - ((150000000*0.85/0.10 + 50000000)/50000000)) < 1.0
def test_reverse_dcf_recovers_growth():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, dcf_value, reverse_dcf_growth
    f = parse_fundamentals(_fin())
    price = dcf_value(f, 0.10, 0.12, 0.025, 5, {"net_debt":0})
    g = reverse_dcf_growth(f, 0.10, 0.025, 5, {"net_debt":0}, price)
    assert abs(g - 0.12) < 0.01
def test_multiples():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, multiples_value
    pr = {"PEERA":{"ttm_pe":18,"ttm_ebitda":1,"net_debt":0},"PEERB":{"ttm_pe":22,"ttm_ebitda":1,"net_debt":0}}
    m = multiples_value(parse_fundamentals(_fin()), pr, {"net_debt":-50000000})
    assert abs(m["pe_implied"] - 20*5.0) < 1e-6   # median peer P/E (20) * EPS (5)
```
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** the four functions:
  - `dcf_value`: project FCF over `horizon` with growth fading linearly from `near_g`→`term_g`; PV each (`/(1+wacc)**t`); terminal = `FCF_h*(1+term_g)/(wacc-term_g)` PV'd; EV=ΣPV+PV(terminal); equity=EV−net_debt; /diluted_shares. Return None if shares/FCF missing or wacc<=term_g.
  - `epv_value`: `ebit*(1-tax)/wacc` → EV; equity=EV−net_debt; /shares. None if EBIT/shares missing or EBIT<=0.
  - `multiples_value`: peer-median `ttm_pe` × `eps` → pe_implied; peer-median (computed) EV/EBITDA × own EBITDA − net_debt → /shares → ev_ebitda_implied. Skip a sub-method if peers lack the field.
  - `reverse_dcf_growth`: bisection on near_g ∈ [−0.5, CAP] s.t. `dcf_value(...near_g...) ≈ price` (50 iters). None if not bracketable.
- [ ] **Step 4 — run, expect pass.** **Step 5 — commit.**

### Task 4: MC-EV helper + orchestration (compute_intrinsic_value)

- [ ] **Step 1 — failing tests:** profile-gated assembly + skip-with-reason + reconciliation + currency caveat:
```python
def test_compute_standard_assembles_range_and_reconciliation():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    fwd = {"scenarios":[{"probability":0.3,"target":140},{"probability":0.4,"target":110},{"probability":0.3,"target":80}]}
    iv = compute_intrinsic_value(_fin(), {"net_debt":-50000000}, {"reference_price":100.0},
                                 {"PEERA":{"ttm_pe":18}}, risk_free=0.04, forward_probabilities=fwd, ticker="ACME")
    assert iv["profile"]=="STANDARD"
    fv=iv["fair_value"]; assert fv["bear"] <= fv["base"] <= fv["bull"]
    r=iv["reconciliation"]; assert r["mc_ev"]==pytest.approx(0.3*140+0.4*110+0.3*80)
    assert r["flag"] in ("AGREE","DIVERGE")
def test_unprofitable_skips_dcf_with_reason():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    loss=_FUND_TXT.replace("Net Income: 100000000","Net Income: -50000000").replace("Free Cash Flow: 90000000","Free Cash Flow: -10000000")
    iv=compute_intrinsic_value(_fin(loss), {"net_debt":0}, {"reference_price":50.0}, {}, risk_free=0.04, ticker="X")
    assert iv["profile"]=="UNPROFITABLE"
    assert any(s["method"]=="dcf" for s in iv["skipped_methods"])
def test_currency_mismatch_no_fx_caveat():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    f=_fin(); f["financial_currency"]="TWD"
    iv=compute_intrinsic_value(f, {"net_debt":0}, {"reference_price":100.0}, {}, risk_free=0.04, ticker="X", fx_rate=None)
    assert iv["currency"]=="TWD" and any("currency" in s["reason"].lower() for s in iv["skipped_methods"]) or iv.get("fx_caveat")
```
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `mc_ev_from_forward` (Σ p·target; tolerate key variants `probability/prob/p`, `target/price`) and `compute_intrinsic_value`: parse fund; classify; cost_of_capital; near_g from fwd_eps/eps−1 capped at CAP (fallback term_g); run profile-appropriate methods (collect skips with reasons); bear/base/bull via DCF ±2pp growth/±1pp wacc (STANDARD) or method spread; base=DCF base or median; margin_of_safety vs reference price; reconciliation via mc_ev; if currency!=USD and fx_rate None → set fx_caveat + skip per-share USD comparison; assemble the full JSON dict per spec C2 incl. `constants_note`.
- [ ] **Step 4 — run, expect pass.** **Step 5 — commit.**

### Task 5: format_intrinsic_value_block

- [ ] **Step 1 — failing test:** block contains a "## Intrinsic value" header, the fair-value range, margin of safety, reconciliation flag, the methods used + skipped-with-reasons, and a constants line; a "not computable" path when no methods ran.
```python
def test_format_block_standard():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value, format_intrinsic_value_block
    iv=compute_intrinsic_value(_fin(), {"net_debt":-50000000}, {"reference_price":100.0}, {"PEERA":{"ttm_pe":18}}, risk_free=0.04, ticker="ACME")
    b=format_intrinsic_value_block(iv)
    assert "## Intrinsic value" in b and "Margin of safety" in b and "WACC" in b
```
- [ ] **Step 2 — run, expect fail.** **Step 3 — implement** formatter (markdown; mirror peer_ratios block tone; print inputs/assumptions, the range table, reconciliation line, skipped methods, and the verbatim-use footer). **Step 4 — pass.** **Step 5 — commit.**

### Task 6: risk-free fetch + researcher wiring + PDF scrub + Tier 15

- [ ] **Step 1 — implement `fetch_risk_free`** in the module (`yf.Ticker("^TNX").history(period="5d")` last close /100; except → RISK_FREE_FALLBACK). No network in tests — covered by stubbing in compute tests; add a tiny test that the fallback returns a float in [0,0.1] when yfinance import fails (monkeypatch).
- [ ] **Step 2 — wire `researcher.py`** after the net-debt block append: compute `risk_free=fetch_risk_free()`; read `forward_probabilities` (already in memory as `forward_probabilities`); `iv=compute_intrinsic_value(financials, net_debt, reference, ratios, risk_free, forward_probabilities=forward_probabilities, ticker=ticker)`; write `raw/intrinsic_value.json`; append `format_intrinsic_value_block(iv)` to `pm_brief.md`. Guard with try/except that, on failure, appends a "(intrinsic value unavailable — <err>)" note rather than crashing the run.
- [ ] **Step 3 — PDF scrub:** add to `cli/research_pdf.py` `_AGENTIC_VOCAB_REPLACEMENTS` (after the other raw/ bare-filename mappings, before the generic catch-all): `(r"\bintrinsic_value\.json\b", "the valuation dataset")` and ensure the generic `raw/<f>.json` catch-all covers `raw/intrinsic_value.json`. Add a customer-facing test in `tests/test_research_pdf_customer.py` asserting `intrinsic_value` filename scrubbed.
- [ ] **Step 4 — Tier 15 in `.claude/agents/report-auditor.md`:** add a `T15 Intrinsic value` bullet (recompute IV from raw/intrinsic_value.json inputs + stated assumptions; verify reconciliation arithmetic; verify applicability honesty; any report IV figure not matching the artifact = FAIL) and add `T15` to the YAML `tiers` map in the output template.
- [ ] **Step 5 — run full unit suite** `.venv/bin/python -m pytest -q -m unit` (expect all green) and **commit** `feat(intrinsic-value): wire block into researcher + PDF scrub + Tier 15 audit`.

---

## Post-implementation (per the goal — verify to A+)
1. Deploy to macmini (push + pull + `pip install -e .`).
2. Re-run each fresh ticker through the IV-enabled stack (cadence-run pattern, `TRADINGRESEARCH_NO_TELEGRAM=1`).
3. Audit each Tier 1-15 via `report-auditor`; confirm a truthful IV section and A+.
4. Promote/register; refresh `TrueKnot-Research-Summary.xlsx`.
5. Hand over only when every fresh report is verified A+ with intrinsic value.

## Self-review notes
- Spec coverage: B1 classifier→T1; B2 methods→T3; B3 assumptions→T2+T4 constants; B4 currency→T4; B5 triangulation→T4; C1 reconciliation→T4; C2 JSON→T4; C3 honesty→T4 skips; C4 testing→T1-5; C5 audit/PDF→T6. ✓
- Types consistent: `parse_fundamentals` dict keys (fcf/eps/forward_eps/beta/net_income/sector/diluted_shares/tax/ebitda/revenue/currency/mktcap) reused across tasks; `net_debt` accessed via `["net_debt"]`. ✓
- No placeholders: method math specified; reverse-DCF = bisection; reconciliation = Σp·target. ✓
