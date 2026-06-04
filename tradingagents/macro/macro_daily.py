"""Orchestrator + CLI for the daily macro regime engine.

Chains: data → pillars → regime → (per ticker) prices → betas → bias →
payload → sheet. Every network boundary lives in macro_data/plan_writer so
this module stays thin and testable with stubs.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from . import macro_data, pillars, regime as regime_mod, betas as betas_mod
from . import reports as reports_mod, bias as bias_mod, plan_writer
from .config import INDICATORS, IndicatorSpec

logger = logging.getLogger(__name__)

# Factor source tickers/series consumed by betas.build_factor_returns.
_FACTOR_SOURCES = {
    "tnx": ("yfinance", "^TNX"), "dxy": ("yfinance", "DX-Y.NYB"),
    "hy": ("fred", "BAMLH0A0HYM2"), "oil": ("yfinance", "CL=F"),
    "spy": ("yfinance", "SPY"), "iwf": ("yfinance", "IWF"),
    "iwd": ("yfinance", "IWD"),
}


def _load_factor_returns(as_of: str):
    raw = {}
    for key, (src, code) in _FACTOR_SOURCES.items():
        raw[key] = macro_data.load_series(
            IndicatorSpec(f"factor_{key}", src, code, "financial_conditions"), as_of)
    return betas_mod.build_factor_returns(raw)


def run(reports_dir, sheet_id, manifest_path, as_of=None, write=True,
        manifest_scope=False) -> dict:
    as_of = as_of or datetime.now().strftime("%Y-%m-%d")

    # 1. Regime (stock-independent)
    series = macro_data.load_all(INDICATORS, as_of)
    pillar_scores = pillars.score_all(series)
    regime = regime_mod.build(pillar_scores)
    logger.info("regime: %s gate=%s score=%.3f", regime.label, regime.gate, regime.score)

    # 2. Per-stock overlay (reports_dir may be a single base or a list of bases)
    factor_returns = _load_factor_returns(as_of)
    base_evs = reports_mod.latest_runs(reports_dir)
    if manifest_scope and manifest_path:
        # Plan universe = the published set: only tickers with a Drive-published
        # PDF in the manifest (so every row gets a link by construction).
        universe = set(plan_writer.load_manifest(manifest_path))
        dropped = [t for t in base_evs if t not in universe]
        base_evs = {t: be for t, be in base_evs.items() if t in universe}
        if dropped:
            logger.info("manifest-scope: kept %d, dropped %d non-published (%s)",
                        len(base_evs), len(dropped), ",".join(sorted(dropped)))
    biases = []
    levels = {}
    for ticker, be in base_evs.items():
        last_px = None
        try:
            px = macro_data.load_prices(ticker, as_of)
            last_px = float(px.iloc[-1])
            stock_ret = px.pct_change().dropna()
            b = betas_mod.compute_betas(ticker, stock_ret, factor_returns)
        except Exception as exc:  # noqa: BLE001 — one bad ticker shouldn't sink the run
            logger.warning("betas failed for %s: %s", ticker, exc)
            b = betas_mod.Betas(ticker, {f: 0.0 for f in betas_mod.FACTORS}, 0.0, "low", 0)
        biases.append(bias_mod.bias_stock(
            ticker, be.rating, regime, b, reports_mod.ev_pct(be)))
        ladder = reports_mod.scenario_ladder(be)
        levels[ticker] = {
            "last_px": last_px, "bear": ladder["bear"], "target": ladder["base"],
            "bull": ladder["bull"], "hard_stop": be.hard_stop,
        }

    # 3. Payload + write
    pdf_links = plan_writer.pdf_links_from_manifest(manifest_path) if manifest_path else {}
    payload = plan_writer.build_payload(regime, biases, pdf_links, levels)
    if write:
        plan_writer.write_to_sheet(plan_writer.to_grid(payload), sheet_id)
    return payload


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Daily macro regime engine → Trading Plan sheet")
    p.add_argument("--reports-dir", required=True, action="append",
                   help="research run-dir base (repeatable; newest-written wins on same date)")
    p.add_argument("--sheet-id", required=True, help="Trading Plan Google Sheet ID")
    p.add_argument("--manifest", default=None, help="pdf_ids.tsv for PDF hyperlinks")
    p.add_argument("--manifest-scope", action="store_true",
                   help="restrict the plan to manifest (published) tickers so every row links")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today, host local date)")
    p.add_argument("--no-write", action="store_true", help="compute only, don't touch the sheet")
    args = p.parse_args(argv)
    try:
        payload = run(args.reports_dir, args.sheet_id, args.manifest,
                      as_of=args.as_of, write=not args.no_write,
                      manifest_scope=args.manifest_scope)
    except Exception:
        logger.exception("macro daily run failed")
        return 1
    print(f"Regime: {payload['regime']['label']} | gate={payload['regime']['gate']} "
          f"| {len(payload['rows'])} names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
