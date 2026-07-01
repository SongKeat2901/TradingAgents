"""Researcher — deterministic data fetcher (no LLM).

Replaces the bind_tools pattern in the original 4 analysts. Pulls all
data the multi-agent pipeline needs once, up front, and writes it to
`<output_dir>/raw/` as JSON / Markdown. Every downstream agent reads
from raw/ — no agent-side data fetching, no ReAct loops over tools.

Wraps the existing dataflows utilities (yfinance, alpha_vantage) as
plain Python functions called from this single deterministic step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


_OHLCV_HEADER = "Date,Open,High,Low,Close,Volume"


def _parse_ohlcv_rows(ohlcv_str: str) -> list[tuple[str, float, float, float]]:
    """Parse the get_stock_data CSV string into (date, high, low, close) rows.

    Skips comment lines (#...) and the header row. Returns rows in the order
    they appear (chronological for yfinance). Malformed rows are skipped.
    """
    rows: list[tuple[str, float, float, float]] = []
    for line in ohlcv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            rows.append((parts[0], float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return rows


def _close_on_or_before(rows: list[tuple[str, float, float, float]], date: str) -> Optional[float]:
    """Return the Close on `date` or the most recent trading day before it."""
    candidate: Optional[float] = None
    for d, _h, _l, c in rows:
        if d <= date:
            candidate = c
        else:
            break
    return candidate


def _close_with_date_on_or_before(
    rows: list[tuple[str, float, float, float]], date: str
) -> tuple[Optional[str], Optional[float]]:
    """Return (close_date, close) on `date` or the most recent trading day
    before it. `close_date` is the ACTUAL session date — it differs from
    `date` when the trade_date's session hasn't closed yet (e.g. an intraday
    'today' run: trade_date 2026-06-02 → close_date 2026-06-01). Recording it
    stops the report from mislabeling a prior close as the trade_date's close
    (MSFT 2026-05-29 reported $426.99 'on 2026-05-29' — actually the May-28
    close)."""
    result: tuple[Optional[str], Optional[float]] = (None, None)
    for d, _h, _l, c in rows:
        if d <= date:
            result = (d, c)
        else:
            break
    return result


def _ytd_high_low(rows: list[tuple[str, float, float, float]], date: str) -> tuple[Optional[float], Optional[float]]:
    """Return (max High, min Low) across rows in the same calendar year as `date`, up to and including `date`."""
    year = date[:4]
    highs: list[float] = []
    lows: list[float] = []
    for d, h, l, _c in rows:
        if d.startswith(year) and d <= date:
            highs.append(h)
            lows.append(l)
    return (max(highs) if highs else None, min(lows) if lows else None)


def _latest_indicator_value(indicator_str: str) -> Optional[float]:
    """Pull the most recent numeric value from a get_indicators output string.

    Format is `## <ind> values from <start> to <end>:\\n\\n<DATE>: <val>\\n...`
    The first non-N/A `<DATE>: <number>` line is the most recent observation.
    """
    if not indicator_str or not isinstance(indicator_str, str):
        return None
    for m in re.finditer(r"^\d{4}-\d{2}-\d{2}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", indicator_str, re.MULTILINE):
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None

def _fetch_financial_currency(ticker: str) -> str | None:
    """Look up yfinance reporting currency for `ticker`. None on any failure.

    yfinance returns balance-sheet cells in the company's reporting currency
    (TWD for Taiwan-domiciled, JPY for Japan-domiciled, etc.) — NOT in USD.
    Phase 7.5 v1.3 reads this from raw/net_debt.json so the validator can
    skip non-USD reporters cleanly rather than flagging spurious drift
    against TWD-denominated cells stored as if USD.
    """
    try:
        import yfinance as yf
        from tradingagents.dataflows.stockstats_utils import yf_retry
        info = yf_retry(lambda: yf.Ticker(ticker.upper()).info)
        ccy = info.get("financialCurrency") if isinstance(info, dict) else None
        return str(ccy).upper() if ccy else None
    except Exception:
        return None


def _fetch_financials(ticker: str, date: str) -> dict[str, Any]:
    """Pull fundamentals + balance sheet + cashflow + income statement for one ticker."""
    from tradingagents.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_income_statement,
    )
    return {
        "ticker": ticker,
        "trade_date": date,
        "financial_currency": _fetch_financial_currency(ticker),
        "fundamentals": get_fundamentals.invoke({"ticker": ticker, "curr_date": date}),
        "balance_sheet": get_balance_sheet.invoke({"ticker": ticker, "curr_date": date}),
        "cashflow": get_cashflow.invoke({"ticker": ticker, "curr_date": date}),
        "income_statement": get_income_statement.invoke({"ticker": ticker, "curr_date": date}),
        "balance_sheet_annual": get_balance_sheet.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
        "cashflow_annual": get_cashflow.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
        "income_statement_annual": get_income_statement.invoke({"ticker": ticker, "curr_date": date, "freq": "annual"}),
    }


def _fetch_news(ticker: str, date: str) -> dict[str, Any]:
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_global_news, get_news
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    return {
        "ticker_news": get_news.invoke({"ticker": ticker, "start_date": start, "end_date": date}),
        "global_news": get_global_news.invoke({"curr_date": date}),
    }


def _fetch_insider(ticker: str, date: str) -> dict[str, Any]:
    # get_insider_transactions accepts only `ticker`; date is unused but kept for signature symmetry.
    from tradingagents.agents.utils.agent_utils import get_insider_transactions
    return {"transactions": get_insider_transactions.invoke({"ticker": ticker})}


def _fetch_social(ticker: str, date: str) -> dict[str, Any]:
    # No dedicated social tool exists; reuse get_news as a sentiment-adjacent source.
    # T8/T9 prompts know to focus on social/sentiment-relevant items.
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_news
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    return {"social_news": get_news.invoke({"ticker": ticker, "start_date": start, "end_date": date})}


def _fetch_prices(ticker: str, date: str) -> dict[str, Any]:
    """5y OHLCV history. Note: get_stock_data returns a string; T6 parses for spot/ATR/etc."""
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_stock_data
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return {
        "ohlcv": get_stock_data.invoke(
            {"symbol": ticker, "start_date": start, "end_date": date}
        ),
    }


def _fetch_indicators(ticker: str, date: str) -> dict[str, Any]:
    """Pull a fixed set of indicators per ticker. T5 (TA agent) parses the strings."""
    from tradingagents.agents.utils.agent_utils import get_indicators
    indicators = ("close_50_sma", "close_200_sma", "rsi", "macd", "boll_ub", "boll_lb", "atr")
    return {
        ind: get_indicators.invoke({"symbol": ticker, "indicator": ind, "curr_date": date})
        for ind in indicators
    }


def _build_reference(ticker, date, prices, indicators):
    """Build the reference dict (was inline in fetch_research_pack)."""
    rows = _parse_ohlcv_rows(prices.get("ohlcv", ""))
    close_date, close_on_date = _close_with_date_on_or_before(rows, date)
    ytd_high, ytd_low = _ytd_high_low(rows, date)
    # Name the ACTUAL close date. When it equals the trade_date the session has
    # closed; when earlier (intraday 'today' run) it's the latest available
    # close — say so explicitly so downstream agents never write "closes at $X
    # on <trade_date>" for a price that is really a prior session's close.
    if close_date and close_date == date:
        _ref_source = f"yfinance close of {close_date}"
    elif close_date:
        _ref_source = (
            f"yfinance close of {close_date} (latest available on/before "
            f"trade_date {date}; {date}'s session has not closed/indexed)"
        )
    else:
        _ref_source = f"yfinance close on or before {date}"
    return {
        "ticker": ticker,
        "trade_date": date,
        "reference_price": close_on_date,
        "reference_close_date": close_date,
        "reference_price_source": _ref_source,
        "spot_50dma": _latest_indicator_value(indicators.get("close_50_sma", "")),
        "spot_200dma": _latest_indicator_value(indicators.get("close_200_sma", "")),
        "ytd_high": ytd_high,
        "ytd_low": ytd_low,
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
    _ref_ok = lambda d: _id(d) and d.get("reference_price") is not None
    reference, r_ref = reuse_or_fetch(raw, "reference.json",
                                      lambda: _build_reference(ticker, date, prices, _fetch_indicators(ticker, date)),
                                      reuse, sanity=_ref_ok)
    news = _fetch_news(ticker, date)     # always fresh (date-sensitive)
    social = _fetch_social(ticker, date)  # always fresh (date-sensitive)
    bundle = {"financials": financials, "prices": prices, "insider": insider,
              "peers_data": peers_data, "reference": reference, "news": news, "social": social}
    reused = {"financials": r_fin, "prices": r_px, "insider": r_ins, "peers": r_peers, "reference": r_ref}
    return bundle, reused


def fetch_research_pack(state: dict) -> None:
    """Fetch all data needed by the multi-agent pipeline. Writes to raw/.

    Required state keys: `company_of_interest`, `trade_date`, `peers`, `raw_dir`.
    """
    ticker = state["company_of_interest"]
    date = state["trade_date"]
    peers = state.get("peers", [])
    raw = Path(state["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    # Per-ticker bundles + peers + reference: routed through the raw-reuse
    # loader (Phase A). With reuse off, behavior is unchanged from before —
    # every artifact is fetched fresh in the same order. With reuse on,
    # financials/prices/insider/peers/reference are loaded from a prior
    # attempt's raw/*.json when present and valid; news/social always fetch
    # fresh (date-sensitive, not reproducible from a stale snapshot).
    reuse = state.get("reuse_raw", False)
    bundle, reused = _gather_raw(ticker, date, peers, raw, reuse)
    financials = bundle["financials"]
    prices = bundle["prices"]
    insider = bundle["insider"]
    peers_data = bundle["peers_data"]
    reference = bundle["reference"]
    news = bundle["news"]
    social = bundle["social"]
    close_on_date = reference["reference_price"]
    if reuse:
        hit = ", ".join(k for k, v in reused.items() if v) or "none"
        print(f"[raw-reuse] reused {sum(reused.values())}/{len(reused)} artifacts "
              f"({hit}); re-fetched fresh: news, social — reused data is verbatim "
              f"from the prior attempt; not for correcting bad data")
    else:
        print("[raw-reuse] off (fetched all fresh)")

    # Write everything
    (raw / "financials.json").write_text(json.dumps(financials, indent=2, default=str), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers_data, indent=2, default=str), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps(insider, indent=2, default=str), encoding="utf-8")
    (raw / "social.json").write_text(json.dumps(social, indent=2, default=str), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps(prices, indent=2, default=str), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")

    # Phase 8.x: deterministic volume profile (liquidity levels). Computed
    # here, written to raw/, and appended to pm_brief.md so TA agents and
    # the forward-distribution model consume real volume-by-price levels.
    # Must run BEFORE classification so volume_profile can be passed in.
    from tradingagents.agents.utils.volume_profile import (
        compute_volume_profile, format_volume_profile_block,
    )
    volume_profile = compute_volume_profile(prices.get("ohlcv", ""))
    (raw / "volume_profile.json").write_text(
        json.dumps(volume_profile, indent=2, default=str), encoding="utf-8"
    )

    # Phase 7.15+ forward distribution: block-bootstrap MC 12-month scenario
    # probabilities anchored to volume-profile liquidity levels. Runs after
    # volume_profile.json is written and before classification so the targets
    # reflect the same levels the classifier uses.
    from tradingagents.agents.utils.forward_distribution import (
        compute_forward_probabilities, format_forward_block,
    )
    from tradingagents.agents.utils.volume_profile import parse_ohlcv as _parse_ohlcv_for_fwd
    _closes = [r[4] for r in _parse_ohlcv_for_fwd(prices.get("ohlcv", ""))]
    forward_probabilities = compute_forward_probabilities(
        ticker, date, spot=close_on_date, closes=_closes,
        volume_profile=volume_profile,
    )
    (raw / "forward_probabilities.json").write_text(
        json.dumps(forward_probabilities, indent=2, default=str), encoding="utf-8"
    )

    # Phase-6 stochasticity mitigation: pure-Python deterministic classifier.
    # See tradingagents/agents/utils/classifier.py + the design spec at
    # docs/superpowers/specs/2026-05-04-deterministic-classifier-design.md
    from tradingagents.agents.utils.classifier import compute_classification
    classification = compute_classification(reference, prices.get("ohlcv", ""),
                                            volume_profile=volume_profile)
    (raw / "classification.json").write_text(
        json.dumps(classification, indent=2, default=str), encoding="utf-8"
    )

    # Phase-6.2 calendar.json is written by PM Pre-flight (which runs before
    # this node and has the peer list). Read-only here.

    # Phase-6.4 deterministic peer ratios: compute authoritative
    # capex/revenue + op margin + P/E from peers_data (already in memory)
    # and append a verbatim "## Peer ratios" block to pm_brief.md (which
    # PM Pre-flight already created). The peer-ratios block must land
    # AFTER the Phase 6.2 calendar table and Phase 6.3 SEC filing footer.
    # Lives in the Researcher (not PM Pre-flight) because peers.json is
    # only written here — PM Pre-flight runs before this node and would
    # find peers_path.exists() == False. See docs/superpowers/specs/
    # 2026-05-05-deterministic-peer-ratios-design.md.
    # Phase 6.4 invariant: the deterministic peer-ratios block must land in
    # pm_brief.md every run, or downstream LLM agents fill the void with
    # fabricated peer numbers (RCL 2026-05-06: peers.json was {}, the block
    # never appended, decision.md cited NCLH/CCL/VIK ratios that came from
    # nowhere). Three guarded paths replace the prior silent-skip:
    #
    #   1. pm_brief.md missing → PM Pre-flight failed; raise.
    #   2. peers_data empty   → upstream peer-discovery returned nothing;
    #                           raise rather than ship a peer-less brief.
    #   3. all peers unavailable → write peer_ratios.json with the
    #                              `_unavailable` list AND append an explicit
    #                              "do not fabricate" warning block so the
    #                              LLM sees the gap and refuses to invent
    #                              numbers.
    pm_brief_path = raw / "pm_brief.md"
    if not pm_brief_path.exists():
        raise RuntimeError(
            "Phase 6.4 invariant: pm_brief.md does not exist before the "
            "Researcher's peer-ratios block runs. PM Pre-flight likely "
            "failed silently; investigate before re-running."
        )
    if not peers_data:
        raise RuntimeError(
            "Phase 6.4 invariant: peers_data is empty (peers.json wrote `{}`). "
            "Upstream peer-discovery returned no peers; the LLM will fabricate "
            "peer ratios downstream if this run is allowed to proceed. Fix the "
            "peer-lookup path for this ticker rather than shipping without a "
            "peer-ratios block."
        )

    from tradingagents.agents.utils.peer_ratios import (
        compute_peer_ratios,
        format_peer_ratios_block,
    )
    ratios = compute_peer_ratios(peers_data, date)
    (raw / "peer_ratios.json").write_text(
        json.dumps(ratios, indent=2, default=str),
        encoding="utf-8",
    )

    # If every peer is unavailable, the standard format renders a table of
    # `(unavailable)` rows — technically correct but the same trailing
    # "Use these values verbatim" footer can read as "use these unavailable
    # cells", which the LLM may interpret as license to substitute memory.
    # Override with an explicit "do not fabricate" warning instead.
    unavailable = set(ratios.get("_unavailable", []))
    peer_keys = [k for k in ratios.keys() if k not in ("trade_date", "_unavailable")]
    if peer_keys and unavailable == set(peer_keys):
        peer_block = (
            f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {date})\n\n"
            "**All peers unavailable** — yfinance returned degenerate or missing "
            "data (revenue/operating-income/capex rows) for every peer in "
            "raw/peers.json. **Do not cite peer ratios in this report.** If a "
            "peer comparison is essential to the thesis, flag it as `(peer data "
            "unavailable)` and do not invent figures from memory.\n"
        )
    else:
        peer_block = format_peer_ratios_block(ratios)
        if not peer_block:
            # Defensive: peers_data was non-empty but format returned ""
            # (all entries were non-dict, etc.). Surface the gap loudly.
            peer_block = (
                f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {date})\n\n"
                "**Peer-ratios table could not be rendered** from raw/peers.json. "
                "**Do not cite peer ratios in this report.**\n"
            )

    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(peer_block)

    # Phase 6.5 deterministic net-debt block: the 2026-05-06 audit found
    # APA cited `Total Debt $6.0B` (raw=$4.59B) and ORCL had definition
    # drift between yfinance Net Debt ($96.2B) and `Total Debt − Cash`
    # ($114B). Append the authoritative balance-sheet cells to pm_brief.md
    # so downstream agents quote them verbatim and QC item 16(b) becomes a
    # cell-match check.
    from tradingagents.agents.utils.net_debt import (
        compute_net_debt,
        format_net_debt_block,
    )
    net_debt = compute_net_debt(financials)
    (raw / "net_debt.json").write_text(
        json.dumps(net_debt, indent=2, default=str), encoding="utf-8"
    )
    net_debt_block = format_net_debt_block(net_debt)
    if not net_debt_block:
        # Total Debt cell missing — surface explicitly rather than letting
        # the LLM invent figures from memory.
        net_debt_block = (
            f"\n\n## Net debt (computed from raw/financials.json balance_sheet, "
            f"trade_date {date})\n\n"
            f"**Balance-sheet net-debt cells unavailable** — "
            f"{net_debt.get('unavailable_reason') or 'required rows missing'}. "
            f"**Do not cite net-debt arithmetic in this report.** If leverage is "
            f"essential to the thesis, flag it as `(balance-sheet data unavailable)` "
            f"and do not invent figures from memory.\n"
        )
    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(net_debt_block)

    # Intrinsic value (2026-06-01): deterministic triangulated fair value +
    # reconciliation against the MC scenario EV. Computed in Python from the
    # data already in memory (financials + net_debt + reference + peer ratios +
    # forward_probabilities); the LLM only interprets it. Decision-support only —
    # the rating still derives from the scenario engine. Runs AFTER net-debt so
    # it can cite net debt, and reads forward_probabilities for the EV reconcile.
    from tradingagents.agents.utils.intrinsic_value import (
        compute_intrinsic_value,
        fetch_risk_free,
        format_intrinsic_value_block,
    )
    iv = None
    try:
        risk_free = fetch_risk_free()
        iv = compute_intrinsic_value(
            financials, net_debt, reference, ratios, risk_free,
            forward_probabilities=forward_probabilities, ticker=ticker,
        )
        (raw / "intrinsic_value.json").write_text(
            json.dumps(iv, indent=2, default=str), encoding="utf-8"
        )
        iv_block = format_intrinsic_value_block(iv)
    except Exception as exc:  # noqa: BLE001 - IV must never crash the run
        iv_block = f"\n\n## Intrinsic value\n\n(intrinsic value unavailable — {exc})\n"
    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(iv_block)

    # Accounting ratios + relative valuation multiples (deterministic).
    # Reuses financials/net_debt/ratios already computed above, and `iv` only
    # opportunistically (for its wacc/market_cap). Each block is guarded in
    # its OWN try/except — mirroring the self-contained net-debt/peer-ratios
    # blocks above — so a failure in compute_intrinsic_value (which leaves
    # `iv = None`, not unbound; see the `iv = None` init before the IV
    # try-block) or in one of these two computations never drops the other.
    from tradingagents.agents.utils.financials_parser import parse_financials
    fin_parsed = parse_financials(financials)

    try:
        from tradingagents.agents.utils.accounting_ratios import (
            compute_accounting_ratios, format_accounting_ratios_block,
        )
        wacc = (iv.get("inputs", {}) or {}).get("wacc") if isinstance(iv, dict) else None
        acct = compute_accounting_ratios(fin_parsed, wacc=wacc, net_debt=net_debt)
        (raw / "accounting_ratios.json").write_text(
            json.dumps(acct, indent=2, default=str), encoding="utf-8")
        acct_block = format_accounting_ratios_block(
            acct, fin_parsed.get("trade_date"), fin_parsed.get("as_of_quarter"))
    except Exception as exc:  # noqa: BLE001 - this block must never crash the run
        acct_block = (
            f"\n\n## Accounting ratios\n\n"
            f"**Accounting ratios unavailable** — {exc}. "
            f"**Do not cite accounting-ratio figures (margins, returns, leverage) "
            f"in this report.** If profitability or leverage context is essential "
            f"to the thesis, flag it as `(accounting ratios unavailable)` and do "
            f"not invent figures from memory.\n"
        )
    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(acct_block)

    # --- Distress screen (Altman Z'') ---
    try:
        from tradingagents.agents.utils.distress_screens import (
            compute_altman_z, format_distress_block,
        )
        z = compute_altman_z(fin_parsed)
        (raw / "distress_screens.json").write_text(
            json.dumps(z, indent=2, default=str), encoding="utf-8")
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(format_distress_block(z))
    except Exception as exc:  # noqa: BLE001 - this block must never crash the run
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## Distress screen (Altman Z″) — unavailable ({exc})\n\n"
                    "*Do not cite a Z-score.*\n")

    try:
        from tradingagents.agents.utils.relative_multiples import (
            compute_relative_multiples, format_relative_multiples_block,
        )
        mc = (iv.get("inputs", {}) or {}).get("market_cap") if isinstance(iv, dict) else None
        if mc is None:
            mc = fin_parsed.get("market_cap")
        nd_val = (net_debt or {}).get("net_debt")
        rel = compute_relative_multiples(
            fin_parsed, market_cap=mc, net_debt=nd_val, peers=ratios,
            forward_eps=fin_parsed.get("forward_eps"))
        (raw / "relative_multiples.json").write_text(
            json.dumps(rel, indent=2, default=str), encoding="utf-8")
        rel_block = format_relative_multiples_block(rel, fin_parsed.get("trade_date"))
    except Exception as exc:  # noqa: BLE001 - this block must never crash the run
        rel_block = (
            f"\n\n## Relative valuation multiples\n\n"
            f"**Relative valuation multiples unavailable** — {exc}. "
            f"**Do not cite relative-multiple figures (P/E, EV/EBITDA, peer medians) "
            f"in this report.** If relative valuation is essential to the thesis, "
            f"flag it as `(relative multiples unavailable)` and do not invent "
            f"figures from memory.\n"
        )
    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(rel_block)

    # Phase 6.9 deterministic latest-session block: the 2026-05-08 COIN run
    # surfaced a forward-projection failure mode where the Market Analyst
    # invented a "May 8 trade-date close of $206.50 on 14.39M shares" when
    # yfinance had only indexed through the 2026-05-07 close ($192.96 on
    # 8.64M shares). The fabricated number then anchored the entire trading
    # plan (no-new-longs at $206.50, R/R math, "trim into intraday strength
    # toward $210-$215"). Append a deterministic block stating exactly which
    # session is the latest in raw/prices.json and forbidding the LLM from
    # citing a later "trade-date close".
    from tradingagents.agents.utils.latest_session import (
        compute_latest_session,
        format_latest_session_block,
    )
    latest_session = compute_latest_session(prices, date)
    (raw / "latest_session.json").write_text(
        json.dumps(latest_session, indent=2, default=str), encoding="utf-8"
    )
    latest_session_block = format_latest_session_block(latest_session)
    if not latest_session_block:
        latest_session_block = (
            f"\n\n## Latest available session (from raw/prices.json)\n\n"
            f"**Price-history unavailable** — "
            f"{latest_session.get('reason') or 'no parseable OHLCV rows'}. "
            f"**Do not cite intraday or trade-date close prices in this report.** "
            f"If price action is essential to the thesis, flag it as `(price "
            f"data unavailable)` and do not invent figures from memory.\n"
        )
    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(latest_session_block)

    # --- Recent closes (deterministic; pins specific-date closes) ---
    try:
        from tradingagents.agents.utils.recent_closes import (
            compute_recent_closes, format_recent_closes_block,
        )
        rc = compute_recent_closes(prices, date)
        (raw / "recent_closes.json").write_text(
            json.dumps(rc, indent=2, default=str), encoding="utf-8")
        rc_block = format_recent_closes_block(rc)
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(rc_block)
    except Exception as exc:  # noqa: BLE001 - this block must never crash the run
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## Recent closes — unavailable ({exc})\n\n"
                    "*Do not cite a closing price for any specific date.*\n")

    with (raw / "pm_brief.md").open("a", encoding="utf-8") as fh:
        fh.write(format_volume_profile_block(volume_profile))

    with (raw / "pm_brief.md").open("a", encoding="utf-8") as fh:
        fh.write(format_forward_block(forward_probabilities))
