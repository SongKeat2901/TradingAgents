"""Deterministic technical-setup classifier (Phase-6 stochasticity mitigation).

The Anthropic API does not produce deterministic outputs at temperature=0
(verified empirically on 2026-05-04). To remove inter-run variance from the
load-bearing setup-classification call, we replace the LLM's free-form
classification with a pure-Python rule engine.

The classifier reads `reference.json` (spot price + DMAs + YTD high/low +
ATR) and the OHLCV CSV string (from `get_stock_data`). It applies a
6-rule taxonomy in priority order (first match wins) and computes
asymmetry math (upside/downside targets and reward/risk ratio).

Output is consumed verbatim by both TA Agent v1 and TA Agent v2 — they
inject the result into their SystemMessage as ground truth and write
prose around it.
"""

from __future__ import annotations

# `statistics.median` is used for the BREAKOUT 5-day volume vs 90-day median
# check. The sigma-based "big move" detection from an earlier draft was
# replaced with ATR-multiple math (see _move_in_atr_multiples below).
import statistics
from typing import Any


_INDETERMINATE = {
    "setup_class": "INDETERMINATE",
    "gap_to_50dma_pct": None,
    "gap_to_200dma_pct": None,
    "ma_alignment": "unknown",
    "recent_volume_signal": "unknown",
    "upside_target": None,
    "upside_pct": None,
    "downside_target": None,
    "downside_pct": None,
    "reward_risk_ratio": None,
    "rationale": "Reference data missing or null; cannot classify deterministically.",
}


def _parse_ohlcv(ohlcv_csv: str) -> list[tuple[str, float, float, float, float, float]]:
    rows = []
    for line in ohlcv_csv.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
        except ValueError:
            continue
    return rows


def _gap_pct(spot: float, target: float) -> float:
    return (spot - target) / spot * 100.0


def _is_top_decile_volume(latest_volume: float, recent_volumes: list[float]) -> bool:
    """Return True if `latest_volume` is in the top decile of `recent_volumes`.

    Uses linear interpolation (NumPy-style) for the 90th percentile so that
    thin histories don't collapse to "must equal the max".
    """
    if not recent_volumes:
        return False
    sorted_vols = sorted(recent_volumes)
    n = len(sorted_vols)
    if n == 1:
        return latest_volume >= sorted_vols[0]
    # Linear interpolation: rank = 0.9 * (n - 1)
    rank = 0.9 * (n - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    weight = rank - lower_idx
    p90 = sorted_vols[lower_idx] * (1 - weight) + sorted_vols[upper_idx] * weight
    return latest_volume >= p90


def _move_in_atr_multiples(latest_close: float, prev_close: float, atr: float) -> float:
    """Return |close-to-close move| expressed as ATR multiples.

    Using ATR as the volatility baseline avoids the sigma-inflation problem
    that occurs when synthetic (or genuinely flat) price history drives
    historical stdev to near-zero, making even tiny moves appear as many σ.
    """
    if atr <= 0:
        return 0.0
    return abs(latest_close - prev_close) / atr


def _compute_asymmetry(
    setup_class: str, spot: float, spot_50dma: float, spot_200dma: float,
    ytd_high: float, ytd_low: float, rows: list,
) -> dict[str, float | None]:
    recent_30 = rows[-30:] if len(rows) >= 30 else rows
    recent_30_high = max((r[2] for r in recent_30), default=spot)
    recent_30_low = min((r[3] for r in recent_30), default=spot)

    if setup_class == "CAPITULATION":
        upside, downside = spot_200dma, ytd_low
    elif setup_class == "BREAKDOWN":
        upside = spot_50dma
        downside = max(ytd_low, spot * 0.90)
    elif setup_class == "DOWNTREND":
        upside, downside = spot_200dma, ytd_low
    elif setup_class == "CONSOLIDATION":
        upside, downside = recent_30_high, recent_30_low
    elif setup_class == "UPTREND":
        upside = max(recent_30_high, spot_200dma * 1.05)
        downside = spot_50dma
    elif setup_class == "BREAKOUT":
        upside = recent_30_high * 1.08
        downside = spot_50dma
    else:
        return {"upside_target": None, "upside_pct": None, "downside_target": None, "downside_pct": None, "reward_risk_ratio": None}

    upside_pct = (upside - spot) / spot * 100.0
    downside_pct = (downside - spot) / spot * 100.0
    rr = abs(upside_pct) / abs(downside_pct) if downside_pct != 0 else None
    return {
        "upside_target": round(upside, 2),
        "upside_pct": round(upside_pct, 2),
        "downside_target": round(downside, 2),
        "downside_pct": round(downside_pct, 2),
        "reward_risk_ratio": round(rr, 1) if rr is not None else None,
    }


def compute_classification(reference: dict, ohlcv_csv: str, history_window: int = 90) -> dict[str, Any]:
    """Apply the 6-class rule engine in priority order; return classification dict.

    Required reference keys: reference_price, spot_50dma, spot_200dma,
    ytd_high, ytd_low, atr_14. If any is None, returns INDETERMINATE.
    """
    spot = reference.get("reference_price")
    ma50 = reference.get("spot_50dma")
    ma200 = reference.get("spot_200dma")
    ytd_high = reference.get("ytd_high")
    ytd_low = reference.get("ytd_low")
    atr = reference.get("atr_14")

    if any(v is None for v in (spot, ma50, ma200, ytd_high, ytd_low, atr)):
        return dict(_INDETERMINATE)

    rows = _parse_ohlcv(ohlcv_csv)
    if not rows:
        return dict(_INDETERMINATE)

    gap_50 = _gap_pct(spot, ma50)
    gap_200 = _gap_pct(spot, ma200)
    bear_aligned = ma50 < ma200
    bull_aligned = ma50 > ma200

    last = rows[-1]
    last_close = last[4]
    last_volume = last[5]
    recent = rows[-history_window:] if len(rows) >= history_window else rows
    recent_volumes = [r[5] for r in recent[:-1]]
    recent_closes = [r[4] for r in recent]

    avg_50d_volume = (sum(r[5] for r in rows[-50:-1]) / max(len(rows[-50:-1]), 1)) if len(rows) >= 2 else 0
    median_90d_volume = statistics.median(r[5] for r in rows[-90:-1]) if len(rows) >= 2 else 0

    top_decile_vol = _is_top_decile_volume(last_volume, recent_volumes)
    prev_close = rows[-2][4] if len(rows) >= 2 else last_close
    atr_multiples = _move_in_atr_multiples(last_close, prev_close, atr)
    big_move = atr_multiples > 1.5

    # Rule priority (first match wins):
    # CAPITULATION > BREAKDOWN > BREAKOUT > CONSOLIDATION > UPTREND > DOWNTREND
    #
    # CONSOLIDATION is checked before DOWNTREND so that tight-range near-MA
    # setups are not swallowed by the bear-alignment catch-all.
    # BREAKOUT is checked before CONSOLIDATION so a fresh golden-cross with
    # volume is not mislabelled as tight-range consolidation.

    if top_decile_vol and big_move and bear_aligned:
        setup = "CAPITULATION"
        rationale = f"Top-decile volume ({last_volume:.0f}) on a {atr_multiples:.1f}× ATR move; 50-DMA below 200-DMA (bear alignment)."
        vol_signal = "capitulation"
    elif spot < ma50 and bear_aligned and gap_200 < -8.0 and last_volume > 1.5 * avg_50d_volume:
        setup = "BREAKDOWN"
        rationale = f"Spot below 50-DMA, bear MA alignment, {abs(gap_200):.1f}% below 200-DMA, latest volume {last_volume / avg_50d_volume:.1f}× 50-day avg."
        vol_signal = "above_average"
    elif spot > ma200 and bull_aligned and (ma50 - ma200) / ma200 < 0.02:
        recent_5d_avg = sum(r[5] for r in rows[-5:]) / max(len(rows[-5:]), 1)
        if recent_5d_avg > median_90d_volume:
            setup = "BREAKOUT"
            rationale = f"Recent 50/200 golden cross ({ma50:.2f} just above {ma200:.2f}); 5-day volume average above 90-day median."
            vol_signal = "breakout_volume"
        else:
            setup = "UPTREND"
            rationale = f"Spot {abs(gap_200):.1f}% above 200-DMA with bull MA alignment (50-DMA above 200-DMA)."
            vol_signal = "normal"
    elif (
        abs(gap_50) < 3.0 and abs(gap_200) < 8.0
        and (max(r[2] for r in rows[-10:]) - min(r[3] for r in rows[-10:])) < 1.5 * atr
    ):
        setup = "CONSOLIDATION"
        rationale = f"Spot near both MAs (gap-50: {gap_50:+.1f}%, gap-200: {gap_200:+.1f}%); 10-day range tight (< 1.5× ATR-14)."
        vol_signal = "below_average"
    elif spot > ma200 and bull_aligned:
        setup = "UPTREND"
        rationale = f"Spot {abs(gap_200):.1f}% above 200-DMA with bull MA alignment (50-DMA above 200-DMA)."
        vol_signal = "normal"
    elif spot < ma200 and bear_aligned:
        setup = "DOWNTREND"
        rationale = f"Spot {abs(gap_200):.1f}% below 200-DMA with bear MA alignment (50-DMA below 200-DMA); no breakdown/capitulation trigger."
        vol_signal = "normal"
    else:
        return dict(_INDETERMINATE) | {"rationale": "No rule matched; spot/MA configuration is ambiguous."}

    asym = _compute_asymmetry(setup, spot, ma50, ma200, ytd_high, ytd_low, rows)
    alignment = "bullish_aligned" if bull_aligned else ("bearish_aligned" if bear_aligned else "mixed")

    return {
        "setup_class": setup,
        "gap_to_50dma_pct": round(gap_50, 2),
        "gap_to_200dma_pct": round(gap_200, 2),
        "ma_alignment": alignment,
        "recent_volume_signal": vol_signal,
        "upside_target": asym["upside_target"],
        "upside_pct": asym["upside_pct"],
        "downside_target": asym["downside_target"],
        "downside_pct": asym["downside_pct"],
        "reward_risk_ratio": asym["reward_risk_ratio"],
        "rationale": rationale,
    }
