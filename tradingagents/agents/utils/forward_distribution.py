"""Block-bootstrap Monte Carlo 12-month forward distribution.

Resamples contiguous blocks of the trailing 36 months of daily log-returns
to build forward price paths, then classifies each path by the first
liquidity level it reaches (first-barrier-touch). Resulting Bull/Base/
Bear probabilities hard-anchor the Portfolio Manager's scenarios.

Deterministic on (ticker, trade_date) seed. stdlib only.
"""
from __future__ import annotations

import math
import random


def daily_log_returns(closes: list[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def simulate_paths(spot: float, returns: list[float], horizon: int = 252,
                   n_paths: int = 10000, block: int = 10,
                   seed: int = 0) -> list[list[float]]:
    """Block-bootstrap: assemble each path from random contiguous blocks of
    the historical return series, then exponentiate the cumulative sum off
    `spot`. Returns n_paths lists of `horizon` prices."""
    rng = random.Random(seed)
    if not returns or spot <= 0:
        return [[spot] * horizon for _ in range(n_paths)]
    max_start = max(len(returns) - block, 0)
    paths: list[list[float]] = []
    for _ in range(n_paths):
        seq: list[float] = []
        while len(seq) < horizon:
            start = rng.randint(0, max_start)
            seq.extend(returns[start:start + block])
        seq = seq[:horizon]
        price = spot
        path: list[float] = []
        for r in seq:
            price *= math.exp(r)
            path.append(price)
        paths.append(path)
    return paths


def first_barrier_probabilities(paths: list[list[float]], bull: float,
                                bear: float) -> dict[str, float]:
    """Classify each path by the FIRST barrier it touches: bull-first,
    bear-first, or neither (base). Mutually exclusive → sums to 1.0."""
    if not paths:
        return {"bull": 0.0, "base": 1.0, "bear": 0.0}
    n_bull = n_bear = n_base = 0
    for path in paths:
        outcome = "base"
        for px in path:
            if bull is not None and px >= bull:
                outcome = "bull"
                break
            if bear is not None and px <= bear:
                outcome = "bear"
                break
        if outcome == "bull":
            n_bull += 1
        elif outcome == "bear":
            n_bear += 1
        else:
            n_base += 1
    total = len(paths)
    return {"bull": n_bull / total, "base": n_base / total, "bear": n_bear / total}


def touch_probabilities(paths: list[list[float]], bull: float,
                        bear: float) -> dict[str, float]:
    """Independent (non-exclusive) probability each level is touched at all."""
    if not paths:
        return {"bull": 0.0, "bear": 0.0}
    tb = sum(1 for p in paths if any(px >= bull for px in p)) / len(paths)
    tr = sum(1 for p in paths if any(px <= bear for px in p)) / len(paths)
    return {"bull": tb, "bear": tr}


def _pick_targets(spot: float, vp: dict) -> tuple[float, float, float]:
    """REFINED (post-smoke-test on GOOGL):
    Bull = nearest HVN above spot, drawn from BOTH structural+tactical HVN
           lists merged; fallback lowest VAH above spot; ultimately spot*1.15.
    Base = tactical 6-mo POC (current acceptance, near spot); fallback spot.
           (The structural POC can sit 50%+ below spot for re-rated stocks,
            making it nonsensical as a base scenario.)
    Bear = nearest HVN below spot, merged HVN lists; fallback highest VAL
           below spot; ultimately spot*0.85."""
    s = vp.get("structural_36mo", {}) if vp else {}
    t = vp.get("tactical_6mo", {}) if vp else {}
    all_hvn: list[float] = list(s.get("hvn") or []) + list(t.get("hvn") or [])
    above = sorted(x for x in all_hvn if x > spot)
    below = sorted((x for x in all_hvn if x < spot), reverse=True)

    if above:
        bull = above[0]
    else:
        vah_above = [v for v in (s.get("vah"), t.get("vah")) if v is not None and v > spot]
        bull = min(vah_above) if vah_above else spot * 1.15

    if below:
        bear = below[0]
    else:
        val_below = [v for v in (s.get("val"), t.get("val")) if v is not None and v < spot]
        bear = max(val_below) if val_below else spot * 0.85

    base = t.get("poc") if t.get("poc") is not None else spot
    return (round(bull, 2), round(base, 2), round(bear, 2))


def _seed_for(ticker: str, trade_date: str) -> int:
    return abs(hash(f"{ticker}:{trade_date}")) % (2 ** 31)


def compute_forward_probabilities(ticker: str, trade_date: str, spot: float,
                                  closes: list[float], volume_profile: dict,
                                  horizon: int = 252, n_paths: int = 10000,
                                  block: int = 10) -> dict:
    """Full pipeline: targets from volume profile → block-bootstrap paths →
    first-barrier-touch scenario probabilities. Deterministic on ticker+date."""
    bull, base, bear = _pick_targets(spot, volume_profile)
    rets = daily_log_returns(closes)
    paths = simulate_paths(spot, rets, horizon=horizon, n_paths=n_paths,
                           block=block, seed=_seed_for(ticker, trade_date))
    fb = first_barrier_probabilities(paths, bull=bull, bear=bear)
    touch = touch_probabilities(paths, bull=bull, bear=bear)
    terminals = sorted(p[-1] for p in paths)
    def q(frac: float) -> float:
        return round(terminals[min(int(frac * len(terminals)), len(terminals) - 1)], 2)
    return {
        "ticker": ticker, "trade_date": trade_date, "spot": spot,
        "method": "block-bootstrap MC, first-barrier-touch",
        "n_paths": n_paths, "block": block, "horizon": horizon,
        "seed": _seed_for(ticker, trade_date),
        "scenarios": {
            "bull": {"target": bull, "probability": round(fb["bull"], 4),
                     "touch_prob": round(touch["bull"], 4)},
            "base": {"target": base, "probability": round(fb["base"], 4)},
            "bear": {"target": bear, "probability": round(fb["bear"], 4),
                     "touch_prob": round(touch["bear"], 4)},
        },
        "terminal_quantiles": {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50),
                               "p75": q(0.75), "p95": q(0.95)},
    }


def format_forward_block(out: dict) -> str:
    s = out["scenarios"]
    def pct(x): return f"{x * 100:.0f}%"
    return (
        "\n\n## 12-month scenario probabilities (block-bootstrap MC on 36-mo history)\n\n"
        "| Scenario | Target | Probability (first-barrier touch) |\n|---|---|---|\n"
        f"| Bull | ${s['bull']['target']:.2f} | {pct(s['bull']['probability'])} |\n"
        f"| Base | ${s['base']['target']:.2f} | {pct(s['base']['probability'])} |\n"
        f"| Bear | ${s['bear']['target']:.2f} | {pct(s['bear']['probability'])} |\n\n"
        f"Terminal price quantiles: p05 ${out['terminal_quantiles']['p05']:.2f} · "
        f"p50 ${out['terminal_quantiles']['p50']:.2f} · "
        f"p95 ${out['terminal_quantiles']['p95']:.2f}.\n\n"
        "*Targets are volume-profile liquidity levels; probabilities are the "
        "fraction of simulated 12-month paths whose first barrier touch is that "
        "level (Base = neither touched). **Use these targets and probabilities "
        "verbatim** in the Bull/Base/Bear scenario table — do not substitute "
        "judgement-based numbers. They sum to 100% by construction.*\n"
    )
