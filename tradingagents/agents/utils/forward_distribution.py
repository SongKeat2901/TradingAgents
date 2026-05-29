"""Block-bootstrap Monte Carlo 12-month forward distribution.

Resamples contiguous blocks of the trailing 36 months of daily log-returns
to build forward price paths, then classifies each path by its TERMINAL price
relative to the liquidity-level barriers (terminal-zone). Resulting Bull/Base/
Bear probabilities hard-anchor the Portfolio Manager's scenarios.

Terminal-zone semantics: Bull = path ends >= bull target, Bear = path ends
<= bear target, Base = path ends between barriers. Mutually exclusive →
sums to 100%. First-barrier-touch and ever-touched are stored as cross-check
fields (first_touch_prob, touch_prob) in scenarios.

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


def terminal_zone_probabilities(paths: list[list[float]], bull: float,
                                bear: float) -> dict[str, float]:
    """Classify each path by its TERMINAL price relative to the barriers:
    bull = terminal >= bull, bear = terminal <= bear, base = in between.
    Mutually exclusive → sums to 1.0. This is the headline scenario anchor;
    `first_barrier_probabilities` and `touch_probabilities` are stored as
    cross-checks (path-touched-the-level vs first-barrier-passage)."""
    if not paths:
        return {"bull": 0.0, "base": 1.0, "bear": 0.0}
    n_bull = n_bear = n_base = 0
    for path in paths:
        if not path:
            n_base += 1
            continue
        terminal = path[-1]
        if terminal >= bull:
            n_bull += 1
        elif terminal <= bear:
            n_bear += 1
        else:
            n_base += 1
    total = len(paths)
    return {"bull": n_bull / total, "base": n_base / total, "bear": n_bear / total}


# Minimum distance (as a fraction of spot) for a Bull/Bear target to count
# as a "meaningful" scenario rather than a trivial next-tick test. 5% is a
# reasonable v1 threshold: it filters out HVNs sitting within day-trader
# range but keeps multi-week reachable levels. Below this threshold, the
# selector falls through to the next-eligible level (or VAH/VAL/spot*1.15).
_MIN_TARGET_DISTANCE_PCT = 0.05


def _pick_targets(spot: float | None, vp: dict) -> tuple[float, float, float]:
    """REFINED (post-GOOGL smoke):
    Bull = nearest HVN at least 5% ABOVE spot, drawn from BOTH structural+
           tactical HVN lists merged; fallback lowest VAH at least 5% above;
           ultimately spot*1.15.
    Bear = nearest HVN at least 5% BELOW spot, merged HVN lists; fallback
           highest VAL at least 5% below; ultimately spot*0.85.
    Base = tactical 6-mo POC, but ONLY if it sits strictly between Bear and
           Bull (so the Base scenario is reachable); else spot itself (the
           Base = "neither barrier touched" interpretation).

    Two-stage refinement history:
      - v1 set Base = structural POC → could sit 50%+ below spot for
        re-rated stocks (GOOGL: structural POC $164 vs spot $383).
      - v2 (this) requires Bull/Bear to be meaningful distance from spot and
        forces Base into the (Bear, Bull) interval. GOOGL smoke showed
        Bull just $2.43 above spot and Base below Bear → fixed."""
    if spot is None or spot <= 0:
        return (0.0, 0.0, 0.0)

    s = vp.get("structural_36mo", {}) if vp else {}
    t = vp.get("tactical_6mo", {}) if vp else {}
    all_hvn: list[float] = list(s.get("hvn") or []) + list(t.get("hvn") or [])
    min_up = spot * (1 + _MIN_TARGET_DISTANCE_PCT)
    min_dn = spot * (1 - _MIN_TARGET_DISTANCE_PCT)

    above = sorted(x for x in all_hvn if x >= min_up)
    if above:
        bull = above[0]
    else:
        vah_above = [v for v in (s.get("vah"), t.get("vah"))
                     if v is not None and v >= min_up]
        bull = min(vah_above) if vah_above else spot * 1.15

    below = sorted((x for x in all_hvn if x <= min_dn), reverse=True)
    if below:
        bear = below[0]
    else:
        val_below = [v for v in (s.get("val"), t.get("val"))
                     if v is not None and v <= min_dn]
        bear = max(val_below) if val_below else spot * 0.85

    tac_poc = t.get("poc")
    if tac_poc is not None and bear < tac_poc < bull:
        base = float(tac_poc)
    else:
        base = float(spot)

    return (round(bull, 2), round(base, 2), round(bear, 2))


def _seed_for(ticker: str, trade_date: str) -> int:
    return abs(hash(f"{ticker}:{trade_date}")) % (2 ** 31)


def compute_forward_probabilities(ticker: str, trade_date: str, spot: float | None,
                                  closes: list[float], volume_profile: dict,
                                  horizon: int = 252, n_paths: int = 10000,
                                  block: int = 10) -> dict:
    """Full pipeline: targets from volume profile → block-bootstrap paths →
    terminal-zone scenario probabilities (headline). Deterministic on
    ticker+date. Returns a sentinel dict with unavailable_reason when spot
    is None."""
    if spot is None or spot <= 0:
        return {
            "ticker": ticker, "trade_date": trade_date, "spot": spot,
            "method": "block-bootstrap MC, terminal-zone scenarios",
            "n_paths": n_paths, "block": block, "horizon": horizon,
            "unavailable_reason": "spot price unavailable (close_on_date returned None)",
            "scenarios": {
                "bull": {"target": None, "probability": 0.0, "first_touch_prob": 0.0,
                         "touch_prob": 0.0},
                "base": {"target": None, "probability": 1.0, "first_touch_prob": 0.0},
                "bear": {"target": None, "probability": 0.0, "first_touch_prob": 0.0,
                         "touch_prob": 0.0},
            },
            "terminal_quantiles": {"p05": None, "p25": None, "p50": None,
                                   "p75": None, "p95": None},
        }
    bull, base, bear = _pick_targets(spot, volume_profile)
    rets = daily_log_returns(closes)
    paths = simulate_paths(spot, rets, horizon=horizon, n_paths=n_paths,
                           block=block, seed=_seed_for(ticker, trade_date))
    tz = terminal_zone_probabilities(paths, bull=bull, bear=bear)
    fb = first_barrier_probabilities(paths, bull=bull, bear=bear)
    touch = touch_probabilities(paths, bull=bull, bear=bear)
    terminals = sorted(p[-1] for p in paths)
    def q(frac: float) -> float:
        return round(terminals[min(int(frac * len(terminals)), len(terminals) - 1)], 2)
    return {
        "ticker": ticker, "trade_date": trade_date, "spot": spot,
        "method": "block-bootstrap MC, terminal-zone scenarios",
        "n_paths": n_paths, "block": block, "horizon": horizon,
        "seed": _seed_for(ticker, trade_date),
        "scenarios": {
            "bull": {"target": bull, "probability": round(tz["bull"], 4),
                     "first_touch_prob": round(fb["bull"], 4),
                     "touch_prob": round(touch["bull"], 4)},
            "base": {"target": base, "probability": round(tz["base"], 4),
                     "first_touch_prob": round(fb["base"], 4)},
            "bear": {"target": bear, "probability": round(tz["bear"], 4),
                     "first_touch_prob": round(fb["bear"], 4),
                     "touch_prob": round(touch["bear"], 4)},
        },
        "terminal_quantiles": {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50),
                               "p75": q(0.75), "p95": q(0.95)},
    }


def format_forward_block(out: dict) -> str:
    if out.get("unavailable_reason"):
        return (
            "\n\n## 12-month scenario probabilities (block-bootstrap MC on 36-mo history)\n\n"
            f"**Forward probabilities unavailable** — {out['unavailable_reason']}. "
            "**Do not cite scenario probabilities in this report.** If scenario "
            "analysis is essential, flag it as `(forward probability data unavailable)` "
            "and do not invent figures.\n"
        )
    s = out["scenarios"]
    def pct(x): return f"{x * 100:.0f}%"
    def tgt(x): return f"${x:.2f}" if x is not None else "(n/a)"
    def qtl(x): return f"${x:.2f}" if x is not None else "(n/a)"
    return (
        "\n\n## 12-month scenario probabilities (block-bootstrap MC on 36-mo history)\n\n"
        "| Scenario | Target | Probability (terminal zone) |\n|---|---|---|\n"
        f"| Bull | {tgt(s['bull']['target'])} | {pct(s['bull']['probability'])} |\n"
        f"| Base | {tgt(s['base']['target'])} | {pct(s['base']['probability'])} |\n"
        f"| Bear | {tgt(s['bear']['target'])} | {pct(s['bear']['probability'])} |\n\n"
        f"Terminal price quantiles: p05 {qtl(out['terminal_quantiles']['p05'])} · "
        f"p50 {qtl(out['terminal_quantiles']['p50'])} · "
        f"p95 {qtl(out['terminal_quantiles']['p95'])}.\n\n"
        "*Targets are volume-profile liquidity levels; probabilities are the "
        "fraction of simulated 12-month paths that END (at month 12) above the "
        "bull target, below the bear target, or in between (Base). "
        "**Use these targets and probabilities "
        "verbatim** in the Bull/Base/Bear scenario table — do not substitute "
        "judgement-based numbers. They sum to 100% by construction.*\n\n"
        "Cross-check fields in raw/forward_probabilities.json: `first_touch_prob` "
        "(path's first-barrier-touch classification — sums to 100%) and `touch_prob` "
        "(independent probability the path ever reached the level at all). "
        "Bull/Base/Bear headline above use terminal-zone (mutually-exclusive partition "
        "of where paths END at month 12).\n"
    )
