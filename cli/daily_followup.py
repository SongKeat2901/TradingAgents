"""Phase 8 MVP — daily follow-up against research thesis.

Walks every current research run, fetches yfinance close data since each run's
trade date, scores realized return vs the published scenarios, detects whether
any Bull/Base/Bear price target has been crossed, and emits a markdown digest
suitable for Telegram delivery.

Usage:
    .venv/bin/python -m cli.daily_followup
    .venv/bin/python -m cli.daily_followup --telegram

This is intentionally regex-based (no LLM re-parse) so it can run as a cron
job after market close without burning API quota.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


_RESEARCH_DIR = Path.home() / ".openclaw" / "data" / "research"


@dataclass
class Scenario:
    name: str
    probability: float | None = None
    target: float | None = None


@dataclass
class TriggerHit:
    direction: str         # "above_bull" / "above_base" / "below_bear" / "stop_breached"
    level: float
    date_crossed: str
    bar_high: float
    bar_low: float
    bar_close: float


@dataclass
class FollowupResult:
    ticker: str
    research_date: str
    rating: str
    reference_price: float
    latest_date: str
    latest_close: float
    days_elapsed: int
    realized_return_pct: float
    spy_return_pct: float | None
    alpha_pct: float | None
    scenarios: list[Scenario] = field(default_factory=list)
    ev: float | None = None
    scenario_bucket: str = ""
    crossings: list[TriggerHit] = field(default_factory=list)
    hard_stop: float | None = None
    stop_breached: bool = False


_SCEN_ROW = re.compile(
    r"^\|\s*(?P<name>Bull|Base|Bear|Tail)\s*\|"
    r"\s*(?P<prob>[\d.]+%?)\s*\|"
    r"\s*\$?(?P<target>[\d,]+(?:\.\d+)?)\s*\|",
    re.MULTILINE,
)

_REF_PRICE = re.compile(
    r"Reference\s+price:?\s*\*?\*?\s*\$?(?P<px>[\d,]+\.\d+)",
    re.IGNORECASE,
)

_RATING = re.compile(
    r"\*\*Rating\s+implication:?\*\*\s*\*?\*?"
    r"\s*(?P<rating>BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)",
    re.IGNORECASE,
)

_EV_NUM = re.compile(
    r"EV\s*=\s*\*?\*?\$?(?P<ev>[\d,]+\.\d+)",
    re.IGNORECASE,
)

# Various hard-stop phrasings the reports use
_HARD_STOP = re.compile(
    r"(?:hard\s+stop|thesis\s+hard\s+stop|hard\s+exit)"
    r"[^\n.]{0,60}?\$(?P<stop>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_research(run_dir: Path) -> dict | None:
    """Pull ticker, trade date, reference price, scenarios, rating, EV, hard stop
    from a single research run. Returns None when essentials are missing.
    """
    state_path = run_dir / "state.json"
    decision_path = run_dir / "decision.md"
    if not state_path.exists() or not decision_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    ticker = state.get("company_of_interest")
    trade_date = state.get("trade_date")
    if not ticker or not trade_date:
        return None

    decision = decision_path.read_text(encoding="utf-8")

    ref_match = _REF_PRICE.search(decision)
    if not ref_match:
        return None
    reference_price = float(ref_match.group("px").replace(",", ""))

    rating_match = _RATING.search(decision)
    rating = rating_match.group("rating").upper() if rating_match else "UNKNOWN"

    ev_match = _EV_NUM.search(decision)
    ev = float(ev_match.group("ev").replace(",", "")) if ev_match else None

    scenarios: list[Scenario] = []
    for m in _SCEN_ROW.finditer(decision):
        prob_raw = m.group("prob").rstrip("%")
        try:
            prob = float(prob_raw) / 100 if prob_raw else None
        except ValueError:
            prob = None
        try:
            target = float(m.group("target").replace(",", ""))
        except ValueError:
            target = None
        scenarios.append(Scenario(name=m.group("name"), probability=prob, target=target))

    stop_match = _HARD_STOP.search(decision)
    hard_stop = float(stop_match.group("stop").replace(",", "")) if stop_match else None

    return {
        "ticker": ticker,
        "research_date": trade_date,
        "reference_price": reference_price,
        "rating": rating,
        "ev": ev,
        "scenarios": scenarios,
        "hard_stop": hard_stop,
    }


def fetch_history(ticker: str, start: str, end: date | None = None):
    """Return list of (date, open, high, low, close, volume) tuples or []."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        # Pull a window starting one day before to ensure we have the start
        start_dt = datetime.strptime(start, "%Y-%m-%d").date() - timedelta(days=1)
        hist = t.history(start=start_dt.isoformat(), end=(end or date.today() + timedelta(days=1)).isoformat())
    except Exception:
        return []
    rows = []
    for idx, row in hist.iterrows():
        rows.append((
            idx.date(),
            float(row.iloc[0]),
            float(row.iloc[1]),
            float(row.iloc[2]),
            float(row.iloc[3]),
            float(row.iloc[4]),
        ))
    return rows


def classify_scenario_bucket(
    realized_pct: float, scenarios: list[Scenario], reference_price: float
) -> str:
    """Map realized return to which scenario target it's tracking toward.

    Returns labels like "ON_TRACK_BULL" / "ON_TRACK_BASE" / "ON_TRACK_BEAR" /
    "BEYOND_BULL" / "BEYOND_BEAR" / "EARLY". Uses scenario targets converted
    to expected-return % from reference.
    """
    if not scenarios or not reference_price:
        return "UNKNOWN"

    # Expected returns per scenario (pct)
    scen_returns: dict[str, float] = {}
    for s in scenarios:
        if s.target is None:
            continue
        scen_returns[s.name] = (s.target - reference_price) / reference_price * 100

    bull = scen_returns.get("Bull")
    base = scen_returns.get("Base", 0)
    bear = scen_returns.get("Bear")
    tail = scen_returns.get("Tail")

    if bull is not None and realized_pct >= bull:
        return "BEYOND_BULL_TARGET"
    if bear is not None and realized_pct <= bear:
        return "BEYOND_BEAR_TARGET"
    if tail is not None and realized_pct <= tail:
        return "INTO_TAIL"

    # Distance to each target
    midpoint_bull_base = ((bull or 0) + base) / 2 if bull is not None else base
    midpoint_base_bear = (base + (bear or 0)) / 2 if bear is not None else base

    if realized_pct >= midpoint_bull_base:
        return "TRACKING_BULL"
    if realized_pct >= base:
        return "TRACKING_BASE_HIGH"
    if realized_pct >= midpoint_base_bear:
        return "TRACKING_BASE_LOW"
    return "TRACKING_BEAR"


def detect_crossings(rows: list, scenarios: list[Scenario], hard_stop: float | None) -> tuple[list[TriggerHit], bool]:
    """Walk daily bars from earliest to latest; emit a TriggerHit when any
    Bull/Base/Bear target is FIRST crossed (above for Bull/Base, below for
    Bear). Also flags hard-stop breach."""
    crossings: list[TriggerHit] = []
    stop_breached = False

    bull = next((s.target for s in scenarios if s.name == "Bull" and s.target), None)
    base = next((s.target for s in scenarios if s.name == "Base" and s.target), None)
    bear = next((s.target for s in scenarios if s.name == "Bear" and s.target), None)

    crossed_bull = False
    crossed_base = False
    crossed_bear = False

    for d, o, h, low, c, v in rows:
        if bull and not crossed_bull and c >= bull:
            crossings.append(TriggerHit("above_bull", bull, d.isoformat(), h, low, c))
            crossed_bull = True
        if base and not crossed_base and c >= base:
            crossings.append(TriggerHit("above_base", base, d.isoformat(), h, low, c))
            crossed_base = True
        if bear and not crossed_bear and c <= bear:
            crossings.append(TriggerHit("below_bear", bear, d.isoformat(), h, low, c))
            crossed_bear = True
        if hard_stop and c < hard_stop:
            crossings.append(TriggerHit("stop_breached", hard_stop, d.isoformat(), h, low, c))
            stop_breached = True
            break

    return crossings, stop_breached


def compute_followup(run_dir: Path, spy_ret_by_date: dict | None = None) -> FollowupResult | None:
    parsed = parse_research(run_dir)
    if parsed is None:
        return None

    rows = fetch_history(parsed["ticker"], parsed["research_date"])
    if not rows:
        return None

    # First row should be on or after research_date
    research_date = datetime.strptime(parsed["research_date"], "%Y-%m-%d").date()
    latest = rows[-1]
    latest_date = latest[0]
    latest_close = latest[4]

    realized_pct = (latest_close - parsed["reference_price"]) / parsed["reference_price"] * 100
    days_elapsed = (latest_date - research_date).days

    spy_ret_pct = None
    alpha_pct = None
    if spy_ret_by_date is not None:
        spy_ret_pct = spy_ret_by_date.get(latest_date)
        if spy_ret_pct is not None:
            alpha_pct = realized_pct - spy_ret_pct

    bucket = classify_scenario_bucket(realized_pct, parsed["scenarios"], parsed["reference_price"])
    crossings, stop_breached = detect_crossings(rows, parsed["scenarios"], parsed["hard_stop"])

    return FollowupResult(
        ticker=parsed["ticker"],
        research_date=parsed["research_date"],
        rating=parsed["rating"],
        reference_price=parsed["reference_price"],
        latest_date=latest_date.isoformat(),
        latest_close=latest_close,
        days_elapsed=days_elapsed,
        realized_return_pct=realized_pct,
        spy_return_pct=spy_ret_pct,
        alpha_pct=alpha_pct,
        scenarios=parsed["scenarios"],
        ev=parsed["ev"],
        scenario_bucket=bucket,
        crossings=crossings,
        hard_stop=parsed["hard_stop"],
        stop_breached=stop_breached,
    )


def build_spy_return_index(start: str) -> dict:
    """SPY-relative returns by date for alpha attribution. Uses earliest
    research_date across all runs as the start. Caches in-memory only."""
    rows = fetch_history("SPY", start)
    if not rows:
        return {}
    anchor = rows[0][4]
    return {d: (c - anchor) / anchor * 100 for (d, o, h, l, c, v) in rows}


def find_runs(base: Path = _RESEARCH_DIR) -> list[Path]:
    """List all current research run dirs (skip *.run-* archives + archive folder)."""
    if not base.exists():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if ".run-" in p.name or p.name.startswith("archive"):
            continue
        if (p / "state.json").exists():
            out.append(p)
    return out


def format_digest(results: list[FollowupResult]) -> str:
    """Markdown digest, one line per ticker, sorted by alpha descending."""
    if not results:
        return "_(no research runs to follow up)_"

    # Sort: ones with fired triggers first, then by alpha (or realized) descending
    def sort_key(r: FollowupResult):
        priority = -1 if r.crossings else 0
        return (priority, -(r.alpha_pct if r.alpha_pct is not None else r.realized_return_pct))

    results = sorted(results, key=sort_key)

    lines = [
        f"# Daily research follow-up — {date.today().isoformat()}",
        "",
        f"{len(results)} current research runs · sorted by trigger fires then alpha",
        "",
        "| Ticker | Days | Ref → Spot | Realized | Alpha vs SPY | Bucket | Rating | Fired |",
        "|---|--:|---|--:|--:|---|---|---|",
    ]

    for r in results:
        alpha_str = f"{r.alpha_pct:+.2f}%" if r.alpha_pct is not None else "—"
        fired_str = ""
        if r.crossings:
            tags = []
            for c in r.crossings:
                tag = c.direction.replace("_", " ")
                tags.append(f"{tag}@${c.level:.2f} ({c.date_crossed})")
            fired_str = "<br>".join(tags)
        elif r.days_elapsed < 1:
            fired_str = "fresh"
        else:
            fired_str = "—"
        lines.append(
            f"| **{r.ticker}** | {r.days_elapsed} | "
            f"${r.reference_price:.2f} → ${r.latest_close:.2f} | "
            f"{r.realized_return_pct:+.2f}% | {alpha_str} | "
            f"{r.scenario_bucket.replace('_', ' ').lower()} | {r.rating.lower()} | "
            f"{fired_str} |"
        )

    # Summary
    fired = [r for r in results if r.crossings]
    stop_breached = [r for r in results if r.stop_breached]
    on_track_bull = [r for r in results if r.scenario_bucket in ("TRACKING_BULL", "BEYOND_BULL_TARGET")]
    on_track_bear = [r for r in results if r.scenario_bucket in ("TRACKING_BEAR", "BEYOND_BEAR_TARGET", "INTO_TAIL")]

    lines += [
        "",
        f"**Crossings:** {len(fired)} tickers had at least one scenario target cross since their research date.",
        f"**Hard-stop breaches:** {len(stop_breached)}.",
        f"**Tracking bull:** {len(on_track_bull)} · **tracking bear:** {len(on_track_bear)}.",
    ]
    return "\n".join(lines)


def send_to_telegram(digest: str, chat_id: str = "-1003753140043") -> bool:
    """Send the digest as a Telegram message. Uses the same auto-discovery
    as the research delivery path."""
    try:
        from cli.research import _auto_discover_telegram_from_openclaw
        import urllib.request
        import urllib.parse
        bot_token, _ = _auto_discover_telegram_from_openclaw()
        if not bot_token:
            print("telegram: no bot token discovered; skipping send", file=sys.stderr)
            return False
        # Telegram has a 4096-char limit per message; chunk if needed
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        for chunk in _chunk_text(digest, 3800):
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": "true",
            }).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10).read()
        return True
    except Exception as exc:
        print(f"telegram send failed: {exc}", file=sys.stderr)
        return False


def _chunk_text(text: str, max_len: int):
    while text:
        if len(text) <= max_len:
            yield text
            return
        split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len
        yield text[:split]
        text = text[split:].lstrip("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="daily_followup")
    parser.add_argument("--telegram", action="store_true",
                        help="also send the digest to the production Telegram chat")
    parser.add_argument("--root", type=str, default=None,
                        help="alternate research directory (default: ~/.openclaw/data/research)")
    args = parser.parse_args(argv)

    base = Path(args.root) if args.root else _RESEARCH_DIR
    runs = find_runs(base)
    if not runs:
        print("no current research runs found", file=sys.stderr)
        return 1

    # Earliest research date for SPY anchor
    dates = []
    for p in runs:
        try:
            s = json.loads((p / "state.json").read_text(encoding="utf-8"))
            d = s.get("trade_date")
            if d:
                dates.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    earliest = min(dates) if dates else None

    spy_idx = build_spy_return_index(earliest) if earliest else {}

    results: list[FollowupResult] = []
    for p in runs:
        r = compute_followup(p, spy_idx)
        if r is not None:
            results.append(r)

    digest = format_digest(results)
    print(digest)

    if args.telegram:
        send_to_telegram(digest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
