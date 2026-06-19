"""Build the Trading Plan sheet payload (pure) and write it idempotently.

The payload builder is pure and fully tested. The actual Sheets write goes
through the `gog` CLI on the mini and replaces cells in a known sheet by ID —
never name-search, per the no-duplicates rule. PDF hyperlinks are resolved
from the existing pdf_ids.tsv manifest (ticker<TAB>driveFileId).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .bias import StockBias
from .config import SHEET_MAX_ROWS
from .regime import Regime


def load_manifest(path: Path) -> dict[str, str]:
    """Parse ticker<TAB>fileId rows. Parsed in Python (never `IFS=$"\\t"` /
    `grep -P`, which are broken on macOS)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pdf_ids manifest not found: {p}")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        ticker, file_id = line.split("\t", 1)
        out[ticker.strip()] = file_id.strip()
    return out


def pdf_links_from_manifest(path: Path) -> dict[str, str]:
    return {t: f"https://drive.google.com/file/d/{fid}/view"
            for t, fid in load_manifest(path).items()}


def build_payload(regime: Regime, biases: list[StockBias],
                  pdf_links: dict[str, str], levels: dict | None = None) -> dict:
    """Pure: assemble the regime board + per-ticker rows, rows sorted by
    adjusted EV descending (best-positioned first). `levels` maps ticker ->
    {intrinsic_fv, mos_pct, bear, target, bull, hard_stop} (all optional).
    Last Px is a live GOOGLEFINANCE formula keyed off the ticker, so no static
    price is threaded through here."""
    levels = levels or {}
    rows = []
    for sb in sorted(biases,
                     key=lambda b: (b.adjusted_ev_pct is None, -(b.adjusted_ev_pct or 0))):
        lv = levels.get(sb.ticker, {})
        rows.append({
            "ticker": sb.ticker,
            "rating": sb.rating,
            "driver": sb.driver,
            "macro_bias": sb.macro_bias,
            "research_ev_pct": sb.research_ev_pct,
            "macro_delta_pct": sb.macro_delta_pct,
            "adjusted_ev_pct": sb.adjusted_ev_pct,
            "conviction": sb.conviction,
            "action": sb.action,
            "intrinsic_fv": lv.get("intrinsic_fv"),
            "mos_pct": lv.get("mos_pct"),
            "bear": lv.get("bear"),
            "target": lv.get("target"),
            "bull": lv.get("bull"),
            "hard_stop": lv.get("hard_stop"),
            "pdf_link": pdf_links.get(sb.ticker, ""),
        })
    return {
        "regime": {
            "score": regime.score, "label": regime.label,
            "quadrant": regime.quadrant, "gate": regime.gate,
            "red_count": regime.red_count,
        },
        "pillars": [{"name": p.name, "score": p.score, "status": p.status}
                    for p in regime.pillars],
        "rows": rows,
    }


def to_grid(payload: dict, generated_at: str | None = None) -> list[list]:
    """Flatten the payload into a 2-D cell grid for a full-range overwrite
    (idempotent: same range, replaced in place — no appends, no dupes).

    `generated_at` (when supplied) stamps a "Last updated: <ts>" onto the regime
    header row so the sheet shows when it was last refreshed."""
    grid: list[list] = []
    r = payload["regime"]
    regime_row = ["MACRO REGIME", r["label"], "Gate:", r["gate"],
                  "Score:", r["score"], "Red pillars:", r["red_count"]]
    if generated_at:
        regime_row += ["Last updated:", generated_at]
    grid.append(regime_row)
    grid.append(["Pillar"] + [p["name"] for p in payload["pillars"]])
    grid.append(["Status"] + [f'{p["status"]} ({p["score"]:+.2f})'
                              for p in payload["pillars"]])
    grid.append([])
    header = ["Ticker", "Rating", "Macro Driver", "Bias", "Research EV%",
              "Macro Δ%", "Adjusted EV%", "Conviction", "Action",
              "Last Px", "Intrinsic FV", "Margin of Safety %",
              "Bear", "Target", "Bull", "Hard Stop", "Research"]
    grid.append(header)
    for row in payload["rows"]:
        grid.append([
            row["ticker"], row["rating"], row["driver"], row["macro_bias"],
            _n(row["research_ev_pct"]), _n(row["macro_delta_pct"]),
            _n(row["adjusted_ev_pct"]), _n(row["conviction"]), row["action"],
            _gfinance(row["ticker"]),
            _n(row["intrinsic_fv"]), _n(row["mos_pct"]),
            _n(row["bear"]), _n(row["target"]),
            _n(row["bull"]), _n(row["hard_stop"]), row["pdf_link"],
        ])
    width = len(header)
    grid = [grow + [""] * (width - len(grow)) for grow in grid]
    while len(grid) < SHEET_MAX_ROWS:
        grid.append([""] * width)
    return grid


def _n(v):
    """Raw numeric cell (or "" for None). Numbers are written un-formatted so
    Sheets stores them as true numbers — display % / $ comes from the column
    number-formats (beautify), and the EV +/- conditional colouring keys off the
    numeric sign. (Writing pre-formatted strings like "+9.7%" made Sheets parse
    inconsistently and broke the colour rules.)"""
    return "" if v is None else v


def _gfinance(ticker: str) -> str:
    """Live-updating price cell. Written with USER_ENTERED so Sheets evaluates
    the formula. NO fallback by design: if GOOGLEFINANCE can't resolve the
    symbol the cell shows an explicit error (#N/A) rather than a stale/misleading
    static price. ~20-min delayed during US hours; last close after hours."""
    return f'=GOOGLEFINANCE("{ticker}","price")'


def write_to_sheet(grid: list[list], sheet_id: str, tab: str | None = None,
                   account: str | None = None, runner=subprocess.run) -> None:
    """Overwrite the sheet's range with `grid` via gog (replace-in-place →
    idempotent: the fixed-height grid always covers the prior write's footprint).

    Verified against gog v0.11.0: `gog sheets update <id> <range> --values-json
    '<json 2D array>' --input USER_ENTERED`. `tab` None targets the first sheet
    (range "A1"); pass a name for "Tab!A1". `account` defaults to the GOG_ACCOUNT
    env var; gog reads the keyring password from GOG_KEYRING_PASSWORD itself.
    `runner` is injectable for tests. Needs the mini's gog auth (7-day token;
    re-auth per the update-summary skill on invalid_grant)."""
    rng = f"{tab}!A1" if tab else "A1"
    account = account or os.environ.get("GOG_ACCOUNT")
    cmd = ["gog", "sheets", "update", sheet_id, rng,
           "--values-json", json.dumps(grid), "--input", "USER_ENTERED", "--no-input"]
    if account:
        cmd += ["-a", account]
    runner(cmd, check=True)
