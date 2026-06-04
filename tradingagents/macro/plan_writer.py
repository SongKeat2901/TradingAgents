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
import tempfile
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
    {last_px, bear, target, bull, hard_stop} (all optional)."""
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
            "last_px": lv.get("last_px"),
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


def to_grid(payload: dict) -> list[list]:
    """Flatten the payload into a 2-D cell grid for a full-range overwrite
    (idempotent: same range, replaced in place — no appends, no dupes)."""
    grid: list[list] = []
    r = payload["regime"]
    grid.append(["MACRO REGIME", r["label"], "Gate:", r["gate"],
                 "Score:", r["score"], "Red pillars:", r["red_count"]])
    grid.append(["Pillar"] + [p["name"] for p in payload["pillars"]])
    grid.append(["Status"] + [f'{p["status"]} ({p["score"]:+.2f})'
                              for p in payload["pillars"]])
    grid.append([])
    header = ["Ticker", "Rating", "Macro Driver", "Bias", "Research EV%",
              "Macro Δ%", "Adjusted EV%", "Conviction", "Action",
              "Last Px", "Bear", "Target", "Bull", "Hard Stop", "Research"]
    grid.append(header)
    for row in payload["rows"]:
        grid.append([
            row["ticker"], row["rating"], row["driver"], row["macro_bias"],
            _pct(row["research_ev_pct"]), _pct(row["macro_delta_pct"]),
            _pct(row["adjusted_ev_pct"]), row["conviction"], row["action"],
            _money(row["last_px"]), _money(row["bear"]), _money(row["target"]),
            _money(row["bull"]), _money(row["hard_stop"]), row["pdf_link"],
        ])
    width = len(header)
    grid = [grow + [""] * (width - len(grow)) for grow in grid]
    while len(grid) < SHEET_MAX_ROWS:
        grid.append([""] * width)
    return grid


def _pct(v) -> str:
    return "" if v is None else f"{v*100:+.1f}%"


def _money(v) -> str:
    return "" if v is None else f"${v:,.2f}"


def write_to_sheet(grid: list[list], sheet_id: str, tab: str = "Macro",
                   runner=subprocess.run) -> None:
    """Overwrite the tab's range with `grid` via gog (replace-in-place →
    idempotent). `runner` is injectable for tests. Requires the mini's gog
    auth (7-day token; re-auth per the update-summary skill on invalid_grant).

    Note for the implementer: the exact `gog sheets update` flags must be
    verified against the installed `gog` version on the mini
    (`gog sheets update --help`)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(grid, fh)
        payload_path = fh.name
    try:
        runner(["gog", "sheets", "update", sheet_id, "--tab", tab,
                "--range", "A1", "--values-json", payload_path], check=True)
    finally:
        os.unlink(payload_path)
