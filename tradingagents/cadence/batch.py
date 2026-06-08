from __future__ import annotations

import json
import re
from pathlib import Path

from tradingagents.cadence.models import RunData

_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-([A-Z0-9.]+)$")

_REF_RE = re.compile(r"Reference price:\**\s*\$([0-9][0-9,]*\.?[0-9]*)")


def find_latest_batch(preaudit_base: Path) -> tuple[str | None, list[Path]]:
    """Return (trade_date, [completed run dirs]) for the newest date present.

    A run is 'completed' iff it contains decision.md. Archived dirs whose name
    contains a dot (e.g. '<date>-<T>.pre-cadence') are excluded — the historical
    glob bug that matched them caused false 'batch complete' signals.
    """
    preaudit_base = Path(preaudit_base)
    if not preaudit_base.is_dir():
        return None, []
    by_date: dict[str, list[Path]] = {}
    for d in preaudit_base.iterdir():
        if not d.is_dir() or "." in d.name:
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        if not (d / "decision.md").is_file():
            continue
        by_date.setdefault(m.group(1), []).append(d)
    if not by_date:
        return None, []
    newest = max(by_date)
    return newest, sorted(by_date[newest], key=lambda p: p.name)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _reference_price(decision_md: Path) -> float | None:
    try:
        text = decision_md.read_text()
    except OSError:
        return None
    m = _REF_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    m = _DIR_RE.match(run_dir.name)
    trade_date = m.group(1) if m else ""
    ticker = m.group(2) if m else run_dir.name
    raw = run_dir / "raw"
    return RunData(
        ticker=ticker,
        trade_date=trade_date,
        run_dir=str(run_dir),
        validation=_read_json(run_dir / "validation_report.json"),
        intrinsic_value=_read_json(raw / "intrinsic_value.json"),
        peer_ratios=_read_json(raw / "peer_ratios.json"),
        financials=_read_json(raw / "financials.json"),
        reference_price=_reference_price(run_dir / "decision.md"),
    )
