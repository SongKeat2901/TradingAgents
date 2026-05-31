"""Phase 9 P2 — Python-authoritative peer-metric correction.

The deterministic pm_brief "## Peer ratios" block already pins authoritative
per-peer cells, but the LLM still restates them in decision.md /
decision_executive.md with small inflations (the 2026-05-31 audit: GOOGL
forward P/E cited 30.57x vs 29.59x true — a 3.3% lift that slips under the
±5% validator yet fails a verbatim audit). Per this repo's hard-won lesson
(deterministic Python blocks > prompt-only rules), we do not trust the LLM
to copy numbers: after the pipeline writes its outputs, we snap every
*verifiable* peer-metric value to the authoritative `peer_ratios.json` cell,
formatted exactly like `peer_ratios.format_peer_ratios_block`.

This is the OUTPUT-side twin of the deterministic INPUT block. It runs before
PDF generation and before the Phase 7 validators, so both consume corrected
text. Subject-ticker metrics are never touched (validated by the deterministic
block); only confirmed peers with a populated cell and a unit-shape-matching
value are corrected.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tradingagents.validators.peer_metric_validator import (
    _METRIC_EXPECTED_KINDS,
    _VERIFIABLE_METRICS,
    _load_peer_ratios,
    _load_peer_tickers,
    _normalise_metric,
    _parse_value,
    iter_peer_metric_spans,
)

_RATIO_FIELDS = {"ttm_pe", "forward_pe", "nd_ebitda"}
_PCT_FIELDS = {"latest_quarter_op_margin", "latest_quarter_capex_to_revenue"}
_DOLLAR_FIELDS = {"net_debt", "ttm_ebitda"}

# Files the corrector rewrites in a run dir. Order is stable for the log.
_TARGET_FILES = ("decision.md", "decision_executive.md")


@dataclass(frozen=True)
class PeerMetricCorrection:
    file: str
    line_no: int
    ticker: str
    metric: str
    old_value: str
    new_value: str


def _format_authoritative(field: str, actual: float, kind: str, glyph: str) -> str | None:
    """Render `actual` in the canonical pm_brief format for this metric.

    Mirrors peer_ratios.format_peer_ratios_block: ratios → `{:.2f}x`, pct →
    `{:.1f}%`, $ magnitudes → `${:.2f}B` / `${:.0f}M` (unit chosen to match
    what the author cited, via `kind`)."""
    if field in _RATIO_FIELDS:
        return f"{actual:.2f}{glyph}"
    if field in _PCT_FIELDS:
        return f"{actual:.1f}%"
    if field in _DOLLAR_FIELDS:
        if kind == "billions":
            return f"${actual / 1_000_000_000:.2f}B"
        if kind == "millions":
            return f"${actual / 1_000_000:.0f}M"
        # Fallback: pick a sensible magnitude.
        if abs(actual) >= 1_000_000_000:
            return f"${actual / 1_000_000_000:.2f}B"
        return f"${actual / 1_000_000:.0f}M"
    return None


def _claimed_render(field: str, claimed: float, kind: str, glyph: str) -> str | None:
    """Render the *claimed* number in the same canonical format, so we can
    compare at display precision (avoids churn when the author already wrote
    the verbatim value, possibly at coarser precision)."""
    return _format_authoritative(field, claimed, kind, glyph)


def correct_peer_metrics_text(
    text: str,
    peer_ratios: dict,
    peer_tickers: set[str],
    main_ticker: str | None = None,
) -> tuple[str, list[PeerMetricCorrection]]:
    """Snap verifiable peer-metric values in `text` to peer_ratios.json cells.

    Returns (corrected_text, corrections). Only corrects when: the metric maps
    to a peer_ratios.json column, the bound ticker is a peer (not the subject)
    with a populated cell, the cited value parses to the column's expected unit
    shape, and the rendered value actually differs from the authoritative one.
    """
    if not text or not peer_tickers or not peer_ratios:
        return text, []

    edits: list[tuple[int, int, str, PeerMetricCorrection]] = []
    for (ticker, metric_raw, value_raw, line_no, _match, vstart, vend) in (
        iter_peer_metric_spans(text, peer_tickers, main_ticker=main_ticker)
    ):
        field = _VERIFIABLE_METRICS.get(_normalise_metric(metric_raw))
        if not field:
            continue
        cell = peer_ratios.get(ticker)
        if not isinstance(cell, dict) or cell.get("unavailable"):
            continue
        actual = cell.get(field)
        if actual is None:
            continue

        core = value_raw.strip()
        bold = core.startswith("**") and core.endswith("**")
        core = core.strip("*").strip()
        core = re.sub(r"^(?:approximately\s+|[~≈]\s*)+", "", core, flags=re.IGNORECASE)
        claimed_num, kind = _parse_value(core)
        if claimed_num is None or kind == "raw":
            continue
        expected = _METRIC_EXPECTED_KINDS.get(field, set())
        if expected and kind not in expected:
            continue

        glyph = "×" if "×" in core else "x"
        new_core = _format_authoritative(field, float(actual), kind, glyph)
        if new_core is None:
            continue
        claimed_render = _claimed_render(field, claimed_num, kind, glyph)
        if claimed_render == new_core:
            continue  # already verbatim at display precision

        new_full = f"**{new_core}**" if bold else new_core
        edits.append((
            vstart, vend, new_full,
            PeerMetricCorrection(
                file="", line_no=line_no, ticker=ticker, metric=metric_raw,
                old_value=value_raw, new_value=new_core,
            ),
        ))

    if not edits:
        return text, []

    # Apply in descending start offset so earlier spans stay valid.
    edits.sort(key=lambda e: e[0], reverse=True)
    out = text
    corrections: list[PeerMetricCorrection] = []
    for vstart, vend, new_full, corr in edits:
        out = out[:vstart] + new_full + out[vend:]
        corrections.append(corr)
    corrections.reverse()  # restore document order
    return out, corrections


def correct_peer_metrics_in_run(run_dir: str | Path) -> dict:
    """Correct peer metrics across a run dir's decision files in place.

    Reads raw/peer_ratios.json + raw/peers.json + state.json (subject ticker),
    rewrites decision.md / decision_executive.md when values drift, and writes
    raw/peer_corrections.json as an audit log. Returns a summary dict."""
    run = Path(run_dir)
    raw = run / "raw"
    peer_ratios = _load_peer_ratios(raw / "peer_ratios.json")
    peer_tickers = _load_peer_tickers(raw / "peers.json")

    main_ticker = None
    state_path = run / "state.json"
    if state_path.exists():
        try:
            main_ticker = json.loads(state_path.read_text(encoding="utf-8")).get(
                "company_of_interest"
            )
        except (OSError, json.JSONDecodeError):
            main_ticker = None

    all_corrections: list[dict] = []
    files_changed: list[str] = []
    if peer_ratios and peer_tickers:
        for fname in _TARGET_FILES:
            fpath = run / fname
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            corrected, corrections = correct_peer_metrics_text(
                text, peer_ratios, peer_tickers, main_ticker=main_ticker
            )
            if corrections:
                fpath.write_text(corrected, encoding="utf-8")
                files_changed.append(fname)
                for c in corrections:
                    rec = asdict(c)
                    rec["file"] = fname
                    all_corrections.append(rec)

    # Always write the log (even empty) so the run dir records that the
    # corrector ran — an absent log means the step was skipped.
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "peer_corrections.json").write_text(
        json.dumps(
            {
                "main_ticker": main_ticker,
                "files_changed": files_changed,
                "total_corrections": len(all_corrections),
                "corrections": all_corrections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "total_corrections": len(all_corrections),
        "files_changed": files_changed,
    }
