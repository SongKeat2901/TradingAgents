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

# Only ratios and percentages are auto-corrected. The $ magnitudes
# (net_debt / ttm_ebitda) are deliberately EXCLUDED: they were not the
# fabrication vector (the 2026-05-31 audit found P/E inflation, not $-cell
# drift), they carry inline arithmetic + markdown bold that reformatting
# would damage, and unit handling ($B vs $M, sign placement) is finicky.
# Touching them only risked cosmetic regressions on otherwise-clean reports.
_CORRECTABLE_FIELDS = _RATIO_FIELDS | _PCT_FIELDS

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


def _value_matches_other_peer(
    field: str,
    claimed_render: str,
    bound_ticker: str,
    peer_ratios: dict,
    kind: str,
    glyph: str,
) -> bool:
    """True if `claimed_render` exactly matches a DIFFERENT peer's authoritative
    cell for this metric.

    TSM 2026-05-29 anti-fabrication guard: in a "X and Y ... a and b
    respectively" list the span iterator binds the second ticker (Y) to the
    first value (a), which actually belongs to X. If that value already
    matches X's authoritative cell at display precision, the number is
    correctly X's — only the name↔number binding is ambiguous — so snapping
    it to Y's value would overwrite a CORRECT figure and fabricate X's metric.
    Skip the correction (conservative: never corrupt a verbatim peer value)."""
    if claimed_render is None:
        return False
    for other, cell in peer_ratios.items():
        if other == bound_ticker or not isinstance(cell, dict):
            continue
        other_actual = cell.get(field)
        if other_actual is None:
            continue
        if _format_authoritative(field, float(other_actual), kind, glyph) == claimed_render:
            return True
    return False


# Header phrase (normalised) → peer_ratios field, for markdown-table columns.
# Reuses the validator's prose phrase map plus a few table-only header spellings.
_HEADER_FIELD_ALIASES: dict[str, str] = {
    **_VERIFIABLE_METRICS,
    "fwd pe": "forward_pe",
    "p/e fwd": "forward_pe",
    "ttm pe": "ttm_pe",
    "p/e ttm": "ttm_pe",
    "capex/revenue": "latest_quarter_capex_to_revenue",
    "capex / rev": "latest_quarter_capex_to_revenue",
}


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and re.fullmatch(r"\|?[\s:|-]+\|?", s) is not None


def _correct_table_cell(
    cell: str, field: str, actual: float
) -> tuple[str, str] | None:
    """Return (new_cell, new_numeric_literal) if the cell's result value drifts
    from `actual`, else None. Only ratio (Xx) and pct (X%) columns are handled;
    a cell whose unit doesn't match the column's expected shape is left alone."""
    if field in _RATIO_FIELDS and "x" not in cell.lower() and "×" not in cell:
        return None
    if field in _PCT_FIELDS and "%" not in cell:
        return None
    if field not in _RATIO_FIELDS and field not in _PCT_FIELDS:
        return None  # $ columns not auto-corrected in tables (rare, messy)

    # The "result" value is whatever follows the last "=" (cells may show an
    # inline computation "A / B = 36.12%"); otherwise the whole cell.
    if "=" in cell:
        head, _, tail = cell.rpartition("=")
    else:
        head, tail = None, cell
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", tail)
    if not m:
        return None
    old_lit = m.group()
    try:
        claimed = float(old_lit.replace(",", ""))
    except ValueError:
        return None
    if round(claimed, 2) == round(float(actual), 2):
        return None  # already verbatim
    new_lit = f"{float(actual):.2f}"
    new_tail = tail[: m.start()] + new_lit + tail[m.end():]
    new_cell = (head + "=" + new_tail) if head is not None else new_tail
    return new_cell, new_lit


def _correct_markdown_tables(
    text: str, peer_ratios: dict, peer_tickers: set[str]
) -> tuple[str, list[PeerMetricCorrection]]:
    """Snap peer-metric columns in markdown tables to peer_ratios.json cells.

    Detects header+separator+data-row blocks, maps headers to peer_ratios
    fields, finds the ticker column, and corrects each peer row's metric cells.
    """
    lines = text.split("\n")
    corrections: list[PeerMetricCorrection] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip().startswith("|") or i + 1 >= n or not _is_separator_row(lines[i + 1]):
            i += 1
            continue
        headers = _split_table_row(line)
        col_field: dict[int, str] = {}
        ticker_col = None
        for idx, h in enumerate(headers):
            hn = _normalise_metric(h)
            if hn in ("peer", "ticker", "company", "name", "peers"):
                ticker_col = idx
            f = _HEADER_FIELD_ALIASES.get(hn)
            if f:
                col_field[idx] = f
        if not col_field:
            i += 1
            continue
        # Walk data rows until the table ends.
        j = i + 2
        while j < n and lines[j].strip().startswith("|"):
            cells = _split_table_row(lines[j])
            # Resolve the ticker for this row: declared ticker col, else any
            # cell that is a known peer.
            row_ticker = None
            if ticker_col is not None and ticker_col < len(cells):
                cand = re.sub(r"[^A-Za-z]", "", cells[ticker_col]).upper()
                if cand in peer_tickers:
                    row_ticker = cand
            if row_ticker is None:
                for c in cells:
                    cand = re.sub(r"[^A-Za-z]", "", c).upper()
                    if cand in peer_tickers:
                        row_ticker = cand
                        break
            ratios_cell = peer_ratios.get(row_ticker) if row_ticker else None
            if not isinstance(ratios_cell, dict) or ratios_cell.get("unavailable"):
                j += 1
                continue
            changed = False
            for idx, field in col_field.items():
                if idx >= len(cells):
                    continue
                actual = ratios_cell.get(field)
                if actual is None:
                    continue
                res = _correct_table_cell(cells[idx], field, actual)
                if res is None:
                    continue
                new_cell, new_lit = res
                corrections.append(PeerMetricCorrection(
                    file="", line_no=j + 1, ticker=row_ticker, metric=headers[idx],
                    old_value=cells[idx], new_value=new_cell,
                ))
                cells[idx] = new_cell
                changed = True
            if changed:
                indent = line[: len(line) - len(line.lstrip())]
                # Preserve the data row's own indent, not the header's.
                row_indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                lines[j] = row_indent + "| " + " | ".join(cells) + " |"
            j += 1
        i = j
    return "\n".join(lines), corrections


def correct_peer_metrics_text(
    text: str,
    peer_ratios: dict,
    peer_tickers: set[str],
    main_ticker: str | None = None,
) -> tuple[str, list[PeerMetricCorrection]]:
    """Snap verifiable peer-metric values in `text` to peer_ratios.json cells.

    Two passes: markdown-table columns first (where the dense P/E fabrications
    live), then prose "<TICKER> <METRIC> <value>" claims. Only corrects when the
    bound ticker is a peer (not the subject) with a populated cell, the value
    parses to the column's expected unit shape, and the rendered value actually
    differs from the authoritative one.
    """
    if not text or not peer_tickers or not peer_ratios:
        return text, []

    text, table_corrections = _correct_markdown_tables(text, peer_ratios, peer_tickers)

    edits: list[tuple[int, int, str, PeerMetricCorrection]] = []
    for (ticker, metric_raw, value_raw, line_no, _match, vstart, vend) in (
        iter_peer_metric_spans(text, peer_tickers, main_ticker=main_ticker)
    ):
        field = _VERIFIABLE_METRICS.get(_normalise_metric(metric_raw))
        if not field or field not in _CORRECTABLE_FIELDS:
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

        # TSM 2026-05-29: skip when the claimed value is actually a different
        # peer's authoritative figure (mis-bound by a "...a and b respectively"
        # list). Snapping it would fabricate that other peer's metric.
        if _value_matches_other_peer(
            field, claimed_render, ticker, peer_ratios, kind, glyph
        ):
            continue

        new_full = f"**{new_core}**" if bold else new_core
        edits.append((
            vstart, vend, new_full,
            PeerMetricCorrection(
                file="", line_no=line_no, ticker=ticker, metric=metric_raw,
                old_value=value_raw, new_value=new_core,
            ),
        ))

    if not edits:
        return text, table_corrections

    # Apply in descending start offset so earlier spans stay valid.
    edits.sort(key=lambda e: e[0], reverse=True)
    out = text
    prose_corrections: list[PeerMetricCorrection] = []
    for vstart, vend, new_full, corr in edits:
        out = out[:vstart] + new_full + out[vend:]
        prose_corrections.append(corr)
    prose_corrections.reverse()  # restore document order
    return out, table_corrections + prose_corrections


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
