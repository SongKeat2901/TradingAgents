"""Opt-in raw-artifact reuse for cheap reruns (rerun-reduction Phase A).

On a rerun with reuse enabled, load a prior attempt's raw/<file>.json from the
same run dir instead of re-fetching from yfinance. Only the raw FETCH outputs
are reused; the deterministic blocks are always recomputed by the caller.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def reuse_or_fetch(raw_dir, filename: str, fetch_fn: Callable[[], Any],
                   reuse: bool, sanity: Callable[[Any], bool] | None = None):
    """Return (data, reused). Load raw_dir/filename when reuse is on and it
    exists, parses, and passes sanity(); otherwise call fetch_fn()."""
    if reuse:
        p = Path(raw_dir) / filename
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if data is not None and (sanity is None or sanity(data)):
                return data, True
    return fetch_fn(), False


def reuse_or_fetch_peers(raw_dir, peers, fetch_all_fn: Callable[[], Any], reuse: bool):
    """peers.json is a dict keyed by peer symbol; reuse only if the key set
    exactly matches the current peer list (else the peer set changed → refetch)."""
    if reuse:
        p = Path(raw_dir) / "peers.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict) and set(data.keys()) == set(peers):
                return data, True
    return fetch_all_fn(), False
