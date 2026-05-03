"""Helpers for reading the per-run `raw/` data folder.

The Researcher writes deterministic data files to `<output_dir>/raw/`.
All downstream agents read from this folder via these helpers — never
via bind_tools or ad-hoc fetches. Single source of truth, deterministic
data path, easy to inspect and test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional


def raw_dir_for(output_dir: str) -> str:
    """Return the `raw/` subdirectory path under the run's output_dir."""
    return str(Path(output_dir) / "raw")


def load_json(raw_dir: str, filename: str) -> Optional[Any]:
    """Read and parse a JSON file from raw/. Returns None if missing."""
    p = Path(raw_dir) / filename
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_text(raw_dir: str, filename: str) -> Optional[str]:
    """Read a text file from raw/. Returns None if missing."""
    p = Path(raw_dir) / filename
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def format_for_prompt(raw_dir: str, files: Iterable[str]) -> str:
    """Concatenate raw/ files for inclusion in an LLM prompt.

    Each file becomes a section with header `## raw/<filename>` followed
    by its contents. Missing files become `## raw/<filename>\\n_(missing)_`.
    JSON files are pretty-printed for readability.
    """
    parts: list[str] = []
    for fname in files:
        p = Path(raw_dir) / fname
        parts.append(f"## raw/{fname}\n")
        if not p.exists():
            parts.append("_(missing)_\n")
            continue
        if fname.endswith(".json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                parts.append("```json\n")
                parts.append(json.dumps(data, indent=2))
                parts.append("\n```\n")
            except json.JSONDecodeError:
                parts.append(p.read_text(encoding="utf-8"))
        else:
            parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n")
    return "\n".join(parts)
