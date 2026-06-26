from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

GOG = "/opt/homebrew/bin/gog"


def _run_default(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def gog_token_valid(account: str = "trueknotsg@gmail.com", run=_run_default) -> bool:
    proc = run([GOG, "auth", "list"])
    out = f"{getattr(proc, 'stdout', '')}{getattr(proc, 'stderr', '')}"
    if "invalid_grant" in out:
        return False
    return account in out and getattr(proc, "returncode", 1) == 0


def _read_manifest(manifest: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if "\t" in line:
                t, i = line.split("\t", 1)
                rows[t.strip()] = i.strip()
    return rows


def _write_manifest(manifest: Path, rows: dict[str, str]) -> None:
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    tmp.write_text("".join(f"{t}\t{i}\n" for t, i in rows.items()))
    os.replace(tmp, manifest)


def publish_pdf(ticker: str, pdf: Path, manifest: Path, *, parent: str,
                account: str, run=_run_default) -> str:
    """Idempotent: replace by known file ID (never name-search). Uploads the new
    PDF and records it BEFORE trashing the old one, so a failed upload leaves the
    old file live and the manifest unchanged (no orphaned/lost state)."""
    rows = _read_manifest(manifest)
    old = rows.get(ticker)
    proc = run([GOG, "drive", "upload", str(pdf), "--parent", parent,
                "-a", account, "-j"])
    file_id = json.loads(proc.stdout)["file"]["id"]
    rows[ticker] = file_id
    _write_manifest(manifest, rows)
    if old and old != file_id:
        run([GOG, "drive", "trash", old, "-a", account])
    return file_id


def promote(run_dir: Path, final_base: Path, week: str) -> Path:
    """Move a graded-A run into final/<week>/. Only destructive step. Refuses to
    move when the destination already exists (avoids shutil.move nesting / a wrong
    returned path on a re-promote)."""
    dest_dir = Path(final_base) / week
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(run_dir).name
    if dest.exists():
        raise FileExistsError(f"already promoted: {dest}")
    shutil.move(str(run_dir), str(dest))
    return dest


def refresh_summary_sheet(*, python: str, script: str, account: str,
                          run=_run_default) -> bool:
    proc = run([python, script, "-a", account])
    return getattr(proc, "returncode", 1) == 0


def refresh_trading_plan(*, script: str, run=_run_default) -> bool:
    """Re-render the macro Trading Plan sheet by running the self-contained
    refresh script (it pulls FRED key + gog keyring pw from the macrodaily plist
    and runs `tradingmacro` against the final/ tree). Returns True on success."""
    proc = run([script])
    return getattr(proc, "returncode", 1) == 0
