from __future__ import annotations

import json
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
    return account.split("@")[0] in out and getattr(proc, "returncode", 1) == 0


def _read_manifest(manifest: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if "\t" in line:
                t, i = line.split("\t", 1)
                rows[t.strip()] = i.strip()
    return rows


def _write_manifest(manifest: Path, rows: dict[str, str]) -> None:
    manifest.write_text("".join(f"{t}\t{i}\n" for t, i in rows.items()))


def publish_pdf(ticker: str, pdf: Path, manifest: Path, *, parent: str,
                account: str, run=_run_default) -> str:
    """Idempotent: replace by known file ID (never name-search)."""
    rows = _read_manifest(manifest)
    old = rows.get(ticker)
    if old:
        run([GOG, "drive", "trash", old, "-a", account])
    proc = run([GOG, "drive", "upload", str(pdf), "--parent", parent,
                "-a", account, "-j"])
    file_id = json.loads(proc.stdout)["file"]["id"]
    rows[ticker] = file_id
    _write_manifest(manifest, rows)
    return file_id


def promote(run_dir: Path, final_base: Path, week: str) -> Path:
    """Move a graded-A run into final/<week>/. Only destructive step."""
    dest_dir = Path(final_base) / week
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(run_dir).name
    shutil.move(str(run_dir), str(dest))
    return dest


def refresh_summary_sheet(*, python: str, script: str, account: str,
                          run=_run_default) -> bool:
    proc = run([python, script, "-a", account])
    return getattr(proc, "returncode", 1) == 0
