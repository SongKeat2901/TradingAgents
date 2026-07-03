import json
import pytest
from pathlib import Path
from tradingagents.cadence import publish

pytestmark = pytest.mark.unit


class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_token_valid_true_when_grep_hits():
    runner = lambda args, **kw: FakeProc(out="shianpin@trueknot.sg  valid")
    assert publish.gog_token_valid(run=runner) is True


def test_token_valid_false_on_invalid_grant():
    runner = lambda args, **kw: FakeProc(rc=1, err='oauth2: "invalid_grant"')
    assert publish.gog_token_valid(run=runner) is False


def test_publish_pdf_appends_when_absent(tmp_path):
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("AAA\tID_A\n")
    calls = []
    def runner(args, **kw):
        calls.append(args)
        return FakeProc(out=json.dumps({"file": {"id": "NEW_ID"}}))
    fid = publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                              parent="PARENT", account="acct", run=runner)
    assert fid == "NEW_ID"
    rows = dict(l.split("\t") for l in manifest.read_text().splitlines())
    assert rows["BBB"] == "NEW_ID"
    assert rows["AAA"] == "ID_A"


def test_publish_pdf_replaces_when_present(tmp_path):
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("BBB\tOLD_ID\n")
    seen = []
    def runner(args, **kw):
        seen.append(args)
        return FakeProc(out=json.dumps({"file": {"id": "REPL_ID"}}))
    fid = publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                              parent="PARENT", account="acct", run=runner)
    assert fid == "REPL_ID"
    # gog v0.31 has no `drive trash`; trash-old must be `drive delete <id> -y`
    assert not any("trash" in a for a in seen)
    deletes = [a for a in seen if "delete" in a]
    assert deletes and "OLD_ID" in deletes[0] and "-y" in deletes[0]
    rows = dict(l.split("\t") for l in manifest.read_text().splitlines())
    assert rows["BBB"] == "REPL_ID"


def test_promote_moves_only_on_call(tmp_path):
    run_dir = tmp_path / "preaudit" / "2026-06-05-BBB"
    (run_dir).mkdir(parents=True)
    (run_dir / "decision.md").write_text("x")
    final_base = tmp_path / "final"
    dest = publish.promote(run_dir, final_base, "wk 24 2026")
    assert dest == final_base / "wk 24 2026" / "2026-06-05-BBB"
    assert (dest / "decision.md").is_file()
    assert not run_dir.exists()


# FIX 1: upload-before-trash
def test_publish_pdf_upload_failure_preserves_old(tmp_path):
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("BBB\tOLD_ID\n")
    seen = []
    def runner(args, **kw):
        seen.append(args)
        if "upload" in args:
            return FakeProc(out="ERROR not json")   # upload fails -> bad JSON
        return FakeProc(out="{}")
    with pytest.raises(Exception):
        publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                            parent="P", account="a", run=runner)
    # old file NOT trashed (gog: `drive delete`), manifest unchanged
    assert not any("delete" in a or "trash" in a for a in seen)
    rows = dict(l.split("\t") for l in manifest.read_text().splitlines())
    assert rows["BBB"] == "OLD_ID"


# FIX 2: promote guard against existing dest
def test_promote_refuses_existing_dest(tmp_path):
    run_dir = tmp_path / "preaudit" / "2026-06-05-BBB"
    run_dir.mkdir(parents=True)
    (run_dir / "decision.md").write_text("x")
    final_base = tmp_path / "final"
    (final_base / "wk 24 2026" / "2026-06-05-BBB").mkdir(parents=True)
    import pytest as _pt
    with _pt.raises(FileExistsError):
        publish.promote(run_dir, final_base, "wk 24 2026")
    assert run_dir.exists()   # source untouched


# FIX 3: full-account token check
def test_token_valid_false_when_rc_nonzero(tmp_path):
    runner = lambda args, **kw: FakeProc(rc=1, out="trueknotsg@gmail.com listed")
    assert publish.gog_token_valid(run=runner) is False
