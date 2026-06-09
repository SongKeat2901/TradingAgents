import json
import pytest
from pathlib import Path
from tradingagents.cadence import publish

pytestmark = pytest.mark.unit


class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_token_valid_true_when_grep_hits():
    runner = lambda args, **kw: FakeProc(out="trueknotsg@gmail.com  valid")
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
        seen.append(args[:3])
        return FakeProc(out=json.dumps({"file": {"id": "REPL_ID"}}))
    fid = publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                              parent="PARENT", account="acct", run=runner)
    assert fid == "REPL_ID"
    assert any("trash" in " ".join(a) for a in seen)
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
