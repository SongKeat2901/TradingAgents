import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import macro_daily

pytestmark = pytest.mark.unit


def _series(n=400, start=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(start + 0.1 * np.arange(n), index=idx)


def test_run_assembles_payload_and_calls_writer(tmp_path, monkeypatch):
    # one fake report dir
    import json
    d = tmp_path / "2026-06-01-AAPL"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "AAPL", "trade_date": "2026-06-01"}))
    (d / "decision.md").write_text(
        "Reference price: **$100.00**\n**Rating: BUY**\nEV = **$112.00**\n")

    # stub every network boundary
    monkeypatch.setattr(macro_daily.macro_data, "load_all",
                        lambda specs, as_of: {sp.name: _series() for sp in specs})
    monkeypatch.setattr(macro_daily.macro_data, "load_series",
                        lambda spec, as_of: _series())   # factor-source fetches
    monkeypatch.setattr(macro_daily.macro_data, "load_prices",
                        lambda t, as_of, period="2y": _series())
    captured = {}
    monkeypatch.setattr(macro_daily.plan_writer, "write_to_sheet",
                        lambda grid, sheet_id, **kw: captured.update(grid=grid, sheet=sheet_id))

    payload = macro_daily.run(reports_dir=tmp_path, sheet_id="SHEET1",
                              manifest_path=None, as_of="2026-06-02", write=True)
    assert payload["regime"]["gate"] in {"GO", "CAUTION", "STAND_DOWN"}
    assert any(r["ticker"] == "AAPL" for r in payload["rows"])
    assert captured["sheet"] == "SHEET1"


def test_run_no_write_skips_writer(tmp_path, monkeypatch):
    import json
    d = tmp_path / "2026-06-01-AAPL"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "AAPL", "trade_date": "2026-06-01"}))
    (d / "decision.md").write_text(
        "Reference price: **$100.00**\n**Rating: BUY**\nEV = **$112.00**\n")
    monkeypatch.setattr(macro_daily.macro_data, "load_all",
                        lambda specs, as_of: {sp.name: _series() for sp in specs})
    monkeypatch.setattr(macro_daily.macro_data, "load_series",
                        lambda spec, as_of: _series())
    monkeypatch.setattr(macro_daily.macro_data, "load_prices",
                        lambda t, as_of, period="2y": _series())
    called = {"n": 0}
    monkeypatch.setattr(macro_daily.plan_writer, "write_to_sheet",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    payload = macro_daily.run(reports_dir=tmp_path, sheet_id="S", manifest_path=None,
                              as_of="2026-06-02", write=False)
    assert any(r["ticker"] == "AAPL" for r in payload["rows"])   # non-empty run
    assert called["n"] == 0                                        # writer still skipped


def test_run_manifest_scope_filters_to_published(tmp_path, monkeypatch):
    import json
    for tk in ("AAPL", "ZZZZ"):
        d = tmp_path / f"2026-06-02-{tk}"
        d.mkdir()
        (d / "state.json").write_text(json.dumps(
            {"company_of_interest": tk, "trade_date": "2026-06-02"}))
        (d / "decision.md").write_text(
            "Reference price: **$100.00**\n**Rating: BUY**\nEV = **$112.00**\n")
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("AAPL\tfid_aapl\n")   # only AAPL is "published"
    monkeypatch.setattr(macro_daily.macro_data, "load_all",
                        lambda specs, as_of: {sp.name: _series() for sp in specs})
    monkeypatch.setattr(macro_daily.macro_data, "load_series",
                        lambda spec, as_of: _series())
    monkeypatch.setattr(macro_daily.macro_data, "load_prices",
                        lambda t, as_of, period="2y": _series())
    monkeypatch.setattr(macro_daily.plan_writer, "write_to_sheet", lambda *a, **k: None)
    payload = macro_daily.run(reports_dir=tmp_path, sheet_id="S", manifest_path=str(manifest),
                              as_of="2026-06-02", write=False, manifest_scope=True)
    tickers = [r["ticker"] for r in payload["rows"]]
    assert "AAPL" in tickers and "ZZZZ" not in tickers
    aapl = next(r for r in payload["rows"] if r["ticker"] == "AAPL")
    assert "fid_aapl" in aapl["pdf_link"]
