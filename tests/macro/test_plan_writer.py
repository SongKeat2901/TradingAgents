import json

import pytest

from tradingagents.macro import plan_writer
from tradingagents.macro.bias import StockBias
from tradingagents.macro.pillars import PillarScore
from tradingagents.macro.regime import Regime

pytestmark = pytest.mark.unit


def _regime():
    ps = [PillarScore("growth", 0.5, "G"), PillarScore("inflation", 0.1, "A")]
    return Regime(score=0.3, label="Risk-On · Goldilocks", quadrant="Goldilocks",
                  gate="GO", pillars=ps, red_count=0)


def _bias(ticker="AAPL"):
    return StockBias(ticker, "BUY", "d_dxy(-1.20)", "G", 0.12, 0.03, 0.15,
                     0.8, "BUY — add/hold")


def test_build_payload_has_regime_board_and_rows():
    payload = plan_writer.build_payload(
        _regime(), [_bias()], pdf_links={"AAPL": "http://x/AAPL.pdf"},
        levels={"AAPL": {"intrinsic_fv": 280.0, "mos_pct": 0.12,
                         "bear": 180.0, "target": 300.0, "bull": 340.0,
                         "hard_stop": 170.0}})
    assert payload["regime"]["gate"] == "GO"
    assert payload["regime"]["quadrant"] == "Goldilocks"
    assert any(p["name"] == "growth" for p in payload["pillars"])
    row = payload["rows"][0]
    assert row["ticker"] == "AAPL"
    assert row["adjusted_ev_pct"] == 0.15
    assert row["hard_stop"] == 170.0
    assert row["intrinsic_fv"] == 280.0 and row["mos_pct"] == 0.12
    assert row["pdf_link"] == "http://x/AAPL.pdf"


def test_build_payload_company_names_populated():
    """company_names dict → row["company"] carries the full name."""
    payload = plan_writer.build_payload(
        _regime(), [_bias("AAPL")], pdf_links={},
        company_names={"AAPL": "Apple Inc."})
    assert payload["rows"][0]["company"] == "Apple Inc."


def test_build_payload_company_names_default_empty():
    """No company_names arg → row["company"] is blank string."""
    payload = plan_writer.build_payload(_regime(), [_bias("AAPL")], pdf_links={})
    assert payload["rows"][0]["company"] == ""


def test_build_payload_company_names_missing_ticker():
    """company_names present but ticker not in it → row["company"] is blank string."""
    payload = plan_writer.build_payload(
        _regime(), [_bias("MSFT")], pdf_links={},
        company_names={"AAPL": "Apple Inc."})
    assert payload["rows"][0]["company"] == ""


def test_rows_sorted_by_adjusted_ev_desc():
    a = StockBias("AAA", "BUY", "", "G", 0.05, 0.0, 0.05, 0.5, "")
    b = StockBias("BBB", "BUY", "", "G", 0.20, 0.0, 0.20, 0.5, "")
    payload = plan_writer.build_payload(_regime(), [a, b], pdf_links={})
    assert [r["ticker"] for r in payload["rows"]] == ["BBB", "AAA"]


def test_load_manifest_parses_tab_separated(tmp_path):
    m = tmp_path / "pdf_ids.tsv"
    m.write_text("AAPL\tfileid_aapl\nMSFT\tfileid_msft\n")
    out = plan_writer.load_manifest(m)
    assert out == {"AAPL": "fileid_aapl", "MSFT": "fileid_msft"}


def test_pdf_links_from_manifest_build_drive_urls(tmp_path):
    m = tmp_path / "pdf_ids.tsv"
    m.write_text("AAPL\tabc123\n")
    links = plan_writer.pdf_links_from_manifest(m)
    assert links["AAPL"] == "https://drive.google.com/file/d/abc123/view"


def test_to_grid_pads_to_constant_height_with_header_and_data():
    # Updated for 18-col layout: "Company" inserted after "Ticker" (col 1).
    from tradingagents.macro.config import SHEET_MAX_ROWS
    payload = plan_writer.build_payload(
        _regime(), [_bias()], pdf_links={"AAPL": "http://x/AAPL.pdf"},
        levels={"AAPL": {"intrinsic_fv": 280.0, "mos_pct": 0.12,
                         "bear": 180.0, "target": 300.0, "bull": 340.0,
                         "hard_stop": 170.0}},
        company_names={"AAPL": "Apple Inc."})
    grid = plan_writer.to_grid(payload)
    assert len(grid) == SHEET_MAX_ROWS
    assert all(len(row) == 18 for row in grid)        # rectangular, 18 cols (Company added)
    header = grid[4]
    assert header[0] == "Ticker" and header[1] == "Company" and header[-1] == "Research"
    assert header[11] == "Intrinsic FV" and header[12] == "Margin of Safety %"
    data_row = grid[5]
    assert data_row[0] == "AAPL"
    assert data_row[1] == "Apple Inc."                # company name at index 1
    assert data_row[7] == 0.15                         # adjusted_ev_pct shifted right by 1
    assert data_row[10] == '=GOOGLEFINANCE("AAPL","price")'  # live px shifted right by 1
    assert data_row[11] == 280.0                       # intrinsic fair value shifted right by 1
    assert data_row[12] == 0.12                        # margin of safety shifted right by 1
    assert data_row[16] == 170.0                       # hard_stop shifted right by 1
    assert grid[-1] == [""] * 18


def test_write_to_sheet_invokes_gog_with_values_json(monkeypatch):
    monkeypatch.setenv("GOG_ACCOUNT", "trueknotsg@gmail.com")
    calls = []
    plan_writer.write_to_sheet([["a", "b"], ["c", "d"]], "SHEET1",
                               runner=lambda cmd, check: calls.append((cmd, check)))
    cmd, check = calls[0]
    assert cmd[:4] == ["gog", "sheets", "update", "SHEET1"]
    assert cmd[4] == "A1"                       # no tab → first sheet
    vj = cmd[cmd.index("--values-json") + 1]
    assert json.loads(vj) == [["a", "b"], ["c", "d"]]
    assert "USER_ENTERED" in cmd
    assert "-a" in cmd and "trueknotsg@gmail.com" in cmd
    assert check is True


def test_write_to_sheet_tab_prefixes_range_and_omits_account(monkeypatch):
    monkeypatch.delenv("GOG_ACCOUNT", raising=False)
    calls = []
    plan_writer.write_to_sheet([["x"]], "S", tab="Macro",
                               runner=lambda cmd, check: calls.append(cmd))
    assert calls[0][4] == "Macro!A1"
    assert "-a" not in calls[0]                 # no account → no -a flag


def test_to_grid_company_column_placement():
    """'Company' is right after 'Ticker' in the header; data row aligns with it."""
    payload = plan_writer.build_payload(
        _regime(), [_bias("MSFT")], pdf_links={},
        company_names={"MSFT": "Microsoft Corporation"})
    grid = plan_writer.to_grid(payload)
    header = grid[4]
    assert header.index("Company") == header.index("Ticker") + 1
    data_row = grid[5]
    assert data_row[1] == "Microsoft Corporation"


def test_to_grid_stamps_generated_at():
    """The regime header row carries a 'Last updated' stamp when generated_at is
    supplied (Trading Plan timestamp); default (no arg) leaves the row unchanged."""
    from tradingagents.macro.plan_writer import to_grid
    payload = {"regime": {"score": 0.1, "label": "Neutral", "quadrant": "Q",
                          "gate": "GO", "red_count": 0}, "pillars": [], "rows": []}
    g = to_grid(payload, "2026-06-19 10:00 SGT")
    assert "Last updated:" in g[0] and "2026-06-19 10:00 SGT" in g[0]
    assert "Last updated:" not in to_grid(payload)[0]
