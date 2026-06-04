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
    from tradingagents.macro.config import SHEET_MAX_ROWS
    payload = plan_writer.build_payload(
        _regime(), [_bias()], pdf_links={"AAPL": "http://x/AAPL.pdf"},
        levels={"AAPL": {"intrinsic_fv": 280.0, "mos_pct": 0.12,
                         "bear": 180.0, "target": 300.0, "bull": 340.0,
                         "hard_stop": 170.0}})
    grid = plan_writer.to_grid(payload)
    assert len(grid) == SHEET_MAX_ROWS
    assert all(len(row) == 17 for row in grid)        # rectangular, 17 cols
    header = grid[4]
    assert header[0] == "Ticker" and header[-1] == "Research"
    assert header[10] == "Intrinsic FV" and header[11] == "Margin of Safety %"
    data_row = grid[5]
    assert data_row[0] == "AAPL"
    assert data_row[6] == "+15.0%"                    # adjusted_ev_pct
    assert data_row[9] == '=GOOGLEFINANCE("AAPL","price")'  # live px, no fallback (errors visibly)
    assert data_row[10] == "$280.00"                  # intrinsic fair value
    assert data_row[11] == "+12.0%"                   # margin of safety
    assert data_row[15] == "$170.00"                  # hard_stop (shifted)
    assert grid[-1] == [""] * 17


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
