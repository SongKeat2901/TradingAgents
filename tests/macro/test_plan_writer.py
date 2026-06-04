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
    payload = plan_writer.build_payload(_regime(), [_bias()],
                                        pdf_links={"AAPL": "http://x/AAPL.pdf"})
    assert payload["regime"]["gate"] == "GO"
    assert payload["regime"]["quadrant"] == "Goldilocks"
    assert any(p["name"] == "growth" for p in payload["pillars"])
    row = payload["rows"][0]
    assert row["ticker"] == "AAPL"
    assert row["adjusted_ev_pct"] == 0.15
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
    payload = plan_writer.build_payload(_regime(), [_bias()],
                                        pdf_links={"AAPL": "http://x/AAPL.pdf"})
    grid = plan_writer.to_grid(payload)
    assert len(grid) == SHEET_MAX_ROWS            # constant height → overwrite covers prior runs
    header = grid[4]
    assert header[0] == "Ticker" and header[-1] == "Research"
    data_row = grid[5]
    assert data_row[0] == "AAPL"
    assert data_row[6] == "+15.0%"                # adjusted_ev_pct 0.15 formatted
    assert grid[-1] == [""] * 10                  # trailing padding row
    assert all(len(row) == 10 for row in grid)   # fully rectangular
