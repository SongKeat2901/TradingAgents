import pytest
from datetime import datetime

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.sec_edgar import _parse_activist_hits, format_activist_block

_CIK10 = "0000789019"  # subject
_DATA = {"hits": {"hits": [
    {"_source": {"ciks": [_CIK10, "0000111"], "file_date": "2024-02-13",
                 "root_forms": ["SC 13G"], "display_names": ["MICROSOFT CORP (CIK 0000789019)", "VANGUARD GROUP INC (CIK 0000102909)"]}},
    {"_source": {"ciks": [_CIK10, "0000222"], "file_date": "2023-06-01",
                 "root_forms": ["SC 13D"], "display_names": ["MICROSOFT CORP (CIK 0000789019)", "Activist LP (CIK 0000999)"]}},
    # mention-only (subject cik absent) -> excluded
    {"_source": {"ciks": ["0000333"], "file_date": "2024-05-01",
                 "root_forms": ["SC 13D"], "display_names": ["OTHER CO", "Filer X"]}},
    # future-dated -> excluded
    {"_source": {"ciks": [_CIK10], "file_date": "2027-01-01",
                 "root_forms": ["SC 13G"], "display_names": ["MICROSOFT CORP", "Future Filer"]}},
    # duplicate of first -> deduped
    {"_source": {"ciks": [_CIK10], "file_date": "2024-02-13",
                 "root_forms": ["SC 13G"], "display_names": ["MICROSOFT CORP (CIK 0000789019)", "VANGUARD GROUP INC (CIK 0000102909)"]}},
]}}


def test_parse_filters_and_classifies():
    td = datetime(2026, 7, 1)
    rows = _parse_activist_hits(_DATA, _CIK10, td, "MICROSOFT CORP", 6)
    # 2 unique kept (mention-only + future excluded, dup collapsed), newest-first
    assert [r["date"] for r in rows] == ["2024-02-13", "2023-06-01"]
    assert rows[0]["activist"] is False  # 13G passive
    assert rows[1]["activist"] is True   # 13D activist
    # subject name stripped from filers
    assert all("MICROSOFT" not in f.upper() for r in rows for f in r["filers"])
    assert any("VANGUARD" in f.upper() for f in rows[0]["filers"])


def test_parse_limit():
    td = datetime(2026, 7, 1)
    assert len(_parse_activist_hits(_DATA, _CIK10, td, "MICROSOFT CORP", 1)) == 1


def test_block_render_and_none():
    td = datetime(2026, 7, 1)
    block = format_activist_block({"ticker": "MSFT", "company": "MICROSOFT CORP",
                                   "filings": _parse_activist_hits(_DATA, _CIK10, td, "MICROSOFT CORP", 6)})
    assert "13D/13G" in block and "activist" in block and "passive" in block
    assert "do NOT infer" in block
    none_block = format_activist_block({"ticker": "X", "filings": []})
    assert "none reported" in none_block.lower()
    na_block = format_activist_block({"unavailable": True, "reason": "unreachable"})
    assert "n/a" in na_block
