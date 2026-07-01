import pytest
from tradingagents.agents.analysts import fundamentals_analyst as fa

pytestmark = pytest.mark.unit


def test_insider_json_in_files_list():
    import inspect
    src = inspect.getsource(fa)
    assert '"insider.json"' in src


def test_insider_section_and_citation_mandate():
    assert "Insider transactions" in fa._SYSTEM
    # closing blanket citation mandate explicitly lists insider.json as a source
    assert "reference.json, or insider.json" in fa._SYSTEM


def test_net_debt_restatement_discipline_present():
    s = fa._SYSTEM
    # must instruct restating net debt from the pinned block, not inventing a derived figure
    assert "net debt" in s.lower()
    assert "raw/net_debt.json" in s or "Net debt block" in s or "## Net debt" in s
    assert "do not compute" in s.lower() or "must not" in s.lower() or "verbatim" in s.lower()
    # tightened: assert the distinctive new sentence itself, not just loose keyword hits
    assert "must not compute and cite a novel derived net-debt" in s.lower()


def test_distress_citation_mandated():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    # tightened: assert the distinctive prohibition clause itself, not just a topic mention
    assert "do not compute your own z-score or invent a zone" in fa._SYSTEM.lower()
