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
