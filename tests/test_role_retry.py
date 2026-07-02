import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.analysts import fundamentals_roles as fr


def test_missing_header_reported():
    report = "## Peer comparison matrix\n" + "x" * 800
    issues = fr.check_role_output(fr._REQUIRED_FINANCIAL, report)
    assert any("Business-model framing" in i for i in issues)
    assert any("Sanity check" in i for i in issues)
    assert not any("Peer comparison matrix" in i for i in issues)  # present -> not flagged


def test_complete_report_passes():
    report = "".join(h + "\n" for h in fr._REQUIRED_QUALITY) + "y" * 800
    assert fr.check_role_output(fr._REQUIRED_QUALITY, report) == []


def test_short_report_flagged():
    report = "".join(h + "\n" for h in fr._REQUIRED_RISK) + "short"
    issues = fr.check_role_output(fr._REQUIRED_RISK, report)
    assert any("too short" in i.lower() or "length" in i.lower() for i in issues)


def test_format_feedback_lists_issues():
    fb = fr.format_role_feedback(["missing section: ## Foo", "report too short"])
    assert "## Foo" in fb and "too short" in fb


def test_cap_constant():
    assert fr.ROLE_RETRY_CAP == 2
