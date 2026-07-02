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


# --- Task 3: node self-check + feedback injection ------------------------------

class _Stub:
    """Returns a fixed body (>=1200c so invoke_with_empty_retry accepts it on the
    first call) and records the HumanMessage content it was handed."""

    def __init__(self, body):
        self.body = body
        self.seen = []

    def invoke(self, msgs):
        self.seen.append(msgs[-1].content)

        class _R:
            content = self.body
        return _R()


def _raw(tmp_path):
    for f in ("pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md"):
        p = tmp_path / f
        p.write_text("{}" if f.endswith(".json") else "# stub", encoding="utf-8")
    return str(tmp_path)


def _base_state(tmp_path, **kw):
    s = {"company_of_interest": "MSFT", "trade_date": "2026-07-01", "raw_dir": _raw(tmp_path)}
    s.update(kw)
    return s


def test_node_fails_check_sets_feedback(tmp_path):
    bad = "## Peer comparison matrix\n" + "x" * 1300  # missing 3 of 4 required headers
    node = fr.create_financial_statement_analyst(_Stub(bad))
    out = node(_base_state(tmp_path))
    assert out["fundamentals_financial_passed"] is False
    assert out["fundamentals_financial_retries"] == 1
    assert "Business-model framing" in out["fundamentals_financial_feedback"]
    assert out["fundamentals_financial_report"] == bad  # partial report kept


def test_node_passes_clears_feedback(tmp_path):
    good = "".join(h + "\n" for h in fr._REQUIRED_FINANCIAL) + "x" * 1300
    node = fr.create_financial_statement_analyst(_Stub(good))
    out = node(_base_state(tmp_path))
    assert out["fundamentals_financial_passed"] is True
    assert out["fundamentals_financial_feedback"] == ""
    assert out["fundamentals_financial_retries"] == 0


def test_node_injects_prior_feedback(tmp_path):
    good = "".join(h + "\n" for h in fr._REQUIRED_FINANCIAL) + "x" * 1300
    stub = _Stub(good)
    node = fr.create_financial_statement_analyst(stub)
    node(_base_state(tmp_path,
                     fundamentals_financial_feedback="- missing required section: ## Peer comparison matrix"))
    assert "Peer comparison matrix" in stub.seen[-1]  # prior feedback reached the prompt


def test_node_increments_from_prior_retries(tmp_path):
    bad = "## Peer comparison matrix\n" + "x" * 1300
    node = fr.create_risk_redflags_analyst(_Stub(bad))
    out = node(_base_state(tmp_path, fundamentals_riskflags_retries=1))
    assert out["fundamentals_riskflags_retries"] == 2  # 1 -> 2
    assert out["fundamentals_riskflags_passed"] is False
