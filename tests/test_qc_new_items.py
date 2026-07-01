"""Tests for QC checklist items 17-18 (Task 8): accounting ratios + relative
valuation multiples citation mandates, enforced by both the independent QC
Agent and the PM's own self-correction checklist."""
import pytest

from tradingagents.agents.managers import qc_agent

pytestmark = pytest.mark.unit


def test_qc_has_18_items_and_new_rules():
    s = qc_agent._SYSTEM
    assert "18-item checklist" in s
    assert "17." in s and "18." in s
    assert "Accounting ratios" in s          # item 17 topic
    assert "Relative valuation multiples" in s or "EV" in s  # item 18 topic


def test_qc_item_17_cites_accounting_ratios_raw_file():
    s = qc_agent._SYSTEM
    assert "raw/accounting_ratios.json" in s


def test_qc_item_18_cites_relative_multiples_raw_file_and_net_debt_tie():
    s = qc_agent._SYSTEM
    assert "raw/relative_multiples.json" in s
    assert "net-debt block" in s
    assert "fabricated peer multiples" in s


def test_pm_qc_checklist_mirrors_new_items():
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST

    assert "15-item checklist" in _QC_CHECKLIST
    assert "raw/accounting_ratios.json" in _QC_CHECKLIST
    assert "raw/relative_multiples.json" in _QC_CHECKLIST
    assert "net-debt block" in _QC_CHECKLIST
