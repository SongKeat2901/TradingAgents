import json
import pytest

pytestmark = pytest.mark.unit


def _write_anchor(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "forward_probabilities.json").write_text(json.dumps({
        "scenarios": {
            "bull": {"target": 120.0, "probability": 0.30},
            "base": {"target": 103.0, "probability": 0.50},
            "bear": {"target": 90.0,  "probability": 0.20},
        }
    }), encoding="utf-8")
    return tmp_path


def test_passes_when_decision_matches_anchor(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_anchor(tmp_path)
    decision = ("| Bull | 30% | $120.00 |\n| Base | 50% | $103.00 |\n"
                "| Bear | 20% | $90.00 |")
    assert validate_scenario_probabilities(decision, tmp_path) == []


def test_flags_probability_drift_from_anchor(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_anchor(tmp_path)
    decision = ("| Bull | 55% | $120.00 |\n| Base | 30% | $103.00 |\n"
                "| Bear | 15% | $90.00 |")
    vios = validate_scenario_probabilities(decision, tmp_path)
    assert any(v.severity == "MATERIAL" and v.type == "scenario_probability_drift"
               and v.scenario == "bull" for v in vios)


def test_flags_target_drift_from_anchor(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_anchor(tmp_path)
    decision = ("| Bull | 30% | $135.00 |\n| Base | 50% | $103.00 |\n"
                "| Bear | 20% | $90.00 |")
    vios = validate_scenario_probabilities(decision, tmp_path)
    assert any(v.type == "scenario_target_drift" and v.scenario == "bull" for v in vios)


def test_skips_when_anchor_missing(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    decision = "| Bull | 30% | $120.00 |"
    assert validate_scenario_probabilities(decision, tmp_path) == []


def test_accepts_dollar_before_percent_column_order(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_anchor(tmp_path)
    decision = ("| Bull | $120.00 | 30% |\n| Base | $103.00 | 50% |\n"
                "| Bear | $90.00 | 20% |")
    assert validate_scenario_probabilities(decision, tmp_path) == []
