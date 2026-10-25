import pytest

from config import HORIZONS
from src.fusion_gate import assess_fusion_staged_trial


def _group(horizon, fused_mae=8.0, lstm_mae=10.0, persistence_mae=12.0,
           fused_direction=0.7, lstm_direction=0.6):
    return {
        "horizon": horizon,
        "fused_mae": fused_mae,
        "lstm_mae": lstm_mae,
        "persistence_mae": persistence_mae,
        "fused_directional_accuracy": fused_direction,
        "lstm_directional_accuracy": lstm_direction,
    }


def _result():
    return {
        "validation": {
            "macro_average": {
                "fused_mae": 9.0, "lstm_mae": 10.0,
                "fused_directional_accuracy": 0.65, "lstm_directional_accuracy": 0.6,
            },
            "per_ticker_horizon": [_group(horizon) for horizon in HORIZONS],
        },
        "test": {
            "macro_average": {
                "fused_mae": 8.0, "lstm_mae": 10.0, "persistence_mae": 12.0,
                "fused_directional_accuracy": 0.7, "lstm_directional_accuracy": 0.6,
            },
            "per_ticker_horizon": [_group(horizon) for horizon in HORIZONS],
        },
    }


def test_fusion_gate_returns_staged_eligibility_without_activation():
    result = assess_fusion_staged_trial(_result())

    assert result["eligible_for_staged_trial"]
    assert not result["production_activation"]
    assert all(result["checks"].values())


def test_fusion_gate_rejects_missing_horizon_or_test_regression():
    missing_horizon = _result()
    missing_horizon["test"]["per_ticker_horizon"].pop()
    result = assess_fusion_staged_trial(missing_horizon)
    assert not result["eligible_for_staged_trial"]
    assert not result["checks"]["all_horizons_present"]

    regression = _result()
    regression["test"]["per_ticker_horizon"][0]["fused_mae"] = 11.0
    result = assess_fusion_staged_trial(regression)
    assert not result["eligible_for_staged_trial"]
    assert not result["checks"]["no_test_horizon_regression"]


def test_fusion_gate_requires_finite_test_metrics():
    result = _result()
    result["test"]["macro_average"]["fused_mae"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        assess_fusion_staged_trial(result)
