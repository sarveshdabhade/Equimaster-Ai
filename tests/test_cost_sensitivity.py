import pandas as pd
import pytest

from scripts.evaluate_cost_sensitivity import _evaluate_trades, select_non_overlapping_forecasts


def test_forecast_holdings_are_non_overlapping_per_ticker_horizon():
    frame = pd.DataFrame(
        {
            "as_of": ["2026-01-01", "2026-01-02", "2026-01-06"],
            "target_date": ["2026-01-06", "2026-01-07", "2026-01-11"],
            "current_close": [100, 101, 103],
            "predicted_close": [102, 104, 104],
            "actual_close": [101, 102, 105],
        }
    )

    result = select_non_overlapping_forecasts(frame)

    assert result["as_of"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-01", "2026-01-06"]


def test_cost_scenario_filters_forecasts_below_cost_and_nets_round_trip_cost():
    result = _evaluate_trades(
        [
            {"predicted_return": 0.02, "actual_return": 0.01},
            {"predicted_return": 0.001, "actual_return": 0.02},
        ],
        cost_bps=10,
    )

    assert result["opportunities"] == 2
    assert result["trades"] == 1
    assert result["mean_realized_net_return_pct"] == pytest.approx(0.9)
