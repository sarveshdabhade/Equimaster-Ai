import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_pooled_return_shrinkage_is_applied_once(monkeypatch):
    import src.forecast as forecast
    import src.panel_training as panel
    artifact = {"metadata": {"features": ["feature"], "window_size": 2,
                "forecast_contract": {"target_kind": "log_return_path"}},
                "pooled": True, "model": "unused", "scaler": "unused",
                "evaluation": {"target_mean": [0.0], "target_scale": [1.0], "return_shrinkage": .5}}
    monkeypatch.setattr(forecast, "resolve_artifact", lambda *a, **kw: artifact)
    monkeypatch.setattr(forecast, "build_lstm_features", lambda frame: pd.DataFrame({"feature": [0., 0.]}))
    monkeypatch.setattr(forecast.joblib, "load", lambda path: SimpleNamespace(transform=lambda values: values))

    class Model:
        def load_weights(self, path):
            pass

        def __call__(self, values, training=False):
            return np.array([[math.log(1.21)]])

    monkeypatch.setattr(panel, "build_panel_inference_model", lambda horizon: Model())
    result = forecast.predict_forecast(pd.DataFrame({"Close": [100., 100.]}), "TEST.NS", 1)
    assert result["prices"][0] == pytest.approx(110.)


def test_evaluation_reports_terminal_interval_coverage(tmp_path, monkeypatch):
    from scripts import evaluate_lstm
    pd.DataFrame({"current_close": [100., 100., 100.], "predicted_close": [100., 100., 100.],
                  "actual_close": [99., 101., 110.]}).to_csv(tmp_path / "backtest_h5.csv", index=False)
    artifact = {"legacy": False, "model": tmp_path / "h5.weights.h5", "metadata": {"version": "test"},
                "evaluation": {"interval": {"lower_residual": [-100., -100., -100., -100., -2.],
                                            "upper_residual": [100., 100., 100., 100., 2.], "test_coverage": .9}}}
    monkeypatch.setattr(evaluate_lstm, "resolve_artifact", lambda *a: artifact)
    result = evaluate_lstm.evaluate_lstm("TEST.NS", 5)
    assert result["interval_test_coverage"] == pytest.approx(2 / 3)
    assert result["interval_path_test_coverage"] == .9
