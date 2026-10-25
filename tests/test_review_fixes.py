import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import data_sequencer
from src.preprocessor import add_technical_indicators
from scripts.update_csv_dates import normalize_csv_dates

ROOT = Path(__file__).resolve().parents[1]


class ReviewFixTests(unittest.TestCase):
    def test_forecast_task_contract_is_explicit_and_shared(self):
        from config import (
            FORECAST_BASELINE, FORECAST_CUTOFF, FORECAST_EVALUATION_METRICS,
            FORECAST_TARGET_COLUMN, FORECAST_TARGET_KIND, HORIZONS, HORIZON_LABELS,
            HORIZON_CHOICES_DESCRIPTION,
            horizon_description,
        )

        self.assertEqual(HORIZONS, [1, 5, 22, 88])
        self.assertEqual(HORIZON_LABELS, {1: "Day", 5: "Week", 22: "Month", 88: "Quarter"})
        self.assertEqual(horizon_description(1), "Day (1 trading session)")
        self.assertEqual(horizon_description(88), "Quarter (88 trading sessions)")
        self.assertEqual(HORIZON_CHOICES_DESCRIPTION, "Day (1 trading session), Week (5 trading sessions), Month (22 trading sessions), Quarter (88 trading sessions)")
        self.assertEqual(FORECAST_TARGET_COLUMN, "Close")
        self.assertEqual(FORECAST_TARGET_KIND, "future_close")
        self.assertEqual(FORECAST_CUTOFF, "after_close")
        self.assertEqual(FORECAST_BASELINE, "persistence")
        self.assertEqual(
            FORECAST_EVALUATION_METRICS,
            ("mae", "rmse", "mape", "lstm_directional_accuracy"),
        )

    def test_targets_do_not_overlap_for_any_horizon(self):
        data = np.arange(1000, dtype=float).reshape(-1, 1)
        boundary = data_sequencer.chronological_split_index(len(data))
        for horizon in (1, 5, 22, 88):
            with self.subTest(horizon=horizon):
                x_train, y_train, x_test, y_test = data_sequencer.split_sequences(
                    data, 60, horizon, 0, boundary)
                self.assertLess(y_train.max(), boundary)
                self.assertEqual(y_test.min(), boundary)
                self.assertEqual(x_test[0, -1, 0], boundary - 1)
                self.assertGreater(len(x_train), 0)

    def test_scaler_never_sees_test_extremes(self):
        import joblib
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = np.arange(1000, dtype=float)
            rows[900:] = 100000
            pd.DataFrame({"Close": rows}).to_csv(directory / "TEST_processed.csv")
            with patch.object(data_sequencer, "PROCESSED_DIR", str(directory)), \
                 patch.object(data_sequencer, "TRAIN_DIR", str(directory)):
                data_sequencer.process_and_save_all()
            scaler = joblib.load(directory / "scaler_TEST.pkl")
            self.assertLess(scaler.data_max_[0], 100000)
            self.assertGreater(np.load(directory / "y_test_TEST.npy").max(), 1)

    def test_indicators_are_replaced_and_react_to_new_price(self):
        closes = 100 + np.arange(400) * .1 + np.sin(np.arange(400))
        raw = pd.DataFrame({"Close": closes})
        processed = add_technical_indicators(raw)
        processed["MACD_12_26_9"] = -999
        appended = pd.concat([processed, processed.tail(1)], ignore_index=True)
        appended.loc[400, "Close"] = 200
        result = add_technical_indicators(appended)
        expected = add_technical_indicators(appended[["Close"]])
        self.assertFalse(result.columns.duplicated().any())
        pd.testing.assert_frame_equal(result, expected)
        self.assertNotEqual(result["EMA_200"].iloc[-1], result["EMA_200"].iloc[-2])
        self.assertTrue(np.isfinite(result.tail(60).to_numpy()).all())

    def test_date_normalization_preserves_history_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("Date,Close\n2020-01-01,10\n2020-01-02,20\n")
            normalize_csv_dates(path)
            self.assertEqual(pd.read_csv(path)["Date"].tolist(), ["2020-01-01", "2020-01-02"])
            path.write_text("Date,Close\n2020-01-01,10\n2020-01-01,20\n")
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                normalize_csv_dates(path)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_model_shows_warning(self):
        from tabs import lstm_model
        from config import HORIZON_LABELS, HORIZONS
        st = MagicMock()
        st.selectbox.return_value = 1
        with patch.object(lstm_model, "st", st), \
             patch.object(lstm_model, "resolve_artifact", side_effect=FileNotFoundError):
            lstm_model.render_lstm_tab(None, "TEST", None, None)
        st.selectbox.assert_called_once_with(
            "Forecast Period", HORIZONS, format_func=HORIZON_LABELS.__getitem__
        )
        st.radio.assert_not_called()
        st.warning.assert_any_call("LSTM Brain offline. Model files not found.")

    def test_forecast_chart_splits_history_and_direct_path_at_last_close(self):
        import plotly.graph_objects as go
        from tabs.lstm_model import _forecast_chart

        frame = pd.DataFrame({"Close": [100., 101., 102., 103.]})
        result = {
            "prices": np.array([104., 106.]),
            "lower": np.array([102., 103.]),
            "upper": np.array([106., 109.]),
        }
        chart = _forecast_chart(frame, result, 3, "Week")

        assert isinstance(chart, go.Figure)
        assert list(chart.data[0].x) == [0.0, 25.0, 50.0]
        assert list(chart.data[0].y) == [101.0, 102.0, 103.0]
        assert list(chart.data[1].x) == [50.0, 75.0, 100.0]
        assert list(chart.data[1].y) == [103.0, 104.0, 106.0]
        assert chart.layout.xaxis.range == (0, 100)
        assert any(shape.type == "line" and shape.x0 == 50 for shape in chart.layout.shapes)

    def test_training_failures_are_reported(self):
        from src import train_model
        with patch.object(train_model.glob, "glob", return_value=["BROKEN_processed.csv"]), \
             patch("src.training.train_ticker", side_effect=ValueError("bad data")), \
             self.assertLogs(train_model.__name__, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "BROKEN"):
                train_model.train_all()


if __name__ == "__main__":
    unittest.main()
