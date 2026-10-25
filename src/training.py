"""One training implementation: chronological fit/calibration/test and versioned artifacts."""
from __future__ import annotations
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import json
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from config import (
    FORECAST_BASELINE,
    FORECAST_CUTOFF,
    FORECAST_EVALUATION_METRICS,
    FORECAST_TARGET_COLUMN,
    FORECAST_TARGET_KIND,
    HORIZONS,
    HORIZON_LABELS,
    resolve_app_path,
)
from src.artifacts import atomic_json, model_directory, sha256, canonical_ticker
from src.data_sequencer import chronological_split_index, split_sequences
from src.evaluate_predictions import evaluate_predictions
from src.train_model import build_model

PIPELINE_VERSION = 3


def inverse_close(values, scaler, close_index):
    values = np.asarray(values)
    return values * scaler.data_range_[close_index] + scaler.data_min_[close_index]


def train_ticker(ticker, horizons=None, epochs=10, window_size=60, only_missing=False):
    ticker = canonical_ticker(ticker)
    horizons = HORIZONS if horizons is None else horizons
    source = Path(resolve_app_path("data", "processed", f"{ticker}_processed.csv"))
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    df = pd.read_csv(io.BytesIO(source_bytes), index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Training data has duplicate dates or non-finite features")
    horizons = sorted(set(int(h) for h in horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("Forecast horizons must be positive")
    root = model_directory()
    pointer = root / f"active_{ticker}.json"
    if pointer.exists():
        old = json.loads(pointer.read_text())
        old_meta = json.loads((root / old["bundle"] / "metadata.json").read_text())
        requested = set(horizons)
        horizons = sorted(requested | {int(h) for h in old_meta.get("horizons", {})})
        old_bundle = root / old["bundle"]
        artifacts_intact = (
            (old_bundle / "scaler.pkl").exists()
            and sha256(old_bundle / "scaler.pkl") == old_meta.get("scaler_sha256")
            and (old_bundle / "processed.csv").exists()
            and sha256(old_bundle / "processed.csv") == source_hash
            and all((old_bundle / entry.get("model", "missing")).is_file()
                    and sha256(old_bundle / entry["model"]) == entry.get("model_sha256")
                    for entry in old_meta.get("horizons", {}).values())
        )
        if only_missing and (old_meta.get("pipeline_version") == PIPELINE_VERSION and
            old_meta.get("data_sha256") == source_hash and
            all(str(h) in old_meta["horizons"] for h in horizons) and artifacts_intact):
            return old_meta
    test_start = chronological_split_index(len(df), window_size)
    calibration_start = chronological_split_index(test_start, window_size, 0.85)
    scaler = MinMaxScaler().fit(df.iloc[:calibration_start].to_numpy())
    scaled = scaler.transform(df.to_numpy())
    close_index = df.columns.get_loc(FORECAST_TARGET_COLUMN)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = root / "versions" / ticker / version
    bundle.mkdir(parents=True)
    (bundle / "processed.csv").write_bytes(source_bytes)
    joblib.dump(scaler, bundle / "scaler.pkl")
    metadata = {"pipeline_version": PIPELINE_VERSION, "ticker": ticker, "version": version,
                "architecture": "residual_lstm_last_close", "training_source": "processed.csv",
                "forecast_contract": {
                    "target_column": FORECAST_TARGET_COLUMN,
                    "target_kind": FORECAST_TARGET_KIND,
                    "cutoff": FORECAST_CUTOFF,
                    "horizons": HORIZONS,
                    "baseline": FORECAST_BASELINE,
                    "evaluation_metrics": list(FORECAST_EVALUATION_METRICS),
                },
                "tensorflow_version": tf.__version__, "numpy_version": np.__version__,
                "trained_at": datetime.now(timezone.utc).isoformat(), "features": list(df.columns),
                "data_sha256": source_hash, "scaler_sha256": sha256(bundle / "scaler.pkl"),
                "data_start": str(df.index[0]), "data_end": str(df.index[-1]),
                "fit_end": str(df.index[calibration_start - 1]),
                "calibration_start": str(df.index[calibration_start]),
                "test_start": str(df.index[test_start]), "window_size": window_size,
                "epochs": epochs, "seed": 42, "horizons": {}}
    for horizon in horizons:
        x_fit, y_fit, x_cal, y_cal = split_sequences(
            scaled[:test_start], window_size, horizon, close_index, calibration_start)
        _, _, x_test, y_test = split_sequences(scaled, window_size, horizon, close_index, test_start)
        y_fit = y_fit.reshape(-1, horizon)
        y_cal = y_cal.reshape(-1, horizon)
        y_test = y_test.reshape(-1, horizon)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_model((window_size, len(df.columns)), horizon, close_index=close_index)
        # tf.data uses a bounded threadpool to support constrained Windows hosts.
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        def dataset(x, y=None):
            ds = tf.data.Dataset.from_tensor_slices(x if y is None else (x, y))
            return ds.batch(32).with_options(options)
        history = model.fit(dataset(x_fit, y_fit), validation_data=dataset(x_cal, y_cal),
                            epochs=epochs, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss", patience=3, restore_best_weights=True)])
        # Reserve calibration for uncertainty; early stopping uses it for model selection,
        # so intervals are empirical validation intervals, not guaranteed conformal coverage.
        cal_pred = inverse_close(model.predict(dataset(x_cal), verbose=0), scaler, close_index)
        predictions = inverse_close(model.predict(dataset(x_test), verbose=0), scaler, close_index)
        actual = inverse_close(y_test, scaler, close_index)
        current = inverse_close(x_test[:, -1, close_index], scaler, close_index)
        metrics = evaluate_predictions(current, predictions[:, -1], actual[:, -1])
        baseline = evaluate_predictions(current, current, actual[:, -1])
        residual = inverse_close(y_cal, scaler, close_index) - cal_pred
        lower, upper = np.quantile(residual, [0.05, 0.95], axis=0)
        covered = (actual >= predictions + lower) & (actual <= predictions + upper)
        name = f"h{horizon}.keras"
        model.save(bundle / name)
        metadata["horizons"][str(horizon)] = {
            "model": name, "model_sha256": sha256(bundle / name), "metrics": metrics,
            "baseline_name": FORECAST_BASELINE, "baseline": baseline,
            "mae_improvement_vs_persistence": baseline["mae"] - metrics["mae"],
            "interval": {"nominal_coverage": 0.9, "lower_residual": lower.tolist(),
                         "upper_residual": upper.tolist(), "test_coverage": float(covered.mean()),
                         "calibration_samples": len(x_cal)},
            "fit_samples": len(x_fit), "test_samples": len(x_test),
            "epochs_run": len(history.history["loss"])}
        records = pd.DataFrame({"as_of": df.index[test_start - 1:len(df) - horizon],
                                "target_date": df.index[test_start + horizon - 1:],
                                "current_close": current, "predicted_close": predictions[:, -1],
                                "actual_close": actual[:, -1]})
        records.to_csv(bundle / f"backtest_h{horizon}.csv", index=False)
        print(f"{ticker} {HORIZON_LABELS.get(horizon, f'h{horizon}')} ({horizon} sessions): MAE={metrics['mae']:.2f} baseline={baseline['mae']:.2f} direction={metrics['lstm_directional_accuracy']:.3f}", flush=True)
    atomic_json(bundle / "metadata.json", metadata)
    if sha256(source) != source_hash:
        raise RuntimeError("Processed data changed during training; the new bundle was not activated")
    # A single atomic pointer publishes matching models/scaler/metrics together.
    atomic_json(pointer, {"bundle": str(bundle.relative_to(root))})
    atomic_json(Path(resolve_app_path("data", "evaluation", f"{ticker}.json")), metadata)
    return metadata
