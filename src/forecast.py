"""Forecast inference with a matching model/scaler/feature contract."""
from __future__ import annotations
import joblib
import numpy as np
import tensorflow as tf
from config import FORECAST_TARGET_COLUMN
from src.artifacts import resolve_artifact
from src.preprocessor import add_technical_indicators


def predict_forecast(df, ticker, horizon=1, recursive=False):
    artifact = resolve_artifact(ticker, 1 if recursive else horizon)
    features = artifact["metadata"].get("features", list(df.columns))
    window_size = artifact["metadata"].get("window_size", 60)
    if set(features) != set(df.columns):
        raise ValueError("Data features differ from the model contract; regenerate and retrain")
    frame = df.reindex(columns=features).copy()
    if len(frame) < window_size:
        raise ValueError(f"Need at least {window_size} historical rows")
    scaler = joblib.load(artifact["scaler"])
    model = tf.keras.models.load_model(artifact["model"], compile=False)
    target_column = artifact["metadata"].get("forecast_contract", {}).get(
        "target_column", FORECAST_TARGET_COLUMN
    )
    close = features.index(target_column)
    def infer(data):
        values = data.tail(window_size).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Indicators contain missing values; more history is required")
        scaled = scaler.transform(values)
        predictions = np.asarray(model(scaled[None, ...], training=False)).reshape(-1)
        return predictions * scaler.data_range_[close] + scaler.data_min_[close]
    if recursive:
        values = []
        for step in range(horizon):
            value = float(infer(frame)[0])
            values.append(value)
            if step == horizon - 1:
                break
            row = frame.iloc[-1].copy()
            for column in ("Close", "Adj Close", "Open", "High", "Low"):
                if column in row:
                    row[column] = value
            # Numeric positions avoid inventing future market calendar dates.
            import pandas as pd
            frame = pd.concat([frame.reset_index(drop=True), row.to_frame().T], ignore_index=True)
            frame = add_technical_indicators(frame).reindex(columns=features)
        values = np.asarray(values)
    else:
        values = infer(frame)
        if len(values) != horizon:
            raise ValueError("Model output does not match the requested forecast horizon")
    interval = artifact["evaluation"].get("interval") if not recursive else None
    return {"prices": values, "lower": values + np.asarray(interval["lower_residual"]) if interval else None,
            "upper": values + np.asarray(interval["upper_residual"]) if interval else None,
            "artifact": artifact}
