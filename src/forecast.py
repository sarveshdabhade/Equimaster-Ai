"""Forecast inference with a matching model/scaler/feature contract."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from config import FORECAST_TARGET_COLUMN
from src.artifacts import resolve_artifact
from src.preprocessor import add_technical_indicators
from src.lstm_features import build_lstm_features


def predict_forecast(df, ticker, horizon=1, recursive=False):
    model_horizon = 1 if recursive else horizon
    artifact = resolve_artifact(ticker, model_horizon, prefer_panel=not recursive)
    features = artifact["metadata"].get("features", list(df.columns))
    window_size = artifact["metadata"].get("window_size", 60)
    panel_returns = artifact["metadata"].get("forecast_contract", {}).get("target_kind") == "log_return_path"
    if panel_returns and recursive:
        raise ValueError("The pooled LSTM uses direct horizon forecasts; recursive prediction is unsupported")
    if panel_returns:
        frame = build_lstm_features(df)
        if list(frame.columns) != list(features):
            raise ValueError("Engineered LSTM features differ from the active model contract")
    else:
        if set(features) != set(df.columns):
            raise ValueError("Data features differ from the model contract; regenerate and retrain")
        frame = df.reindex(columns=features).copy()
    if len(frame) < window_size:
        raise ValueError(f"Need at least {window_size} historical rows")
    scaler = joblib.load(artifact["scaler"])
    target_column = artifact["metadata"].get("forecast_contract", {}).get(
        "target_column", FORECAST_TARGET_COLUMN
    )
    if artifact.get("pooled"):
        from src.panel_training import build_panel_inference_model
        model = build_panel_inference_model(model_horizon)
        model.load_weights(artifact["model"])
    elif artifact["metadata"].get("architecture") == "residual_lstm_last_close_calibrated_weights":
        from src.train_model import build_model
        model = build_model((window_size, len(features)), model_horizon,
                            close_index=features.index(target_column), compile_model=False)
        model.load_weights(artifact["model"])
    else:
        model = tf.keras.models.load_model(artifact["model"], compile=False)
    close = features.index(target_column) if not panel_returns else None
    def infer(data):
        values = data.tail(window_size).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Indicators contain missing values; more history is required")
        scaled = scaler.transform(values)
        predictions = np.asarray(model(scaled[None, ...], training=False)).reshape(-1)
        if panel_returns:
            entry = artifact["evaluation"]
            target_mean = np.asarray(entry["target_mean"], dtype=float)
            target_scale = np.asarray(entry["target_scale"], dtype=float)
            log_returns = (predictions * target_scale + target_mean) * float(entry.get("return_shrinkage", 1.0))
            anchor = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
            return anchor * np.exp(log_returns)
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
            frame = pd.concat([frame.reset_index(drop=True), row.to_frame().T], ignore_index=True)
            frame = add_technical_indicators(frame).reindex(columns=features)
        values = np.asarray(values)
    else:
        values = infer(frame)
        if len(values) != horizon:
            raise ValueError("Model output does not match the requested forecast horizon")
        shrinkage = float(artifact["evaluation"].get("return_shrinkage", 1.0))
        if shrinkage != 1.0 and not panel_returns:
            anchor = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
            values = anchor + shrinkage * (values - anchor)
    interval = artifact["evaluation"].get("interval") if not recursive else None
    if panel_returns and interval:
        anchor = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
        lower = anchor * np.exp(np.log(values / anchor) + np.asarray(interval["lower_return_residual"]))
        upper = anchor * np.exp(np.log(values / anchor) + np.asarray(interval["upper_return_residual"]))
    else:
        lower = values + np.asarray(interval["lower_residual"]) if interval else None
        upper = values + np.asarray(interval["upper_residual"]) if interval else None
    return {"prices": values, "lower": lower, "upper": upper,
            "artifact": artifact}
