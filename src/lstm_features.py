"""Causal, scale-independent features for the pooled equity LSTM."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "log_return_1", "log_return_5", "log_return_20", "log_return_60",
    "realized_volatility_5", "realized_volatility_20", "realized_volatility_60",
    "intraday_range", "overnight_gap", "close_location", "volume_change_1",
    "volume_vs_median_20", "rsi_centered", "distance_sma_50", "distance_ema_200",
    "macd_relative", "macd_hist_relative", "macd_signal_relative",
)


def build_lstm_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transform raw/processed OHLCV into features available at each close only."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV fields missing for LSTM features: {sorted(missing)}")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce").clip(lower=0)
    log_close = np.log(close.where(close > 0))
    one_day = log_close.diff()
    output = pd.DataFrame(index=frame.index)
    output["log_return_1"] = one_day
    for period in (5, 20, 60):
        output[f"log_return_{period}"] = log_close.diff(period)
    for period in (5, 20, 60):
        output[f"realized_volatility_{period}"] = one_day.rolling(period, min_periods=period).std()
    previous_close = close.shift(1)
    output["intraday_range"] = (high - low) / close.where(close > 0)
    output["overnight_gap"] = np.log(open_.where(open_ > 0) / previous_close.where(previous_close > 0))
    output["close_location"] = ((close - low) / (high - low).replace(0, np.nan)) - 0.5
    output["volume_change_1"] = np.log1p(volume).diff()
    volume_median = volume.rolling(20, min_periods=20).median()
    output["volume_vs_median_20"] = np.log((volume + 1.0) / (volume_median + 1.0))
    if "RSI" in frame:
        output["rsi_centered"] = (pd.to_numeric(frame["RSI"], errors="coerce") - 50.0) / 50.0
    else:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
        rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
        output["rsi_centered"] = (rsi - 50.0) / 50.0
    for name, source in (("distance_sma_50", "SMA_50"), ("distance_ema_200", "EMA_200")):
        moving_average = pd.to_numeric(frame[source], errors="coerce") if source in frame else close.rolling(
            50 if source == "SMA_50" else 200, min_periods=50 if source == "SMA_50" else 200
        ).mean()
        output[name] = close / moving_average.where(moving_average != 0) - 1.0
    macd_columns = {"macd_relative": "MACD_12_26_9", "macd_hist_relative": "MACDh_12_26_9",
                    "macd_signal_relative": "MACDs_12_26_9"}
    for target, source in macd_columns.items():
        if source not in frame:
            output[target] = np.nan
        else:
            output[target] = pd.to_numeric(frame[source], errors="coerce") / close.where(close > 0)
    output = output.replace([np.inf, -np.inf], np.nan)
    # Damp provider glitches and split artifacts without deleting the date.
    return output.clip(lower=-10.0, upper=10.0)
