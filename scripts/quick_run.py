#!/usr/bin/env python3
"""Quick end-to-end runner for one ticker (preprocess -> sequence -> train -> predict)

Usage: python scripts/quick_run.py --ticker RELIANCE.NS --epochs 1
"""
import os
import sys
import argparse

# Ensure repo root is on path so we can import src modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.preprocessor import add_technical_indicators
from src.data_sequencer import create_sequences, create_multi_step_sequences
from src.train_model import build_model


def preprocess_one(ticker):
    raw_path = f"data/{ticker}.csv"
    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)
    out_path = f"{proc_dir}/{ticker}_processed.csv"

    if os.path.exists(out_path):
        print(f"Processed file already exists: {out_path}")
        return out_path

    print(f"Preprocessing {raw_path} -> {out_path} ...")
    # Auto-detect CSV format: some raw files have a 3-line metadata header (from fetch_nifty.py),
    # while others (from update_data.py) are standard yfinance CSVs with a header row.
    with open(raw_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    if first_line.startswith("Price,"):
        # 3-line metadata header format (fetch_nifty.py output)
        df = pd.read_csv(raw_path, skiprows=3, names=["Date","Close","High","Low","Open","Volume"], parse_dates=[0], index_col=0)
    else:
        # Standard yfinance CSV format (update_data.py / data_loader.py output)
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df.insert(0, "Adj Close", df["Close"])
    df = add_technical_indicators(df)
    df = df.dropna()
    df.to_csv(out_path)
    print("Saved processed CSV")
    return out_path


def sequence_one(ticker, window_size=60, horizon=1):
    proc_path = f"data/processed/{ticker}_processed.csv"
    train_dir = "data/train_data"
    os.makedirs(train_dir, exist_ok=True)

    print(f"Sequencing {proc_path} ...")
    df = pd.read_csv(proc_path, index_col=0)
    dataset = df.values

    if len(dataset) <= window_size:
        raise RuntimeError(f"Not enough rows ({len(dataset)}) for window_size={window_size}")

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(dataset)
    joblib.dump(scaler, f"{train_dir}/scaler_{ticker}.pkl")

    if horizon <= 1:
        X, y = create_sequences(scaled, window_size, close_col_idx=1)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        np.save(f"{train_dir}/X_train_{ticker}.npy", X_train)
        np.save(f"{train_dir}/y_train_{ticker}.npy", y_train)
        np.save(f"{train_dir}/X_test_{ticker}.npy", X_test)
        np.save(f"{train_dir}/y_test_{ticker}.npy", y_test)
    else:
        X, Y = create_multi_step_sequences(scaled, window_size, horizon, close_col_idx=1)
        if X.size == 0 or Y.size == 0:
            raise RuntimeError(f"Not enough rows to create horizon={horizon} sequences")
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = Y[:split], Y[split:]

        np.save(f"{train_dir}/X_train_{ticker}_h{horizon}.npy", X_train)
        np.save(f"{train_dir}/y_train_{ticker}_h{horizon}.npy", y_train)
        np.save(f"{train_dir}/X_test_{ticker}_h{horizon}.npy", X_test)
        np.save(f"{train_dir}/y_test_{ticker}_h{horizon}.npy", y_test)

    print(f"Saved sequences and scaler to {train_dir}")
    return X_train, y_train, X_test, y_test


def train_one(ticker, X_train, y_train, epochs=1, horizon=1):
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print(f"Building model for {ticker} with input shape {(X_train.shape[1], X_train.shape[2])} and horizon={horizon} ...")
    model = build_model((X_train.shape[1], X_train.shape[2]), output_size=(y_train.shape[1] if len(y_train.shape) > 1 else 1))
    print(f"Training for {epochs} epoch(s) ...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    suffix = f"_h{horizon}" if horizon and horizon > 1 else ""
    out = f"{models_dir}/lstm_{ticker}{suffix}.h5"
    model.save(out)
    print(f"Saved model to {out}")
    return out


def predict_once(ticker, model_path, scaler_path, X_test, y_test, close_col_idx=1):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    print("Loading model and scaler for prediction...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    preds = model.predict(X_test)
    n_features = scaler.n_features_in_
    dummy_pred = np.zeros((len(X_test), n_features))
    # y_test can be 1-D (single-step) or 2-D (multi-step). For plotting,
    # compare the first forecasted step (T+1) for multi-step models so
    # charts remain comparable across horizons.
    if preds.ndim == 1:
        pred_for_plot = preds
    elif preds.ndim == 2:
        pred_for_plot = preds[:, 0]
    else:
        pred_for_plot = preds.flatten()

    if y_test.ndim == 1:
        actual_for_plot = y_test
    else:
        actual_for_plot = y_test[:, 0]

    dummy_pred[:, close_col_idx] = pred_for_plot
    dummy_actual = np.zeros((len(actual_for_plot), n_features))
    dummy_actual[:, close_col_idx] = actual_for_plot
    inv_pred = scaler.inverse_transform(dummy_pred)[:, close_col_idx]
    inv_actual = scaler.inverse_transform(dummy_actual)[:, close_col_idx]

    # quick plot saved
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(inv_actual, label='Actual')
    plt.plot(inv_pred, label='Pred')
    plt.legend()
    out = f"images/prediction_{ticker}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved prediction plot to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='RELIANCE.NS')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--horizon', type=int, default=1, help='Forecast horizon in trading days (1,5,22,88)')
    args = p.parse_args()

    ticker = args.ticker
    epochs = args.epochs

    # 1. Preprocess (single ticker)
    preprocess_one(ticker)

    # 2. Sequence
    X_train, y_train, X_test, y_test = sequence_one(ticker, window_size=60, horizon=args.horizon)

    # 3. Train
    model_path = train_one(ticker, X_train, y_train, epochs=epochs, horizon=args.horizon)

    # 4. Predict/plot
    scaler_path = f"data/train_data/scaler_{ticker}.pkl"
    predict_once(ticker, model_path, scaler_path, X_test, y_test)


if __name__ == '__main__':
    main()
