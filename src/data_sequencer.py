import os
import sys

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import glob
import numpy as np
import pandas as pd
import joblib  
from sklearn.preprocessing import MinMaxScaler
from config import HORIZON_LABELS

# 1. Config
PROCESSED_DIR = "data/processed"
TRAIN_DIR = "data/train_data"
os.makedirs(TRAIN_DIR, exist_ok=True)

WINDOW_SIZE = 60  

def create_sequences(data, window_size, close_col_idx):
    """
    Converts a 2D array into a 3D array of sequences.
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i, close_col_idx])

    return np.array(X), np.array(y)


def create_multi_step_sequences(data, window_size, horizon, close_col_idx):
    """
    Create X, Y where Y is a vector of the next `horizon` close values.
    X shape: (samples, window_size, features)
    Y shape: (samples, horizon)
    """
    X, Y = [], []
    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i - window_size: i])
        # collect next `horizon` close values
        ys = [data[i + j, close_col_idx] for j in range(horizon)]
        Y.append(ys)
    return np.array(X), np.array(Y)

def chronological_split_index(row_count, window_size=60, train_fraction=0.8):
    split = window_size + int((row_count - window_size) * train_fraction)
    if not window_size < split < row_count:
        raise ValueError("Not enough rows for non-empty training and test sets")
    return split


def split_sequences(data, window_size, horizon, close_col_idx, split_row):
    """Keep training targets strictly before the first test target date."""
    X, y = create_multi_step_sequences(data, window_size, horizon, close_col_idx)
    train_count = split_row - window_size - horizon + 1
    test_start = split_row - window_size
    if train_count <= 0 or test_start >= len(X):
        raise ValueError(f"Not enough rows for disjoint horizon={horizon} targets")
    if horizon == 1:
        y = y[:, 0]
    return X[:train_count], y[:train_count], X[test_start:], y[test_start:]


def process_and_save_all():
    files = glob.glob(f"{PROCESSED_DIR}/*_processed.csv")
    print(f"Starting Sequence Generation for {len(files)} stocks...")

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            ticker = filename.replace("_processed.csv", "")
            print(f"\nProcessing {ticker}...")

            # Load Data
            df = pd.read_csv(file_path, index_col=0)

            # THE FIX: Find the 'Close' column dynamically no matter where it is!
            if "Close" not in df.columns:
                print(f"Skipped {ticker}: 'Close' column missing!")
                continue

            close_col_idx = df.columns.get_loc("Close")

            dataset = df.values

            if len(dataset) == 0 or len(dataset) <= WINDOW_SIZE:
                print(f"Skipped {ticker}: Not enough rows ({len(dataset)}). Need > {WINDOW_SIZE}.")
                continue

            # 2. SCALE
            scaler = MinMaxScaler(feature_range=(0, 1))
            split_row = chronological_split_index(len(dataset), WINDOW_SIZE)
            scaler.fit(dataset[:split_row])
            scaled_data = scaler.transform(dataset)

            # Save a unique Scaler for THIS stock
            joblib.dump(scaler, f"{TRAIN_DIR}/scaler_{ticker}.pkl")

            # 3. SEQUENCE (single-step)
            X_train, y_train, X_test, y_test = split_sequences(
                scaled_data, WINDOW_SIZE, 1, close_col_idx, split_row
            )

            # 5. SAVE TENSORS (single-step)
            np.save(f"{TRAIN_DIR}/X_train_{ticker}.npy", X_train)
            np.save(f"{TRAIN_DIR}/y_train_{ticker}.npy", y_train)
            np.save(f"{TRAIN_DIR}/X_test_{ticker}.npy", X_test)
            np.save(f"{TRAIN_DIR}/y_test_{ticker}.npy", y_test)

            # Also prepare multi-horizon sequences for common horizons
            horizons = [5, 22, 63]  # Week, Month, and approximately one trading quarter.
            for h in horizons:
                try:
                    X_train_m, y_train_m, X_test_m, y_test_m = split_sequences(
                        scaled_data, WINDOW_SIZE, h, close_col_idx, split_row
                    )

                    np.save(f"{TRAIN_DIR}/X_train_{ticker}_h{h}.npy", X_train_m)
                    np.save(f"{TRAIN_DIR}/y_train_{ticker}_h{h}.npy", y_train_m)
                    np.save(f"{TRAIN_DIR}/X_test_{ticker}_h{h}.npy", X_test_m)
                    np.save(f"{TRAIN_DIR}/y_test_{ticker}_h{h}.npy", y_test_m)
                except ValueError as exc:
                    label = HORIZON_LABELS.get(h, f"h{h}")
                    print(f"Skipped {ticker} {label} ({h} sessions): {exc}")

            print(f"Ready: {ticker}")

        except Exception as e:
            print(f"Skipped {ticker}: {e}")

if __name__ == "__main__":
    process_and_save_all()
