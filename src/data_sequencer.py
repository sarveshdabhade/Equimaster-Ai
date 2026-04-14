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
            scaled_data = scaler.fit_transform(dataset)

            # Save a unique Scaler for THIS stock
            joblib.dump(scaler, f"{TRAIN_DIR}/scaler_{ticker}.pkl")

            # 3. SEQUENCE (single-step)
            X, y = create_sequences(scaled_data, WINDOW_SIZE, close_col_idx)

            # 4. SPLIT (single-step)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # 5. SAVE TENSORS (single-step)
            np.save(f"{TRAIN_DIR}/X_train_{ticker}.npy", X_train)
            np.save(f"{TRAIN_DIR}/y_train_{ticker}.npy", y_train)
            np.save(f"{TRAIN_DIR}/X_test_{ticker}.npy", X_test)
            np.save(f"{TRAIN_DIR}/y_test_{ticker}.npy", y_test)

            # Also prepare multi-horizon sequences for common horizons
            horizons = [5, 22, 88]  # 1 week, 1 month (~22 trading days), 4 months (~88)
            for h in horizons:
                try:
                    Xm, Ym = create_multi_step_sequences(scaled_data, WINDOW_SIZE, h, close_col_idx)
                    if Xm.size == 0 or Ym.size == 0:
                        continue
                    split_m = int(len(Xm) * 0.8)
                    X_train_m, X_test_m = Xm[:split_m], Xm[split_m:]
                    y_train_m, y_test_m = Ym[:split_m], Ym[split_m:]

                    np.save(f"{TRAIN_DIR}/X_train_{ticker}_h{h}.npy", X_train_m)
                    np.save(f"{TRAIN_DIR}/y_train_{ticker}_h{h}.npy", y_train_m)
                    np.save(f"{TRAIN_DIR}/X_test_{ticker}_h{h}.npy", X_test_m)
                    np.save(f"{TRAIN_DIR}/y_test_{ticker}_h{h}.npy", y_test_m)
                except Exception:
                    # skip horizon if any error (not enough rows etc.)
                    continue

            print(f"Ready: {ticker}")

        except Exception as e:
            print(f"Skipped {ticker}: {e}")

if __name__ == "__main__":
    process_and_save_all()