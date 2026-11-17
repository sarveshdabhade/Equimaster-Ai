import os
import sys
import hashlib
import io
import json
from pathlib import Path

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import pandas_ta as ta
import glob

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_PRICE_COLUMNS = (
    "Adj Close", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
)


def can_reuse_active_snapshot(snapshot, candidate, rtol=1e-6, atol=1e-6):
    """Keep a trained data snapshot when a refresh adds no session or material price change.

    Rolling provider windows may omit the oldest row and revise historical floats
    slightly. Those changes should not replace the exact dataset paired with an
    active model when the latest completed session has not advanced.
    """
    if snapshot is None or candidate is None or snapshot.empty or candidate.empty:
        return False
    if list(snapshot.columns) != list(candidate.columns):
        return False
    if snapshot.index.has_duplicates or candidate.index.has_duplicates:
        return False
    if snapshot.index.max() != candidate.index.max():
        return False

    start = snapshot.index.get_indexer([candidate.index[0]])[0]
    if start < 0 or not snapshot.index[start:].equals(candidate.index):
        return False

    columns = [column for column in _PRICE_COLUMNS if column in snapshot.columns]
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(columns):
        return False
    try:
        previous_prices = snapshot.loc[candidate.index, columns].to_numpy(dtype=float)
        refreshed_prices = candidate.loc[:, columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.allclose(previous_prices, refreshed_prices, rtol=rtol, atol=atol, equal_nan=True))


def _load_active_processed_snapshot(ticker):
    """Load the integrity-checked processed CSV saved with the active bundle."""
    try:
        from src.artifacts import model_directory

        model_root = model_directory().resolve()
        pointer_path = model_root / f"active_{ticker}.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        bundle = (model_root / pointer["bundle"]).resolve()
        if not bundle.is_relative_to(model_root):
            return None
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        snapshot_bytes = (bundle / "processed.csv").read_bytes()
        if hashlib.sha256(snapshot_bytes).hexdigest() != metadata.get("data_sha256"):
            return None
        snapshot = pd.read_csv(io.BytesIO(snapshot_bytes), index_col=0, parse_dates=True)
        return snapshot, snapshot_bytes
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

# 1. Setup Folders - FIXED TO POINT TO RAW
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def add_technical_indicators(df):
    """Adds RSI, SMA, and MACD to the dataframe."""
    df = df.sort_index().copy()
    # Replace previously calculated indicators rather than concatenating duplicates.
    indicator_columns = [c for c in df.columns if c in ('RSI', 'SMA_50', 'EMA_200', 'MACD')
                         or c.startswith(('MACD_', 'MACDh_', 'MACDs_'))]
    df = df.drop(columns=indicator_columns)

    # A. RSI 
    rsi = df.ta.rsi(length=14)
    if isinstance(rsi, pd.DataFrame):
        rsi = rsi.iloc[:, 0] if rsi.shape[1] >= 1 else rsi.squeeze()
    df['RSI'] = rsi

    # B. SMA
    sma50 = df.ta.sma(length=50)
    if isinstance(sma50, pd.DataFrame):
        sma50 = sma50.iloc[:, 0] if sma50.shape[1] >= 1 else sma50.squeeze()
    df['SMA_50'] = sma50

    # C. EMA
    ema200 = df.ta.ema(length=200)
    if isinstance(ema200, pd.DataFrame):
        ema200 = ema200.iloc[:, 0] if ema200.shape[1] >= 1 else ema200.squeeze()
    df['EMA_200'] = ema200

    # D. MACD 
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        df = pd.concat([df, macd], axis=1)
    else:
        df['MACD'] = macd

    return df

def process_all_files():
    csv_files = glob.glob(f"{RAW_DIR}/*.csv")
    print(f"Found {len(csv_files)} files to process in {RAW_DIR}...")

    failures = []
    for file_path in csv_files:
        try:
            ticker = os.path.basename(file_path).replace(".csv", "")
            print(f"Processing {ticker}...")
            
            # 1. Load Data (Fixed for the clean auto-downloader format!)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                raise ValueError(f"Missing required price columns: {required - set(df.columns)}")
            df.index = pd.to_datetime(df.index, errors="raise")
            if df.index.has_duplicates or df.index.isna().any():
                raise ValueError("Missing or duplicate historical dates")
            
            # Ensure Adj Close exists for downstream scripts
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df.insert(0, "Adj Close", df["Close"])

            # 2. Add Indicators
            df_processed = add_technical_indicators(df)

            # 3. Clean Data (Drop NaNs)
            df_clean = df_processed.dropna()
            if df_clean.empty or not np.isfinite(df_clean.to_numpy(dtype=float)).all():
                raise ValueError("No finite processed rows; retain existing data and refresh sufficient history")

            # 4. Save to New Folder
            save_path = f"{PROCESSED_DIR}/{ticker}_processed.csv"
            active_snapshot = _load_active_processed_snapshot(ticker)
            if active_snapshot and can_reuse_active_snapshot(active_snapshot[0], df_clean):
                # Preserve the original artifact bytes, including its float
                # precision and formatting, so its data hash remains valid.
                snapshot_bytes = active_snapshot[1]
                temporary = save_path + ".tmp"
                with open(temporary, "wb") as output:
                    output.write(snapshot_bytes)
                os.replace(temporary, save_path)
                print(f"Retained active model snapshot for {ticker}; no new completed session")
                continue

            df_clean.to_csv(save_path + ".tmp")
            os.replace(save_path + ".tmp", save_path)
            print(f"Saved clean data with {df_clean.shape[1]} features to {save_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failures.append(f"{file_path}: {e}")

    if failures:
        raise RuntimeError(f"Preprocessing failed: {failures}")

if __name__ == "__main__":
    process_all_files()
