import os
import sys

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pandas_ta as ta
import glob

# 1. Setup Folders - FIXED TO POINT TO RAW
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def add_technical_indicators(df):
    """Adds RSI, SMA, and MACD to the dataframe."""
    df = df.sort_index()

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

    for file_path in csv_files:
        try:
            ticker = os.path.basename(file_path).replace(".csv", "")
            print(f"Processing {ticker}...")
            
            # 1. Load Data (Fixed for the clean auto-downloader format!)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Ensure Adj Close exists for downstream scripts
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df.insert(0, "Adj Close", df["Close"])

            # 2. Add Indicators
            df_processed = add_technical_indicators(df)

            # 3. Clean Data (Drop NaNs)
            df_clean = df_processed.dropna()

            # 4. Save to New Folder
            save_path = f"{PROCESSED_DIR}/{ticker}_processed.csv"
            df_clean.to_csv(save_path)
            print(f"Saved clean data with {df_clean.shape[1]} features to {save_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_all_files()