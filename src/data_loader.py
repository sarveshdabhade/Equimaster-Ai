import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

# Set yfinance cache to project folder (avoids "unable to open database file" on restricted envs)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", ".yf_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

import yfinance as yf
yf.set_tz_cache_location(_CACHE_DIR)

import pandas as pd

# 1. Define the directory to save data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)

# Import centralized ticker configuration
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import ALL_DOWNLOAD_TICKERS as TICKERS

def _download_single_ticker(ticker):
    """Download data for a single ticker."""
    try:
        # Fetch 10 years of data
        df = yf.download(ticker, period="10y", interval="1d", progress=False)

        # Save to CSV (to raw directory, consistent with update_data.py)
        file_path = f"{DATA_DIR}/raw/{ticker}.csv"
        df.to_csv(file_path)

        return (ticker, len(df), None)
    except Exception as e:
        return (ticker, 0, str(e))

def download_data(max_workers=3):
    """Download data for all tickers in parallel."""
    print(f"Starting parallel download for {len(TICKERS)} stocks (workers={max_workers})...")

    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_ticker, ticker): ticker for ticker in TICKERS}
        
        for future in as_completed(futures):
            ticker, row_count, error = future.result()
            
            if error:
                print(f"[ERROR] {ticker}: {error}")
                error_count += 1
            else:
                print(f"[OK] {ticker}: {row_count} rows")
                success_count += 1
            
            # Small delay to be nice to the API
            time.sleep(0.1)

    print(f"\nDownloads complete! Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    download_data()