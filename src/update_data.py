import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
import time
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Configuration
# Import centralized ticker configuration
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import UPDATE_TICKERS as TICKERS, provider_symbol
from src.market_data import merge_history_without_window_loss, remove_incomplete_current_daily_bar

RAW_DIR = os.path.join(ROOT, "data", "raw")
yf.set_tz_cache_location(os.path.join(ROOT, "data", ".yf_cache"))

os.makedirs(RAW_DIR, exist_ok=True)


def equivalent_history(previous, current):
    """Ignore provider floating-point noise while retaining material corrections."""
    if not previous.index.equals(current.index) or set(previous.columns) != set(current.columns):
        return False
    return bool(np.allclose(previous.reindex(columns=current.columns).to_numpy(dtype=float),
                            current.to_numpy(dtype=float), rtol=5e-7, atol=1e-6, equal_nan=True))


def _latest_date(path: str):
    """Return max index date for a CSV file, or None if unavailable."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df.index.max()
    except Exception:
        return None


def run_downstream_pipeline(skip_training: bool = False) -> bool:
    """Run preprocess -> sequence -> train after raw data updates."""
    steps = [
        [sys.executable, "src/preprocessor.py"],
        [sys.executable, "src/data_sequencer.py"],
    ]

    if not skip_training:
        steps.append([sys.executable, "src/train_model.py"])

    for cmd in steps:
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Downstream step failed: {' '.join(cmd)}")
            print(str(exc))
            return False

    return True

def _update_single_ticker(ticker):
    """Update data for a single ticker."""
    try:
        # We download the last 5 years of data up to TODAY
        stock = yf.Ticker(provider_symbol(ticker))
        df = stock.history(period="5y", interval="1d", timeout=20)
        df = remove_incomplete_current_daily_bar(df)
        
        if not df.empty:
            # Remove timezone to keep it clean
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Save and overwrite the old CSV with fresh data
            safe_ticker = ticker.replace("^", "")
            save_path = f"{RAW_DIR}/{safe_ticker}.csv"

            prev_max_date = _latest_date(save_path)
            if os.path.exists(save_path):
                previous = pd.read_csv(save_path, index_col=0, parse_dates=True)
                if previous.index.tz is not None:
                    previous.index = previous.index.tz_localize(None)

                # `period="5y"` advances its start date each day. Preserve any
                # older rows that fall just outside the refreshed provider window.
                df = merge_history_without_window_loss(previous, df)

                if equivalent_history(previous, df):
                    return (ticker, True, False, None)
            
            temporary = save_path + ".tmp"
            df.to_csv(temporary)
            os.replace(temporary, save_path)

            new_max_date = _latest_date(save_path)
            has_new_data = True  # New dates or a material historical correction.

            return (ticker, True, has_new_data, None)
        else:
            return (ticker, False, False, "No complete daily data found (today's bar may still be in progress)")
                
    except Exception as e:
        return (ticker, False, False, str(e))

def update_all_stocks(skip_downstream: bool = False, skip_training: bool = False, max_workers=3):
    print(f"Starting parallel daily data fetch at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total assets to fetch: {len(TICKERS)} (workers={max_workers})\n")

    tickers_with_new_rows = 0
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_update_single_ticker, ticker): ticker for ticker in TICKERS}
        
        for future in as_completed(futures):
            ticker, success, has_new_data, error = future.result()
            
            if error:
                print(f"[ERROR] {ticker}: {error}")
                error_count += 1
            else:
                safe_ticker = ticker.replace("^", "")
                if has_new_data:
                    tickers_with_new_rows += 1
                    print(f"[OK+NEW] {safe_ticker}.csv updated with new data")
                else:
                    print(f"[OK] {safe_ticker}.csv updated")
                success_count += 1
            
            # Small delay to be nice to the API
            time.sleep(0.1)
            
    print(f"\nAll CSVs updated! Success: {success_count}, Errors: {error_count}, New data: {tickers_with_new_rows}")
    if error_count:
        raise RuntimeError(f"Market refresh failed for {error_count} ticker(s); existing files retained for failures.")

    if tickers_with_new_rows > 0 and not skip_downstream:
        print(f"Detected new rows in {tickers_with_new_rows} ticker files. Running downstream pipeline...")
        ok = run_downstream_pipeline(skip_training=skip_training)
        if ok:
            print("Downstream pipeline completed successfully.")
        else:
            raise RuntimeError("Downstream pipeline failed after raw data update.")
    elif tickers_with_new_rows == 0:
        print("No new market rows detected. Skipping downstream pipeline.")
    else:
        print("Skipping downstream pipeline by flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update raw data and optionally trigger retraining pipeline.")
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Only update raw CSVs. Do not run preprocess/sequence/train steps.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Run preprocess/sequence but skip model training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers for fetching (default: 3)",
    )
    args = parser.parse_args()

    update_all_stocks(skip_downstream=args.skip_downstream, skip_training=args.skip_training, max_workers=args.workers)
