import yfinance as yf
import pandas as pd
import os
import datetime
import time
import sys
import subprocess
import argparse

if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Configuration
# All 50 NIFTY stocks + The NIFTY 50 Index (^NSEI)
TICKERS = [
    "^NSEI", "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", 
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", 
    "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", 
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", 
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", 
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", 
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", 
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", 
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

RAW_DIR = "data/raw"  

os.makedirs(RAW_DIR, exist_ok=True)


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

def update_all_stocks(skip_downstream: bool = False, skip_training: bool = False):
    print(f"Starting daily data fetch at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total assets to fetch: {len(TICKERS)}\n")

    tickers_with_new_rows = 0
    
    for ticker in TICKERS:
        print(f"Fetching latest data for {ticker}...")
        try:
            # We download the last 5 years of data up to TODAY
            stock = yf.Ticker(ticker)
            df = stock.history(period="5y", interval="1d")
            
            if not df.empty:
                # Remove timezone to keep it clean
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Save and overwrite the old CSV with fresh data
                # We replace the '^' in the index ticker so Windows doesn't get confused
                safe_ticker = ticker.replace("^", "")
                save_path = f"{RAW_DIR}/{safe_ticker}.csv"

                prev_max_date = _latest_date(save_path)
                
                df.to_csv(save_path)

                new_max_date = _latest_date(save_path)
                if prev_max_date is None or (new_max_date is not None and new_max_date > prev_max_date):
                    tickers_with_new_rows += 1

                print(f"Updated {safe_ticker}.csv successfully.")
            else:
                print(f"No data found for {ticker}.")
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
        # PRO-TIP: Pause for 1 second between downloads so Yahoo Finance doesn't block your IP!
        time.sleep(1)
            
    print("\nAll CSVs are now up to date.")

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
    args = parser.parse_args()

    update_all_stocks(skip_downstream=args.skip_downstream, skip_training=args.skip_training)
    