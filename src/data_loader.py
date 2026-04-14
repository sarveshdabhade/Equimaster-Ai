import os
import sys
import time

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

# 2. Define stocks to download — NIFTY 50 + BANK NIFTY (NSE)
# We use ".NS" because these are on the National Stock Exchange of India
# Nifty 50 constituents (check NSE for latest)
NIFTY_50 = [
    "ADANIENT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "LICI.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "LTIM.NS", "TATACONSUM.NS",
]
# Bank Nifty (Nifty Bank) — 12 constituents; overlap with Nifty 50 merged below
BANK_NIFTY = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "AUBANK.NS",
]
# Combined, no duplicates (order: Nifty 50 first, then Bank Nifty-only tickers)
TICKERS = list(dict.fromkeys(NIFTY_50 + BANK_NIFTY))

def download_data():
    print(f"Starting download for {len(TICKERS)} stocks...")

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        try:
            # Fetch 10 years of data
            df = yf.download(ticker, period="10y", interval="1d", progress=False)

            # Save to CSV
            file_path = f"{DATA_DIR}/{ticker}.csv"
            df.to_csv(file_path)

            print(f"Saved {ticker} to {file_path} ({len(df)} rows)")
            time.sleep(0.5)  # avoid rate limiting

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    print("\nAll downloads complete!")

if __name__ == "__main__":
    download_data()