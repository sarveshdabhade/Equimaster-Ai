#!/usr/bin/env python3
"""Update raw CSV files in data/ by fetching missing daily rows from yfinance.

This script looks for CSV files in the repository `data/` directory (files
matching '*.csv' but not inside processed/ or train_data/) and, for each file,
fetches OHLCV rows from Yahoo Finance starting the day after the file's last
date up to today (inclusive). New rows are appended in the same format as the
existing files (Date,AdjClose,Close,High,Low,Open,Volume) which matches the
project's preprocessor expectations.

Usage examples:
  # Dry-run to see which files would be updated (limit to 50 files):
  python scripts/update_csvs.py --dry-run --limit 50

  # Update up to 50 CSVs and append new rows to disk:
  python scripts/update_csvs.py --limit 50

Note: Requires the project's virtualenv with `yfinance` installed.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta
import argparse
import glob
import traceback

import pandas as pd
import yfinance as yf

# Reuse repo yfinance cache configuration if present
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, ".yf_cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(CACHE_DIR)


def parse_last_date_from_file(path: str) -> datetime | None:
    """Return the last date present in the CSV (assumes CSV has header + 3 meta rows).

    The repository raw CSVs use a 3-line header (Price,Ticker,Date) then rows like
    YYYY-MM-DD,... We parse the last non-empty line's date.
    """
    last_date = None
    with open(path, "r", encoding="utf-8") as f:
        # Read from end efficiently by reading all lines (files are moderate size)
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Skip first 3 metadata lines if present
    data_lines = lines[3:] if len(lines) > 3 else lines
    if not data_lines:
        return None
    last_line = data_lines[-1]
    parts = last_line.split(",")
    try:
        dt = datetime.strptime(parts[0], "%Y-%m-%d")
        return dt
    except Exception:
        return None


def append_rows_to_csv(path: str, df_new: pd.DataFrame) -> int:
    """Append rows from yfinance DataFrame to CSV in the project's format.

    Returns number of rows appended.
    """
    if df_new.empty:
        return 0

    # Ensure columns exist; if Adj Close is missing, fall back to Close
    required = ["Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df_new.columns:
            raise RuntimeError(f"Missing column in yfinance data: {c}")
    if "Adj Close" not in df_new.columns:
        # For many NSE yfinance downloads Adj Close == Close; fill accordingly
        df_new = df_new.copy()
        df_new["Adj Close"] = df_new["Close"]

    lines = []
    for idx, row in df_new.iterrows():
        # idx is Timestamp
        date_str = idx.strftime("%Y-%m-%d")
        adj = row.get("Adj Close", "")
        close = row.get("Close", "")
        high = row.get("High", "")
        low = row.get("Low", "")
        openp = row.get("Open", "")
        vol = int(row.get("Volume", 0) or 0)
        # Format floats with full precision to match existing files
        line = f"{date_str},{adj},{close},{high},{low},{openp},{vol}\n"
        lines.append(line)

    # Append to file
    with open(path, "a", encoding="utf-8", newline="") as f:
        for l in lines:
            f.write(l)

    return len(lines)


def update_csv(path: str, dry_run: bool = False) -> tuple[int, str]:
    """Update a single CSV file and return (rows_appended, message)."""
    try:
        fname = os.path.basename(path)
        # Derive ticker from filename (strip .csv)
        ticker = fname.rsplit('.', 1)[0]
        last_date = parse_last_date_from_file(path)
        today = datetime.utcnow().date()

        # If we couldn't parse a last date (empty or malformed file), fetch full history
        if last_date is None:
            # Use period='max' to get full history
            df = yf.download(ticker, period="max", interval="1d", progress=False)
        else:
            if last_date.date() >= today:
                return 0, "Up-to-date"
            # Start fetching from next calendar day
            start = last_date + timedelta(days=1)
            # yfinance end is exclusive for some uses; set end to tomorrow to include today
            end = today + timedelta(days=1)
            # Fetch daily history between start and end
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
        if df is None or df.empty:
            return 0, "No new rows"

        if dry_run:
            return len(df), f"Would append {len(df)} rows"

        appended = append_rows_to_csv(path, df)
        return appended, f"Appended {appended} rows"

    except Exception as e:
        return 0, f"Error: {e} \n{traceback.format_exc()}"


def main():
    p = argparse.ArgumentParser(description="Update raw CSVs in data/ from Yahoo Finance")
    p.add_argument("--limit", type=int, default=50, help="Maximum number of CSV files to update (default: 50)")
    p.add_argument("--dry-run", action="store_true", help="Do not write files; just report what would be updated")
    p.add_argument("--pattern", default="*.NS.csv", help="Filename glob pattern to match in data/ (default: '*.NS.csv')")
    args = p.parse_args()

    pattern = os.path.join(DATA_DIR, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No CSV files found matching pattern:", pattern)
        return

    files = [f for f in files if os.path.isfile(f) and 
             not os.path.normpath(f).startswith(os.path.join(DATA_DIR, 'processed'))]

    to_process = files[: args.limit]
    print(f"Found {len(files)} files; updating up to {len(to_process)} files")

    summary = []
    for path in to_process:
        print(f"Updating {os.path.basename(path)} ...")
        rows, msg = update_csv(path, dry_run=args.dry_run)
        print("  ->", msg)
        summary.append((os.path.basename(path), rows, msg))

    print("\nSummary:")
    for name, rows, msg in summary:
        print(f"{name}: {rows} -> {msg}")


if __name__ == '__main__':
    main()
