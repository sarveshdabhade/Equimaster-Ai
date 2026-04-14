#!/usr/bin/env python3
"""Fetch NIFTY 50 (NSEI) daily history from Yahoo Finance and save to data/NSEI.csv

Saves CSV in the repository's raw format (3-line header then rows Date,Adj Close,Close,High,Low,Open,Volume)

Usage:
  python scripts/fetch_nifty.py --days 3650
"""
import os
import argparse
from datetime import datetime
import yfinance as yf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'NSEI.csv')

os.makedirs(DATA_DIR, exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument('--days', type=int, default=3650, help='Number of days of history to fetch (approx)')
args = p.parse_args()

period = f"{max(30, args.days)}d"
print(f"Fetching NIFTY (NSEI) history for ~{args.days} days...")

df = yf.download('^NSEI', period=period, interval='1d', progress=False)
if df is None or df.empty:
    print('No data fetched')
    raise SystemExit(1)

# Ensure Adj Close present
if 'Adj Close' not in df.columns:
    df['Adj Close'] = df['Close']

# Write in repo's raw CSV format: add 3-line header to match preprocess expectations
with open(RAW_PATH, 'w', encoding='utf-8', newline='') as fh:
    fh.write('Price,NSEI,Daily\n')
    fh.write(f'Ticker,NSEI,Fetched:{datetime.utcnow().isoformat()}\n')
    fh.write('Date,Adj Close,Close,High,Low,Open,Volume\n')
    for idx, row in df.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        adj = row.get('Adj Close', '')
        close = row.get('Close', '')
        high = row.get('High', '')
        low = row.get('Low', '')
        openp = row.get('Open', '')
        vol = int(row.get('Volume', 0) or 0)
        fh.write(f"{date_str},{adj},{close},{high},{low},{openp},{vol}\n")

print(f"Saved {RAW_PATH}")
