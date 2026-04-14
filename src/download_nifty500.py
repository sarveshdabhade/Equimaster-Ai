import os
import sys
import time
from io import StringIO

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, ".yf_cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

import requests  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

# Reuse yfinance cache directory configuration
yf.set_tz_cache_location(CACHE_DIR)


def _normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a clean 'Symbol' column in the DataFrame."""
    candidates = ["Symbol", "SYMBOL", "symbol"]
    sym_col = None
    for c in candidates:
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        # Fallback: assume first column holds the symbol
        sym_col = df.columns[0]

    df["Symbol"] = df[sym_col].astype(str).str.strip()
    df = df[df["Symbol"] != ""]
    return df


def fetch_nifty500_constituents() -> pd.DataFrame:
    """
    Get the Nifty 500 constituents.

    Preference order:
    1) If a local CSV exists at data/nifty500_symbols.csv, use that.
    2) Otherwise, try scraping the NIFTY 500 Wikipedia table.
    3) Finally, fall back to the official CSV from NSE Indices.
    """
    # 1) Local CSV, if user has provided one
    local_csv = os.path.join(DATA_DIR, "nifty500_symbols.csv")
    if os.path.exists(local_csv):
        print(f"Using local Nifty 500 symbol file: {local_csv}")
        df_local = pd.read_csv(local_csv)
        return _normalize_symbol_column(df_local)

    # 2) Try Wikipedia (often more reachable than NSE indices CSV)
    wiki_url = "https://en.wikipedia.org/wiki/NIFTY_500"
    try:
        print(f"Fetching Nifty 500 table from: {wiki_url}")
        resp = requests.get(wiki_url, timeout=30)
        resp.raise_for_status()

        tables = pd.read_html(resp.text)
        candidates = []
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any(c == "symbol" for c in cols_lower):
                candidates.append(t)

        if candidates:
            # The main constituent table is usually the first with a Symbol column
            df_wiki = candidates[0]
            print(f"Parsed Nifty 500 symbols from Wikipedia table (rows={len(df_wiki)})")
            return _normalize_symbol_column(df_wiki)
        else:
            print("Could not find a table with a 'Symbol' column on Wikipedia page.")
    except Exception as e:
        print(f"Failed to fetch/parse Nifty 500 from Wikipedia: {e}")

    # 3) Fallback: NSE Indices CSV (may timeout on some networks)
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    print(f"Fetching Nifty 500 constituent list from: {url}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    csv_text = resp.text
    df_nse = pd.read_csv(StringIO(csv_text))
    print(f"Parsed Nifty 500 symbols from NSE CSV (rows={len(df_nse)})")
    return _normalize_symbol_column(df_nse)


def build_nse_tickers(df: pd.DataFrame) -> list[str]:
    """
    Convert NSE symbols to Yahoo Finance NSE tickers (append .NS).
    """
    tickers = sorted({f"{sym}.NS" for sym in df["Symbol"] if isinstance(sym, str) and sym})
    return tickers


def download_nifty500_data():
    try:
        df_const = fetch_nifty500_constituents()
    except Exception as e:
        print(f"Failed to fetch Nifty 500 list from NSE Indices: {e}")
        print("   Please ensure internet access is available, or manually provide a symbol CSV.")
        return

    tickers = build_nse_tickers(df_const)
    print(f"Starting download for Nifty 500: {len(tickers)} stocks...")

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Downloading {ticker}...")
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} to {file_path} ({len(df)} rows)")
            time.sleep(0.5)  # be gentle with Yahoo
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    print("\nNifty 500 downloads complete!")


if __name__ == "__main__":
    download_nifty500_data()

