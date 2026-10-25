import os
import requests
import pandas as pd
import streamlit as st

# Prepare yfinance cache directory
_CACHE_DIR = os.path.join("data", ".yf_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
try:
    yf.set_tz_cache_location(_CACHE_DIR)
except Exception:
    # yfinance may not provide this function in all versions — ignore safely
    pass


def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import INDEX_TICKERS


def _extract_close_series(hist, symbol):
    """Return the close series for one ticker from yfinance output, even when DataFrame columns are multi-indexed or symbol mismatched."""
    if hist is None or hist.empty:
        return pd.Series(dtype=float)

    close_data = hist.get("Close")
    if close_data is None:
        return pd.Series(dtype=float)

    close_data = close_data.dropna()
    if isinstance(close_data, pd.DataFrame):
        if symbol in close_data.columns:
            close_series = close_data[symbol]
        elif isinstance(close_data.columns, pd.MultiIndex):
            try:
                close_series = close_data.xs(symbol, level=-1, axis=1)
            except Exception:
                close_series = close_data.iloc[:, 0]
        else:
            close_series = close_data.iloc[:, 0]
    else:
        close_series = close_data

    close_series = pd.to_numeric(close_series, errors="coerce").dropna()
    return close_series


def _fetch_single_index(label_symbol):
    """Fetch data for a single index."""
    label, symbol = label_symbol
    try:
        import yfinance as yf
        hist = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=False)
        close_series = _extract_close_series(hist, symbol)

        if close_series.empty:
            return None

        last_val = float(close_series.iloc[-1])
        prev_val = float(close_series.iloc[-2]) if len(close_series) > 1 else last_val
        delta = last_val - prev_val
        pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0
        return (label, {"last": last_val, "delta": delta, "pct": pct, "as_of": str(close_series.index[-1])[:10]})
    except Exception:
        pass
    return None

@st.cache_data(ttl=600)
def fetch_index_snapshot():
    """Fetch all index data in parallel."""
    snapshot = {}

    # yfinance can cross-contaminate values when multiple symbols are downloaded
    # concurrently in the same process; sequential fetches keep each index tied to
    # its own symbol and avoid duplicated price deltas across the ticker strip.
    for item in INDEX_TICKERS.items():
        result = _fetch_single_index(item)
        if result:
            label, data = result
            snapshot[label] = data
    
    return snapshot


@st.cache_data(ttl=300)
def fetch_stock_snapshot(symbol: str):
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        hist = t.history(period="3d", interval="1d")
        if hist.empty:
            return None

        close_data = hist["Close"].dropna()
        if isinstance(close_data, pd.DataFrame):
            close_series = close_data.iloc[:, 0].dropna()
        else:
            close_series = close_data

        if close_series.empty:
            return None

        last_val = float(close_series.iloc[-1])
        prev_val = float(close_series.iloc[-2]) if len(close_series) > 1 else last_val
        delta = last_val - prev_val
        pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0

        name = None
        try:
            info = t.info
            name = info.get('longName') or info.get('shortName')
        except Exception:
            name = None

        return {"symbol": symbol, "name": name or symbol, "last": last_val, "delta": delta, "pct": pct, "as_of": str(close_series.index[-1])[:10]}
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_data(selected_ticker, data_path, _data_fingerprint):
    """
    Prefer local CSV when available (fast). Only fetch live data when no local file exists
    or when the user explicitly requests the market index view.
    Lazy-imports `yfinance` to avoid startup overhead.

    _data_fingerprint: a stable hash of (file_size, mtime_rounded_to_minute)
                       used solely to invalidate cache when the file changes.
    """
    # If a processed CSV is available, load it immediately (fast path)
    if data_path and os.path.exists(data_path):
        try:
            df_local = pd.read_csv(data_path, index_col=0, parse_dates=True)
            return df_local.sort_index()
        except FileNotFoundError:
            pass

    # If user selected the home market index, fetch live NIFTY (fallback)
    if selected_ticker == "NIFTY 50 (Home)":
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI")
            df = nifty.history(period="2y", interval="1d")
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                return df
        except Exception as e:
            # Keep failure silent for UI; return None so app can continue
            return None

    return None
