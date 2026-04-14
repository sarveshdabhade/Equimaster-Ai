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
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
}


@st.cache_data(ttl=60)
def fetch_index_snapshot():
    # Lazy import yfinance to avoid heavy imports at module import time
    try:
        import yfinance as yf
    except Exception:
        return {}

    snapshot = {}
    for label, symbol in INDEX_TICKERS.items():
        try:
            hist = yf.download(symbol, period="5d", interval="1d", progress=False)
            if not hist.empty:
                close_data = hist["Close"].dropna()

                if isinstance(close_data, pd.DataFrame):
                    if symbol in close_data.columns:
                        close_series = close_data[symbol].dropna()
                    else:
                        close_series = close_data.iloc[:, 0].dropna()
                else:
                    close_series = close_data

                if close_series.empty:
                    continue

                last_val = float(close_series.iloc[-1])
                prev_val = float(close_series.iloc[-2]) if len(close_series) > 1 else last_val
                delta = last_val - prev_val
                pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0
                snapshot[label] = {"last": last_val, "delta": delta, "pct": pct}
        except Exception:
            continue
    return snapshot


@st.cache_data(ttl=60)
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

        return {"symbol": symbol, "name": name or symbol, "last": last_val, "delta": delta, "pct": pct}
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_data(selected_ticker, data_path, data_mtime):
    """
    Prefer local CSV when available (fast). Only fetch live data when no local file exists
    or when the user explicitly requests the market index view.
    Lazy-imports `yfinance` to avoid startup overhead.
    """
    # If a processed CSV is available, load it immediately (fast path)
    if data_path and os.path.exists(data_path):
        try:
            _ = data_mtime
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
