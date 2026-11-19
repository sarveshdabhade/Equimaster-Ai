import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
    """Fetch the latest index quote, with a one-minute history fallback."""
    label, symbol = label_symbol
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        # Prefer the quote endpoint used by the live ticker cards. It includes
        # the provider's quote timestamp and the preceding regular-session close.
        try:
            info = ticker.info
            last_raw = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("lastPrice")
            prev_raw = info.get("regularMarketPreviousClose") or info.get("previousClose")
            market_time = info.get("regularMarketTime")
            if last_raw is not None and prev_raw is not None and market_time is not None:
                last_val, prev_val = float(last_raw), float(prev_raw)
                if last_val > 0 and prev_val > 0:
                    timezone_name = info.get("exchangeTimezoneName") or "Asia/Kolkata"
                    try:
                        exchange_tz = ZoneInfo(timezone_name)
                    except (KeyError, ValueError):
                        exchange_tz = ZoneInfo("Asia/Kolkata")
                    quote_time = datetime.fromtimestamp(float(market_time), timezone.utc).astimezone(exchange_tz)
                    delta = last_val - prev_val
                    pct = delta / prev_val * 100.0
                    return (label, {"last": last_val, "delta": delta, "pct": pct,
                                    "as_of": quote_time.isoformat()})
        except Exception:
            pass

        # Keep minute bars as a fallback when Yahoo's quote metadata is absent.
        intraday = ticker.history(period="5d", interval="1m", prepost=False,
                                  auto_adjust=False, timeout=10)
        close_series = _extract_close_series(intraday, symbol)

        if close_series.empty:
            return None

        last_val = float(close_series.iloc[-1])
        session_closes = close_series.groupby(close_series.index.date).last()
        if len(session_closes) >= 2:
            prev_val = float(session_closes.iloc[-2])
        else:
            daily = ticker.history(period="5d", interval="1d", prepost=False,
                                    auto_adjust=False, timeout=10)
            daily_closes = _extract_close_series(daily, symbol)
            latest_session = close_series.index[-1].date()
            previous_sessions = daily_closes[daily_closes.index.date < latest_session]
            if previous_sessions.empty:
                return None
            prev_val = float(previous_sessions.iloc[-1])
        delta = last_val - prev_val
        pct = (delta / prev_val * 100.0) if prev_val != 0 else 0.0
        return (label, {"last": last_val, "delta": delta, "pct": pct,
                        "as_of": close_series.index[-1].isoformat()})
    except Exception:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_index_snapshot():
    """Fetch the latest available quotes for all configured market indices."""
    snapshot = {}

    # Keep requests sequential to bound provider load and isolate each quote.
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
