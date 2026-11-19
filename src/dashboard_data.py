"""Fast saved-data views used by the research dashboard; no invented quotes or scores."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from config import ALL_DOWNLOAD_TICKERS, resolve_app_path
from src.fundamentals import read_cached_details


def data_fingerprint():
    processed = Path(resolve_app_path("data", "processed"))
    fundamentals = Path(resolve_app_path("data", "fundamentals"))
    files = [processed / f"{ticker}_processed.csv" for ticker in ALL_DOWNLOAD_TICKERS]
    files += [fundamentals / f"{ticker}.json" for ticker in ALL_DOWNLOAD_TICKERS]
    files.append(processed / "NSEI_processed.csv")
    files = [path for path in files if path.exists()]
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(files))


@st.cache_data(ttl=120, show_spinner=False)
def load_dashboard_rows(fingerprint=()):
    records = []
    processed = Path(resolve_app_path("data", "processed"))
    for ticker in ALL_DOWNLOAD_TICKERS:
        path = processed / f"{ticker}_processed.csv"
        if not path.exists():
            continue
        try:
            history = pd.read_csv(path, index_col=0).tail(253)
            closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if len(closes) < 2:
                continue
            price, previous = float(closes.iloc[-1]), float(closes.iloc[-2])
            details = read_cached_details(ticker) or {}
            metrics = details.get("metrics", {})
            def value(key):
                return metrics.get(key, {}).get("value")
            records.append({"Ticker": ticker, "Symbol": ticker.removesuffix(".NS"),
                            "Company": details.get("Company") or ticker.removesuffix(".NS"),
                            "Sector": details.get("Sector") or "Unclassified", "Industry": details.get("Industry"),
                            "Price": price, "Change": price - previous,
                            "Change %": (price / previous - 1) * 100 if previous else None,
                            "1M %": (price / float(closes.iloc[-22]) - 1) * 100 if len(closes) >= 22 and closes.iloc[-22] else None,
                            "Volume": float(history["Volume"].iloc[-1]) if "Volume" in history else None,
                            "RSI": float(history["RSI"].iloc[-1]) if "RSI" in history else None,
                            "P/E": value("pe"), "P/B": value("pb"), "ROE %": value("roe"),
                            "ROCE %": value("roce"), "Debt/equity": value("debt_equity"),
                            "Market cap (Cr)": value("market_cap") / 1e7 if value("market_cap") is not None else None,
                            "Financial sector": details.get("financial_sector", False),
                            "As of": str(history.index[-1])[:10]})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(records)


@st.cache_data(ttl=120, show_spinner=False)
def index_history(fingerprint=()):
    path = Path(resolve_app_path("data", "processed", "NSEI_processed.csv"))
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True).tail(253)
    except (OSError, ValueError):
        return pd.DataFrame()


def apply_screen(frame, sector="All sectors", preset="All companies", query=""):
    result = frame.copy()
    if result.empty:
        return result
    if sector != "All sectors":
        result = result[result["Sector"] == sector]
    if query:
        text = result["Symbol"].astype(str) + " " + result["Company"].astype(str)
        result = result[text.str.contains(query, case=False, regex=False)]
    if preset == "Positive P/E under 20":
        result = result[(result["P/E"] > 0) & (result["P/E"] < 20)]
    elif preset == "ROE at least 15%":
        result = result[result["ROE %"] >= 15]
    elif preset == "Debt/equity under 0.5x":
        result = result[(result["Debt/equity"] >= 0) & (result["Debt/equity"] < .5) & ~result["Financial sector"]]
    elif preset == "Positive 1-month return":
        result = result[result["1M %"] > 0]
    return result


def cached_news(limit=6):
    from src.news.fetcher import _is_trusted_article
    cutoff, now = datetime.now(timezone.utc) - timedelta(days=3), datetime.now(timezone.utc)
    articles, seen = [], set()
    for path in Path(resolve_app_path("data", "news_cache")).glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for article in payload if isinstance(payload, list) else []:
                published = datetime.fromisoformat(str(article.get("published_at")).replace("Z", "+00:00"))
                published = published.replace(tzinfo=timezone.utc) if published.tzinfo is None else published
                url = article.get("url")
                if cutoff <= published <= now and url not in seen and _is_trusted_article(article):
                    articles.append(article)
                    seen.add(url)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(articles, key=lambda item: item.get("published_at", ""), reverse=True)[:limit]
