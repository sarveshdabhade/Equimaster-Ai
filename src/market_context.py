from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.news.sentiment import analyze_ticker_sentiment
from utils import load_data


def _data_path(ticker: str) -> str:
    return "data/processed/NSEI_processed.csv" if ticker == "NIFTY 50" else f"data/processed/{ticker}_processed.csv"


def _technical_signal(dataframe: pd.DataFrame | None) -> tuple[float, float]:
    if dataframe is None or dataframe.empty or "Close" not in dataframe:
        return 0.0, 0.0
    closes = dataframe["Close"].dropna()
    if len(closes) < 2:
        return 0.0, 0.0
    previous = float(closes.iloc[-2])
    current = float(closes.iloc[-1])
    short = float(closes.iloc[-10:-1].mean()) if len(closes) >= 10 else float(closes.iloc[:-1].mean())
    long = float(closes.iloc[-30:-1].mean()) if len(closes) >= 30 else short
    trend = ((current - short) / short + (current - long) / long) / 4 if short and long else 0.0
    return (current - previous) / previous if previous else 0.0, trend


def build_market_context(
    ticker: str,
    horizon: int = 5,
    sentiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ticker = (ticker or "").strip().upper()
    sentiment = sentiment if sentiment is not None else analyze_ticker_sentiment(ticker, max_articles=5)
    path = _data_path(ticker)
    dataframe = None
    if os.path.exists(path):
        stat = os.stat(path)
        fingerprint = f"{stat.st_size}_{int(stat.st_mtime) // 60}"
        dataframe = load_data(ticker, path, fingerprint)
    price_trend, technical_signal = _technical_signal(dataframe)
    sentiment_score = float(sentiment.get("overall_score", 0.0))
    combined = technical_signal * 0.5 + sentiment_score * 0.5
    label = "Bullish" if combined > 0.03 else "Bearish" if combined < -0.03 else "Neutral"
    return {
        "ticker": ticker,
        "horizon": horizon,
        "data_available": dataframe is not None and not dataframe.empty,
        "price_trend": round(price_trend, 4),
        "technical_signal": round(technical_signal, 4),
        "sentiment": sentiment,
        "combined_score": round(combined, 4),
        "label": label,
        "summary": f"{ticker} combines technical momentum with evidence-backed news sentiment.",
    }
