from __future__ import annotations

import os
from typing import Any, Dict

from utils import load_data
from src.news.sentiment import analyze_ticker_sentiment


def _normalise_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _resolve_data_path(ticker: str) -> str | None:
    ticker = _normalise_ticker(ticker)
    if not ticker:
        return None
    if ticker == "NIFTY 50":
        return "data/processed/NSEI_processed.csv"
    return f"data/processed/{ticker}_processed.csv"


def _compute_technical_signal(df):
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0

    close = df["Close"].dropna()
    if len(close) < 5:
        return 0.0

    current = float(close.iloc[-1])
    short_ma = float(close.iloc[-10:-1].mean()) if len(close) >= 10 else float(close.iloc[:-1].mean())
    long_ma = float(close.iloc[-30:-1].mean()) if len(close) >= 30 else float(close.iloc[:-1].mean())

    if not short_ma or not long_ma:
        return 0.0

    trend = (current - short_ma) / short_ma + (current - long_ma) / long_ma
    return round(trend / 4.0, 4)


def build_market_context(ticker: str, horizon: int = 5) -> Dict[str, Any]:
    """Compose a simple market intelligence snapshot from technical trend + news sentiment.

    This gives the app a first working decision engine that combines model-like price
    momentum with an evidence-backed sentiment score.
    """
    ticker = _normalise_ticker(ticker)
    data_path = _resolve_data_path(ticker)
    signal = {
        "ticker": ticker,
        "horizon": horizon,
        "data_available": False,
        "price_trend": 0.0,
        "technical_signal": 0.0,
        "sentiment": {"overall_score": 0.0, "label": "Neutral", "confidence": 0.0},
        "combined_score": 0.0,
        "label": "Neutral",
        "summary": "Market context is unavailable right now.",
    }

    sentiment = analyze_ticker_sentiment(ticker, max_articles=5)
    signal["sentiment"] = {
        "overall_score": sentiment.get("overall_score", 0.0),
        "label": sentiment.get("label", "Neutral"),
        "confidence": sentiment.get("confidence", 0.0),
        "drivers": sentiment.get("drivers", []),
        "summary": sentiment.get("summary", ""),
    }

    if data_path and os.path.exists(data_path):
        df = load_data(ticker, data_path, os.path.getmtime(data_path))
        if df is not None and not df.empty and "Close" in df.columns:
            signal["data_available"] = True
            closes = df["Close"].dropna()
            if len(closes) >= 2:
                current = float(closes.iloc[-1])
                previous = float(closes.iloc[-2]) if len(closes) > 1 else current
                price_trend = (current - previous) / previous if previous else 0.0
                signal["price_trend"] = round(price_trend, 4)
            signal["technical_signal"] = _compute_technical_signal(df)

    sentiment_score = float(signal["sentiment"]["overall_score"])
    technical_signal = float(signal["technical_signal"])
    combined_score = (technical_signal * 0.5) + (sentiment_score * 0.5)
    signal["combined_score"] = round(combined_score, 4)

    if combined_score > 0.03:
        signal["label"] = "Bullish"
    elif combined_score < -0.03:
        signal["label"] = "Bearish"
    else:
        signal["label"] = "Neutral"

    signal["summary"] = (
        f"{ticker} combines a technical signal of {technical_signal:.4f}, a sentiment score of {sentiment_score:.2f}, "
        f"and a combined market score of {combined_score:.4f}."
    )
    if signal["sentiment"].get("summary"):
        signal["summary"] = signal["sentiment"]["summary"]

    return signal
