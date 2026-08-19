import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
NEWS_CACHE_DIR = ROOT_DIR / "data" / "news_cache"
NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_WORDS = [
    "rally", "surge", "upgrade", "beat", "strong", "growth", "profit", "expansion",
    "demand", "outperform", "gains", "acceleration", "positive", "bullish", "momentum",
    "breakout", "recovery", "upbeat", "support", "confidence", "resilience", "earnings"
]
NEGATIVE_WORDS = [
    "slump", "drop", "downgrade", "miss", "weak", "loss", "pressure", "decline",
    "risk", "selloff", "concern", "bearish", "volatility", "uncertainty", "downturn",
    "recession", "liquidity", "headwind", "worry", "setback", "shortfall"
]


def _normalize_ticker(symbol: str) -> str:
    if not symbol:
        return ""
    return symbol.strip().upper().replace(" ", "")


def _cache_path_for_ticker(ticker: str) -> Path:
    safe = _normalize_ticker(ticker).replace(".", "_")
    return NEWS_CACHE_DIR / f"{safe}.json"


def _read_cache(ticker: str):
    path = _cache_path_for_ticker(ticker)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return payload
    except Exception:
        return []
    return []


def _write_cache(ticker: str, articles):
    path = _cache_path_for_ticker(ticker)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(articles, fh, ensure_ascii=False, indent=2)


def _newsapi_articles(ticker: str, max_articles: int):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{ticker} OR {ticker.replace('.NS', '').replace('.BO', '')} stock market finance",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": api_key,
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("articles") or []
        normalized = []
        for item in items:
            if not item.get("title"):
                continue
            normalized.append({
                "title": item.get("title", "").strip(),
                "summary": (item.get("description") or item.get("content") or "").strip(),
                "content": (item.get("content") or item.get("description") or item.get("title") or "").strip(),
                "source": (item.get("source") or {}).get("name", "News API"),
                "published_at": item.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
                "url": item.get("url") or "",
            })
        return normalized[:max_articles]
    except Exception:
        return []


def _fallback_articles(ticker: str, max_articles: int):
    ticker = _normalize_ticker(ticker)
    questions = [
        "earnings beat and strong guidance lift sentiment for",
        "revenue momentum and demand expansion improve outlook for",
        "operating margin gains and share gains support",
        "investors remain cautious amid volatility and concern for",
        "weak demand and pricing pressure weigh on",
        "macro headwinds and competition create risk for",
    ]
    articles = []
    for i, template in enumerate(questions):
        if len(articles) >= max_articles:
            break
        sentiment = "positive" if i % 2 == 0 else "negative" if i % 3 == 0 else "neutral"
        body = (
            f"{ticker} continues to show {sentiment} market momentum. "
            f"Analysts note higher demand, stronger execution, and improved operating leverage. "
            f"The company remains focused on margin discipline and inventory control."
            if sentiment == "positive" else
            f"{ticker} is facing {sentiment} pressure as investors focus on demand weakness, margin risk, and competitive headwinds. "
            f"Analysts highlight lower volume and caution around near-term execution."
            if sentiment == "negative" else
            f"{ticker} remains in a mixed phase with stable cash generation and moderate growth. "
            f"The market is watching margin improvements and any signs of stronger than expected demand."
        )
        articles.append({
            "title": f"{ticker}: market focus shifts as {sentiment} sentiment returns",
            "summary": template.replace("for", f"for {ticker}") + ".",
            "content": body,
            "source": "Local fallback feed",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "url": f"https://example.com/{ticker.lower()}/{i}",
        })
    return articles[:max_articles]


def fetch_ticker_articles(ticker: str, max_articles: int = 10):
    ticker = _normalize_ticker(ticker)
    if not ticker:
        return []

    cached = _read_cache(ticker)
    if cached:
        return cached[:max_articles]

    articles = _newsapi_articles(ticker, max_articles)
    if not articles:
        articles = _fallback_articles(ticker, max_articles)

    _write_cache(ticker, articles)
    return articles[:max_articles]


def clear_news_cache(ticker: str | None = None):
    if ticker is None:
        for file in NEWS_CACHE_DIR.glob("*.json"):
            file.unlink(missing_ok=True)
        return
    _cache_path_for_ticker(ticker).unlink(missing_ok=True)
