from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fetcher import fetch_ticker_articles
from .embeddings import EmbeddingRetriever

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Module logger
logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_from_newsapi(ticker: str, api_key: str, max_articles: int = 20, max_attempts: int = 3, timeout: int = 15) -> List[Dict[str, Any]]:
    """Fetch recent articles about ticker from NewsAPI.org with retry/backoff.

    Raises an exception if all attempts fail so callers can fall back.
    """
    q = urllib.parse.quote(f"{ticker} stock OR {ticker} earnings OR {ticker} market")
    params = {
        "q": q,
        "pageSize": str(max_articles),
        "sortBy": "publishedAt",
        "language": "en",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{NEWSAPI_URL}?{query}&apiKey={urllib.parse.quote(api_key)}"

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Equimaster-News-Ingest/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)

            articles: List[Dict[str, Any]] = []
            for a in data.get("articles", []):
                articles.append(
                    {
                        "title": a.get("title") or "",
                        "summary": a.get("description") or "",
                        "content": a.get("content") or "",
                        "source": a.get("source", {}).get("name", "Unknown"),
                        "published_at": a.get("publishedAt"),
                        "url": a.get("url"),
                    }
                )
            logger.debug("Fetched %d articles from NewsAPI for %s (attempt %d)", len(articles), ticker, attempt)
            return articles
        except Exception as exc:
            last_exc = exc
            backoff = (2 ** (attempt - 1)) + (0.1 * attempt)
            logger.warning("NewsAPI fetch attempt %d for %s failed: %s; backing off %.1fs", attempt, ticker, str(exc), backoff)
            time.sleep(backoff)

    logger.error("All NewsAPI fetch attempts failed for %s: %s", ticker, str(last_exc))
    raise last_exc


def _cache_path_for(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"articles_{ticker}.json"


def load_cached_articles(ticker: str, cache_dir: Path) -> List[Dict[str, Any]]:
    p = _cache_path_for(ticker, cache_dir)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.exception("Failed to load cache for %s: %s", ticker, str(exc))
        return []


def save_cached_articles(ticker: str, articles: List[Dict[str, Any]], cache_dir: Path) -> None:
    _ensure_dir(cache_dir)
    p = _cache_path_for(ticker, cache_dir)
    try:
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(articles, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.exception("Failed to save cache for %s: %s", ticker, str(exc))


def dedupe_articles(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_urls = {a.get("url") for a in existing if a.get("url")}
    appended = []
    for a in new:
        if a.get("url") and a["url"] in existing_urls:
            continue
        appended.append(a)
    return existing + appended


def index_articles_for_ticker(ticker: str, articles: List[Dict[str, Any]], persist_dir: Path) -> None:
    # Build texts in the same way the retriever expects: full article text
    texts = []
    for a in articles:
        title = a.get("title", "")
        summary = a.get("summary", "")
        content = a.get("content", "")
        texts.append(" ".join([title, summary, content]).strip())

    if not texts:
        return

    retriever = EmbeddingRetriever(persist_dir=str(persist_dir))
    # Try to load existing index and append only new documents when possible.
    if retriever.load_index():
        existing_count = len(retriever.documents)
        if len(texts) > existing_count:
            new_texts = texts[existing_count:]
            retriever.add_documents(new_texts)
        elif len(texts) < existing_count:
            # If the cached index has more documents than current texts, rebuild to align
            retriever.build_index(texts)
    else:
        retriever.build_index(texts)


def ingest_ticker(
    ticker: str,
    api_key: Optional[str],
    cache_dir: Optional[str] = None,
    persist_dir: Optional[str] = None,
    max_articles: int = 20,
) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    cache_dir_path = Path(cache_dir) if cache_dir else Path.cwd() / ".cache" / "news" / "articles"
    persist_dir_path = Path(persist_dir) if persist_dir else Path.cwd() / ".cache" / "news" / "embeddings"
    _ensure_dir(cache_dir_path)
    now = datetime.utcnow().isoformat() + "Z"

    # 1) fetch (try live provider if api_key present, otherwise fallback to existing fetcher)
    try:
        if api_key:
            articles = fetch_from_newsapi(ticker, api_key, max_articles=max_articles)
        else:
            # fallback to existing fetcher implementation
            articles = fetch_ticker_articles(ticker, max_articles=max_articles)
    except Exception:
        # On any provider failure, fall back to local fetcher
        articles = fetch_ticker_articles(ticker, max_articles=max_articles)

    # 2) load cache and dedupe
    existing = load_cached_articles(ticker, cache_dir_path)
    combined = dedupe_articles(existing, articles)

    # 3) save cache
    save_cached_articles(ticker, combined, cache_dir_path)

    # 4) index/persist embeddings
    try:
        index_articles_for_ticker(ticker, combined, persist_dir_path)
        indexed = True
    except Exception:
        indexed = False

    return {
        "ticker": ticker,
        "fetched": len(articles),
        "cached_total": len(combined),
        "indexed": indexed,
        "timestamp": now,
    }


def run_scheduler(
    tickers: List[str],
    interval_seconds: int = 3600,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    persist_dir: Optional[str] = None,
    max_articles: int = 20,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run a simple blocking scheduler that ingests articles for tickers every interval_seconds.

    This function blocks until stop_event is set (if provided) or KeyboardInterrupt.
    """
    cache_dir_path = Path(cache_dir) if cache_dir else None
    persist_dir_path = Path(persist_dir) if persist_dir else None

    print(f"Starting news ingest scheduler for: {', '.join(tickers)} (interval {interval_seconds}s)")
    try:
        while True:
            for t in tickers:
                res = ingest_ticker(t, api_key, cache_dir=str(cache_dir_path) if cache_dir_path else None, persist_dir=str(persist_dir_path) if persist_dir_path else None, max_articles=max_articles)
                print(f"[{datetime.utcnow().isoformat()}] Ingest result for {t}: fetched={res['fetched']} cached_total={res['cached_total']} indexed={res['indexed']}")
            if stop_event and stop_event.is_set():
                break
            # sleep with interruption
            for _ in range(int(interval_seconds)):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(1)
            if stop_event and stop_event.is_set():
                break
    except KeyboardInterrupt:
        print("Scheduler interrupted, exiting.")
