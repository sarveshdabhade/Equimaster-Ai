"""Run this script to start an ingestion scheduler for news articles.

Usage examples:
  # run with default tickers and interval (no NewsAPI key -> fallback fetcher used)
  python scripts/news_ingest.py

  # run with NewsAPI key (from env)
  NEWSAPI_KEY=xxx python scripts/news_ingest.py

  # configure tickers and interval via env
  NEWS_TICKERS=AAPL,MSFT,GOOGL INTERVAL_SECONDS=1800 python scripts/news_ingest.py
"""

import logging
import os
from threading import Event

from src.news.ingest import run_scheduler


def _env_list(key: str, default: str = "AAPL,MSFT,GOOGL"):
    raw = os.environ.get(key, default)
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def main():
    # configure logging early so ingest module logs are visible
    log_level = os.environ.get("NEWS_INGEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = os.environ.get("NEWSAPI_KEY")
    tickers = _env_list("NEWS_TICKERS")
    try:
        interval = int(os.environ.get("INTERVAL_SECONDS", "3600"))
    except Exception:
        interval = 3600

    cache_dir = os.environ.get("NEWS_CACHE_DIR")
    persist_dir = os.environ.get("NEWS_PERSIST_DIR")

    stop = Event()
    run_scheduler(tickers=tickers, interval_seconds=interval, api_key=api_key, cache_dir=cache_dir, persist_dir=persist_dir, stop_event=stop)


if __name__ == "__main__":
    main()
