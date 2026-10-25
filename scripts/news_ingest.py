"""Poll Yahoo and configured publisher feeds for NIFTY constituents.

Usage examples:
  # Poll configured NIFTY symbols every 15 minutes; public RSS needs no API key
  python scripts/news_ingest.py

  # Optionally enable NewsAPI using a key stored in the local process environment

  # Configure tickers and interval with environment variables:
  # NEWS_TICKERS=TCS.NS,INFY.NS INTERVAL_SECONDS=900 python scripts/news_ingest.py
"""

import logging
import os
from threading import Event
from config import NIFTY_50_TICKERS

from src.news.ingest import run_scheduler


def _env_list(key: str, default: str = "AAPL,MSFT,GOOGL"):
    raw = os.environ.get(key, default)
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def main():
    # configure logging early so ingest module logs are visible
    log_level = os.environ.get("NEWS_INGEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = os.environ.get("NEWS_API_KEY")
    tickers = _env_list("NEWS_TICKERS", ",".join(NIFTY_50_TICKERS))
    try:
        interval = int(os.environ.get("INTERVAL_SECONDS", "900"))
    except Exception:
        interval = 3600

    cache_dir = os.environ.get("NEWS_CACHE_DIR")
    persist_dir = os.environ.get("NEWS_PERSIST_DIR")

    stop = Event()
    run_scheduler(tickers=tickers, interval_seconds=interval, api_key=api_key, cache_dir=cache_dir, persist_dir=persist_dir, stop_event=stop)


if __name__ == "__main__":
    main()
