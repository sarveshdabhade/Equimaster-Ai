"""Refresh verified news and persist timestamped observations."""
from pathlib import Path
import sys
import argparse
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.news.fetcher import fetch_ticker_articles
from config import ALL_DOWNLOAD_TICKERS
from src.artifacts import atomic_json
from datetime import datetime, timezone


def refresh_news_for_tickers(tickers, batch_size=5):
    results = []
    for ticker in tickers:
        try:
            articles = fetch_ticker_articles(ticker, max_articles=8, force_refresh=True)
            result = {"ticker": ticker, "articles": len(articles), "ok": bool(articles)}
        except Exception as exc:
            result = {"ticker": ticker, "articles": 0, "ok": False, "error": str(exc)}
        results.append(result)
        print(result, flush=True)
    atomic_json(ROOT / "logs" / "news_refresh.json", {"as_of": datetime.now(timezone.utc).isoformat(), "results": results})
    return sum(r["ok"] for r in results), sum(not r["ok"] for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?")
    args = parser.parse_args()
    success, errors = refresh_news_for_tickers([args.ticker.upper()] if args.ticker else ["NSEI"] + ALL_DOWNLOAD_TICKERS)
    print(f"News refreshed: {success}; no verified results: {errors}")
    if success == 0:
        raise SystemExit("No verified news returned. Check providers or configure NEWS_API_KEY.")
