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
from config import NEWS_MAX_ARTICLES
from src.artifacts import atomic_json
from datetime import datetime, timezone


def refresh_news_for_tickers(tickers, batch_size=5, required_articles=NEWS_MAX_ARTICLES, report_name="news_refresh.json"):
    results = []
    for index, ticker in enumerate(tickers):
        try:
            articles = fetch_ticker_articles(
                ticker,
                max_articles=required_articles,
                force_refresh=True,
                refresh_shared_sources=index == 0,
                require_live=True,
            )
            result = {
                "ticker": ticker,
                "articles": len(articles),
                "ok": len(articles) >= required_articles,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            if not result["ok"]:
                result["error"] = (
                    f"Expected {required_articles} fresh verified articles; "
                    f"received {len(articles)}"
                )
        except Exception as exc:
            result = {
                "ticker": ticker,
                "articles": 0,
                "ok": False,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        results.append(result)
        print(result, flush=True)
    atomic_json(ROOT / "logs" / report_name, {"as_of": datetime.now(timezone.utc).isoformat(), "results": results})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Record unavailable coverage without returning a failing exit code")
    args = parser.parse_args()
    results = refresh_news_for_tickers([args.ticker.upper()] if args.ticker else ALL_DOWNLOAD_TICKERS)
    success = sum(result["ok"] for result in results)
    errors = sum(not result["ok"] for result in results)
    print(f"News refreshed: {success}; no verified results: {errors}")
    if errors and not args.allow_incomplete:
        raise SystemExit(
            f"News refresh incomplete for {errors} ticker(s); training requires "
            f"{NEWS_MAX_ARTICLES} fresh verified articles for every ticker."
        )
