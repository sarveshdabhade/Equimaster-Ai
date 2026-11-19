#!/usr/bin/env python3
"""Canonical model training CLI and background progress reporting."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import ALL_DOWNLOAD_TICKERS, HORIZONS, NEWS_MAX_ARTICLES, PIPELINE_VERSION
from src.artifacts import atomic_json, model_directory, sha256
from scripts.refresh_all_news import refresh_news_for_tickers


def write_progress(ticker, data):
    atomic_json(ROOT / "logs" / f"progress_{ticker}.json",
                {**data, "last_updated": datetime.now(timezone.utc).isoformat()})


def configured_model_tickers():
    """Current stock universe plus the NIFTY index model."""
    return list(dict.fromkeys([*ALL_DOWNLOAD_TICKERS, "NSEI"]))


def active_bundle_matches_data(ticker, horizons, data_dir=None, model_root=None):
    """Return whether an active bundle fully matches current data and files."""
    data_dir = Path(data_dir or ROOT / "data" / "processed")
    model_root = Path(model_root or model_directory()).resolve()
    try:
        source = data_dir / f"{ticker}_processed.csv"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        pointer = json.loads((model_root / f"active_{ticker}.json").read_text(encoding="utf-8"))
        bundle = (model_root / pointer["bundle"]).resolve()
        if not bundle.is_relative_to(model_root):
            return False
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("pipeline_version") != PIPELINE_VERSION or metadata.get("data_sha256") != source_hash:
            return False
        required_horizons = {str(int(horizon)) for horizon in horizons}
        if not required_horizons.issubset(metadata.get("horizons", {})):
            return False
        scaler = bundle / "scaler.pkl"
        processed = bundle / "processed.csv"
        if not scaler.is_file() or sha256(scaler) != metadata.get("scaler_sha256"):
            return False
        if not processed.is_file() or sha256(processed) != metadata.get("data_sha256"):
            return False
        for horizon in required_horizons:
            entry = metadata["horizons"][horizon]
            model = bundle / entry["model"]
            if not model.is_file() or sha256(model) != entry.get("model_sha256"):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def process_and_train_ticker(
    ticker, horizons, window_size, epochs, only_missing=False, news_validation=None
):
    from src.training import train_ticker
    try:
        write_progress(ticker, {"status": "training", "step": "fit", "pid": os.getpid(), "message": "Training versioned models"})
        result = train_ticker(
            ticker,
            horizons,
            epochs,
            window_size,
            only_missing,
            news_validation=news_validation,
        )
        write_progress(ticker, {"status": "done", "step": "complete", "message": f"Active version {result['version']}"})
        return {"ticker": ticker, "ok": True}
    except Exception as exc:
        write_progress(ticker, {"status": "error", "step": "exception", "message": str(exc)})
        return {"ticker": ticker, "ok": False, "error": str(exc)}
    except KeyboardInterrupt:
        write_progress(ticker, {"status": "interrupted", "step": "fit", "message": "Training was interrupted; the previous active bundle is retained"})
        raise


def main(workers=1, horizons=None, window_size=60, epochs=60, only_missing=False, tickers=None):
    horizons = HORIZONS if horizons is None else horizons
    tickers = tickers or configured_model_tickers()
    tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    if only_missing:
        tickers = [ticker for ticker in tickers
                   if not active_bundle_matches_data(ticker, horizons)]
        if not tickers:
            skipped = {"as_of": datetime.now(timezone.utc).isoformat(), "state": "complete",
                       "skipped": True, "reason": "All active bundles match current data",
                       "tickers": [], "completed": [], "failures": []}
            atomic_json(ROOT / "logs" / "training_status.json", skipped)
            atomic_json(ROOT / "logs" / "training_news.json", {**skipped, "results": []})
            print("All configured active bundles match the current data; no training or news refresh is needed.", flush=True)
            return
        print(f"Training is needed for {len(tickers)} ticker(s): {', '.join(tickers)}", flush=True)
    failures = []
    news_results = refresh_news_for_tickers(
        tickers, required_articles=NEWS_MAX_ARTICLES
    )
    atomic_json(ROOT / "logs" / "training_news.json", {
        "as_of": datetime.now(timezone.utc).isoformat(), "results": news_results})
    news_by_ticker = {result["ticker"]: result for result in news_results}
    insufficient_news = [
        {
            "ticker": ticker,
            "articles": news_by_ticker.get(ticker, {}).get("articles", 0),
            "error": news_by_ticker.get(ticker, {}).get(
                "error", "No fresh news validation was returned"
            ),
        }
        for ticker in tickers
        if (
            ticker not in news_by_ticker
            or not news_by_ticker[ticker].get("ok")
            or news_by_ticker[ticker].get("articles", 0) < NEWS_MAX_ARTICLES
        )
    ]
    for item in insufficient_news:
        failure = {"ticker": item["ticker"], "ok": False, "error": item["error"],
                   "articles": item["articles"], "blocked_by": "training news gate"}
        failures.append(failure)
        write_progress(item["ticker"], {"status": "blocked", "step": "news_validation",
                                        "message": item["error"]})
        print(failure, flush=True)
    trainable_tickers = [ticker for ticker in tickers if ticker not in {item["ticker"] for item in insufficient_news}]
    news_validation_by_ticker = {
        ticker: {
            "article_count": news_by_ticker[ticker]["articles"],
            "fetched_at": news_by_ticker[ticker]["fetched_at"],
        }
        for ticker in trainable_tickers
    }
    completed = []

    def save_status(state):
        atomic_json(ROOT / "logs" / "training_status.json", {
            "as_of": datetime.now(timezone.utc).isoformat(), "state": state,
            "pid": os.getpid(), "tickers": tickers, "horizons": horizons,
            "epochs": epochs, "workers": workers, "news_results": news_results,
            "completed": completed, "failures": failures})

    save_status("running")
    if workers <= 1:
        results = (
            process_and_train_ticker(
                ticker,
                horizons,
                window_size,
                epochs,
                only_missing,
                news_validation_by_ticker[ticker],
            )
            for ticker in trainable_tickers)
        for result in results:
            completed.append(result)
            if not result["ok"]:
                failures.append(result)
                print(result, flush=True)
            save_status("running")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_and_train_ticker,
                    ticker,
                    horizons,
                    window_size,
                    epochs,
                    only_missing,
                    news_validation_by_ticker[ticker],
                ): ticker
                for ticker in trainable_tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"ticker": ticker, "ok": False, "error": str(exc)}
                if not result["ok"]:
                    failures.append(result)
                    print(result, flush=True)
                completed.append(result)
                save_status("running")
    save_status("failed" if failures else "complete")
    if failures:
        raise RuntimeError(f"Training incomplete; blocked or failed tickers: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="Number of ticker training processes; each uses bounded TensorFlow threads.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                        help="Comma-separated session counts: 1=Day, 5=Week, 22=Month, 63=Quarter.")
    parser.add_argument("--tickers", help="Comma-separated internal symbols; default: configured stocks plus NSEI")
    args = parser.parse_args()
    main(args.workers, [int(x) for x in args.horizons.split(",")], epochs=args.epochs,
         only_missing=args.only_missing, tickers=args.tickers.split(",") if args.tickers else None)
