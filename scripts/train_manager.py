#!/usr/bin/env python3
"""Canonical model training CLI and background progress reporting."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import HORIZONS
from src.artifacts import atomic_json


def write_progress(ticker, data):
    atomic_json(ROOT / "logs" / f"progress_{ticker}.json",
                {**data, "last_updated": datetime.now(timezone.utc).isoformat()})


def process_and_train_ticker(ticker, horizons, window_size, epochs, only_missing=False):
    from src.training import train_ticker
    try:
        write_progress(ticker, {"status": "training", "step": "fit", "message": "Training versioned models"})
        result = train_ticker(ticker, horizons, epochs, window_size, only_missing)
        write_progress(ticker, {"status": "done", "step": "complete", "message": f"Active version {result['version']}"})
        return {"ticker": ticker, "ok": True}
    except Exception as exc:
        write_progress(ticker, {"status": "error", "step": "exception", "message": str(exc)})
        return {"ticker": ticker, "ok": False, "error": str(exc)}


def main(workers=1, horizons=None, window_size=60, epochs=10, only_missing=False, tickers=None):
    horizons = HORIZONS if horizons is None else horizons
    tickers = tickers or [p.name.removesuffix("_processed.csv") for p in sorted((ROOT / "data" / "processed").glob("*_processed.csv"))]
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_and_train_ticker, t, horizons, window_size, epochs, only_missing): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"ticker": ticker, "ok": False, "error": str(exc)}
            if not result["ok"]:
                failures.append(result)
                print(result, flush=True)
    if failures:
        raise RuntimeError(f"Training failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                        help="Comma-separated session counts: 1=Day, 5=Week, 22=Month, 88=Quarter.")
    parser.add_argument("--tickers", help="Comma-separated internal symbols; default: all processed datasets")
    args = parser.parse_args()
    main(args.workers, [int(x) for x in args.horizons.split(",")], epochs=args.epochs,
         only_missing=args.only_missing, tickers=args.tickers.split(",") if args.tickers else None)
