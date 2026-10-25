"""Fetch complete public fundamentals for the configured companies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import ALL_DOWNLOAD_TICKERS
from src.artifacts import atomic_json
from src.fundamentals import company_details, read_cached_details, CACHE_HOURS


def refresh(ticker, only_stale=False):
    cached = read_cached_details(ticker)
    if cached and only_stale:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cached["As of"])
        if age.total_seconds() < CACHE_HOURS * 3600:
            return {"ticker": ticker, "ok": True, "reused": True,
                    "available": sum(m["value"] is not None for m in cached["metrics"].values())}
    try:
        details = company_details(ticker, refresh=True)
        fallback = any("Refresh failed" in message for message in details.get("warnings", []))
        return {"ticker": ticker, "ok": not fallback, "reused": fallback,
                "available": sum(m["value"] is not None for m in details["metrics"].values()),
                "parameters": len(details["metrics"]), "statement_sections": list(details["statements"]),
                "as_of": details["As of"], "warnings": details.get("warnings", [])}
    except Exception as exc:
        return {"ticker": ticker, "ok": False, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated symbols; defaults to all configured companies")
    parser.add_argument("--only-stale", action="store_true")
    args = parser.parse_args()
    tickers = args.tickers.split(",") if args.tickers else ALL_DOWNLOAD_TICKERS
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(lambda ticker: refresh(ticker, args.only_stale), tickers):
            results.append(result)
            print(result, flush=True)
    atomic_json(ROOT / "logs" / "fundamentals_refresh.json", {
        "as_of": datetime.now(timezone.utc).isoformat(), "results": results})
    failures = [r for r in results if not r["ok"]]
    print(f"Fundamentals: {len(results) - len(failures)}/{len(results)} companies refreshed or reused")
    if failures:
        raise SystemExit(f"Fundamentals unavailable for {len(failures)} company/companies; see logs/fundamentals_refresh.json")


if __name__ == "__main__":
    main()
