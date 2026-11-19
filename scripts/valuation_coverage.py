"""Export a reproducible valuation estimate and input record for every configured stock.

The default run only reads saved fundamentals. Use --refresh-missing to fetch a
snapshot for a stock that has no compatible local cache. No news or return-model
prediction is needed to calculate these valuation estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_DOWNLOAD_TICKERS, resolve_app_path
from src.artifacts import atomic_json, canonical_ticker
from src.fundamental_swot import calculate_fundamental_valuation, load_valuation_assumptions
from src.fundamentals import CACHE_HOURS, company_details, read_cached_details


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _valuation_errors(estimate: Mapping[str, Any]) -> list[str]:
    errors = []
    if not estimate.get("available"):
        errors.append("No supported valuation estimate was returned.")
    point, low, high = (_finite_number(estimate.get(key)) for key in ("value", "low", "high"))
    if point is None or point <= 0:
        errors.append("Valuation must be finite and positive.")
    if low is None or high is None or low <= 0 or high <= 0:
        errors.append("Both scenario bounds must be finite and positive.")
    elif point is not None and not low <= point <= high:
        errors.append("The point estimate must lie inside its reported scenario range.")
    return errors


def _snapshot_age_hours(details: Mapping[str, Any]) -> float | None:
    try:
        value = datetime.fromisoformat(str(details.get("As of")).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - value).total_seconds() / 3600, 2)
    except (TypeError, ValueError):
        return None


def _input_records(details: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the available financial inputs and their provenance reviewable."""
    keys = (
        "price", "market_cap", "shares", "eps", "forward_eps", "book_value", "roe",
        "revenue", "profit", "ebitda", "net_debt", "cash", "debt", "cfo", "capex",
        "net_borrowing", "fcf", "fcfe", "pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales",
    )
    result = []
    metrics = details.get("metrics", {})
    for key in keys:
        metric = metrics.get(key, {})
        if not isinstance(metric, Mapping):
            continue
        value = _finite_number(metric.get("value"))
        if value is None:
            continue
        result.append({"metric": key, "value": value, **{
            field: metric.get(field) for field in ("source", "period", "basis", "currency")
        }})
    return result


def build_coverage_report(tickers: list[str], *, refresh_missing: bool = False) -> dict[str, Any]:
    """Calculate every requested stock against the same cached comparison universe."""
    tickers = list(dict.fromkeys(canonical_ticker(ticker.strip()) for ticker in tickers if ticker.strip()))
    universe = {}
    cache_directory = Path(resolve_app_path("data", "fundamentals"))
    cached_tickers = {path.stem for path in cache_directory.glob("*.json")}
    for ticker in sorted(cached_tickers | set(tickers)):
        details = read_cached_details(ticker)
        if details:
            universe[ticker] = details

    fetch_errors = {}
    if refresh_missing:
        for ticker in tickers:
            if ticker not in universe:
                try:
                    universe[ticker] = company_details(ticker, refresh=True)
                except Exception as exc:
                    fetch_errors[ticker] = f"{type(exc).__name__}: {exc}"

    results = []
    for ticker in tickers:
        details = universe.get(ticker)
        if details is None:
            results.append({
                "ticker": ticker, "company": ticker, "available": False,
                "value": None, "low": None, "high": None, "coverage": "unavailable",
                "confidence": "low", "method": "No company snapshot", "currency": None,
                "missing_inputs": ["Compatible company snapshot"],
                "validation_errors": [fetch_errors.get(ticker, "No compatible local fundamentals snapshot.")],
            })
            continue
        try:
            beta = _finite_number(details.get("metrics", {}).get("beta", {}).get("value"))
            assumptions = load_valuation_assumptions(ticker, beta)
            context = calculate_fundamental_valuation(details, None, assumptions, valuation_universe=universe)
            estimate = {**context["valuation"], "dcf": context["dcf"],
                        "peer_value": context["peer_value"], "valuation_parameters": assumptions}
            for method_key in ("dcf", "peer_value"):
                method = context[method_key]
                if not method.get("available") and method.get("reason"):
                    estimate["limitations"].append(f"{method_key}: {method['reason']}")
        except Exception as exc:
            estimate = {
                "available": False, "value": None, "low": None, "high": None,
                "coverage": "unavailable", "confidence": "low", "method": "Calculation failed",
                "missing_inputs": [], "limitations": [f"{type(exc).__name__}: {exc}"],
            }
        errors = _valuation_errors(estimate)
        age = _snapshot_age_hours(details)
        if age is None or age > CACHE_HOURS:
            errors.append("Fundamentals snapshot is stale; refresh before using this estimate.")
        price = _finite_number(details.get("metrics", {}).get("price", {}).get("value"))
        point = _finite_number(estimate.get("value"))
        results.append({
            **estimate,
            "ticker": ticker, "company": details.get("Company") or ticker,
            "sector": details.get("Sector"), "industry": details.get("Industry"),
            "currency": estimate.get("currency") or details.get("Currency"),
            "current_price": price,
            "upside": point / price - 1 if point is not None and price is not None and price > 0 else None,
            "snapshot_as_of": details.get("As of"), "cache_age_hours": age,
            "stale_snapshot": age is None or age > CACHE_HOURS,
            "inputs": _input_records(details), "validation_errors": errors,
        })

    successful = [row for row in results if not row["validation_errors"]]
    failed = [row["ticker"] for row in results if row["validation_errors"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Configured company stocks" if tickers == ALL_DOWNLOAD_TICKERS else "Requested company stocks",
        "range_kind": "Scenario dispersion, not a confidence interval or validated forecast",
        "summary": {
            "requested": len(tickers), "accounted_for": len(results), "valued": len(successful), "failed": len(failed),
            "complete": bool(tickers) and not failed, "failed_tickers": failed,
            "comparison_universe": len(universe),
            "methods": dict(sorted(Counter(row.get("method", "Unknown") for row in successful).items())),
            "confidence": dict(sorted(Counter(row.get("confidence", "Unknown") for row in successful).items())),
            "coverage": dict(sorted(Counter(row.get("coverage", "Unknown") for row in successful).items())),
            "stale_snapshots": sum(bool(row.get("stale_snapshot")) for row in results),
        },
        "results": results,
    }


def write_report(report: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    atomic_json(json_path, dict(report))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "ticker", "company", "sector", "industry", "currency", "current_price", "value", "low", "high",
        "upside", "method", "confidence", "coverage", "peer_scope", "peer_count", "basis",
        "snapshot_as_of", "cache_age_hours", "stale_snapshot", "range_kind", "assumptions",
        "limitations", "missing_inputs", "components", "inputs", "dcf", "peer_value", "valuation_parameters", "validation_errors",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in report["results"]:
            row = {key: result.get(key) for key in columns}
            for key, value in row.items():
                if isinstance(value, (list, dict, tuple)):
                    row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated symbols; defaults to every configured stock")
    parser.add_argument("--refresh-missing", action="store_true", help="Fetch only stocks without a compatible saved snapshot")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "valuation_coverage.json")
    parser.add_argument("--csv-output", type=Path, help="Defaults to the JSON output path with a .csv suffix")
    args = parser.parse_args(argv)
    tickers = args.tickers.split(",") if args.tickers else list(ALL_DOWNLOAD_TICKERS)
    report = build_coverage_report(tickers, refresh_missing=args.refresh_missing)
    csv_path = args.csv_output or args.output.with_suffix(".csv")
    write_report(report, args.output, csv_path)
    summary = report["summary"]
    print(f"Valuation coverage: {summary['valued']}/{summary['requested']} stocks; {summary['failed']} failures")
    print("Confidence:", json.dumps(summary["confidence"], sort_keys=True))
    print("Methods:", json.dumps(summary["methods"], sort_keys=True))
    if summary["failed_tickers"]:
        print("Failed tickers:", ", ".join(summary["failed_tickers"]))
    print(f"JSON: {args.output}")
    print(f"CSV: {csv_path}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
