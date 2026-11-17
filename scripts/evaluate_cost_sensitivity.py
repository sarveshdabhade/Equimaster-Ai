"""Measure frozen-holdout long-only forecast returns under fee scenarios."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import HORIZONS
from scripts.train_manager import configured_model_tickers
from src.artifacts import resolve_artifact


COST_SCENARIOS_BPS = (0, 10, 25, 50)


def select_non_overlapping_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Select forecasts whose holding periods do not overlap for one ticker."""
    required = {"as_of", "target_date", "current_close", "predicted_close", "actual_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast backtest is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["as_of"] = pd.to_datetime(result["as_of"], errors="coerce")
    result["target_date"] = pd.to_datetime(result["target_date"], errors="coerce")
    for column in ("current_close", "predicted_close", "actual_close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=list(required)).sort_values("as_of")
    result = result[(result["current_close"] > 0) & (result["target_date"] > result["as_of"])]

    selected = []
    next_entry = None
    for index, row in result.iterrows():
        if next_entry is not None and row["as_of"] < next_entry:
            continue
        selected.append(index)
        next_entry = row["target_date"]
    if not selected:
        return result.iloc[0:0].copy()
    return result.loc[selected].reset_index(drop=True)


def _evaluate_trades(decisions: list[dict], cost_bps: int) -> dict:
    cost = cost_bps / 10_000.0
    trades = [item for item in decisions if item["predicted_return"] > cost]
    net = np.asarray([item["actual_return"] - cost for item in trades], dtype=float)
    predicted = np.asarray([item["predicted_return"] for item in trades], dtype=float)
    return {
        "round_trip_cost_bps": cost_bps,
        "opportunities": len(decisions),
        "trades": len(trades),
        "participation_pct": 100.0 * len(trades) / len(decisions) if decisions else 0.0,
        "mean_predicted_return_pct": float(predicted.mean() * 100) if trades else None,
        "mean_realized_net_return_pct": float(net.mean() * 100) if trades else None,
        "median_realized_net_return_pct": float(np.median(net) * 100) if trades else None,
        "profitable_trade_pct": float(np.mean(net > 0) * 100) if trades else None,
        "cash_baseline_return_pct": 0.0,
    }


def evaluate(tickers: list[str]) -> dict:
    decisions = {horizon: [] for horizon in HORIZONS}
    model_counts = {str(horizon): 0 for horizon in HORIZONS}
    failures = []
    for ticker in tickers:
        for horizon in HORIZONS:
            try:
                artifact = resolve_artifact(ticker, horizon, prefer_panel=False)
                if artifact.get("legacy"):
                    raise ValueError("Legacy model has no verified holdout")
                backtest = Path(artifact["model"]).parent / f"backtest_h{horizon}.csv"
                frame = select_non_overlapping_forecasts(pd.read_csv(backtest))
                if frame.empty:
                    raise ValueError("No non-overlapping holdout forecasts")
                model_counts[str(horizon)] += 1
                for row in frame.itertuples(index=False):
                    decisions[horizon].append({
                        "ticker": ticker,
                        "predicted_return": float(row.predicted_close / row.current_close - 1.0),
                        "actual_return": float(row.actual_close / row.current_close - 1.0),
                    })
            except Exception as exc:
                failures.append({"ticker": ticker, "horizon": horizon, "error": str(exc)})

    horizons = {}
    for horizon, rows in decisions.items():
        horizons[str(horizon)] = {
            "models": model_counts[str(horizon)],
            "decision_rows": len(rows),
            "scenarios": [_evaluate_trades(rows, bps) for bps in COST_SCENARIOS_BPS],
        }
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "method": {
            "dataset": "saved purged chronological test rows from active model bundles",
            "trade_rule": "long only when the fixed forecast return exceeds the round-trip cost scenario",
            "position_overlap": "one open holding period per ticker and horizon",
            "weighting": "equal weight per trade; cash earns zero",
            "scenarios_bps_round_trip": list(COST_SCENARIOS_BPS),
            "limitations": [
                "Models are frozen within their holdout; this is not a refitted walk-forward test.",
                "Cost scenarios are sensitivity assumptions, not broker-specific taxes or fees.",
                "No portfolio sizing, market impact, liquidity, dividends, or financing is modeled.",
                "The rule is fixed and not optimized on this holdout; results do not establish a trading edge.",
            ],
        },
        "requested_tickers": tickers,
        "horizons": horizons,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated symbols; defaults to configured stocks plus NSEI")
    parser.add_argument("--output", default="data/evaluation/cost-sensitivity-20261026.json")
    args = parser.parse_args()
    tickers = ([item.strip().upper() for item in args.tickers.split(",") if item.strip()]
               if args.tickers else configured_model_tickers())
    result = evaluate(tickers)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {output}")
    for horizon, entry in result["horizons"].items():
        print(f"h{horizon}: models={entry['models']}, decisions={entry['decision_rows']}")
    if result["failures"]:
        raise RuntimeError(f"Cost sensitivity incomplete: {result['failures']}")


if __name__ == "__main__":
    main()
