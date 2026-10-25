from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HORIZONS, HORIZON_LABELS, NIFTY_50_TICKERS
from scripts.evaluate_lstm import evaluate_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate horizon-one LSTM models across the NIFTY watchlist.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N tickers.")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                        help="Comma-separated session counts: 1=Day, 5=Week, 22=Month, 88=Quarter.")
    parser.add_argument("--output", default="data/evaluation/watchlist_all_horizons.json")
    args = parser.parse_args()
    results = []
    failures = []
    tickers = [p.name.removesuffix("_processed.csv") for p in sorted((ROOT / "data" / "processed").glob("*_processed.csv"))]
    for ticker in tickers[:args.limit]:
        for horizon in [int(x) for x in args.horizons.split(",")]:
            try:
                result = evaluate_lstm(ticker, horizon=horizon)
                results.append(result)
                print(f"{ticker} {HORIZON_LABELS[horizon]} ({horizon} sessions): accuracy={result['lstm_directional_accuracy']:.3f} mape={result['mape']:.3f}%")
            except Exception as exc:
                failures.append({"ticker": ticker, "horizon": horizon, "error": str(exc)})
                print(f"{ticker} {HORIZON_LABELS.get(horizon, f'h{horizon}')}: FAILED ({exc})")
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    if results:
        average_accuracy = sum(item["lstm_directional_accuracy"] for item in results) / len(results)
        print(f"Evaluated {len(results)} ticker/horizon pairs; mean directional accuracy={average_accuracy:.3f}")
    print(f"Saved {output_path}")
    if failures:
        raise RuntimeError(f"Evaluation failures: {failures}")


if __name__ == "__main__":
    main()
