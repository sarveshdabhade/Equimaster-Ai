from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import CLOSE_COL_IDX, HORIZONS, HORIZON_LABELS, MODEL_STORAGE_PATH_ENV
from src.evaluate_predictions import evaluate_predictions
from src.artifacts import resolve_artifact


def _model_path(ticker: str, horizon: int, model_dir: Path) -> Path:
    suffix = "" if horizon == 1 else f"_h{horizon}"
    for extension in (".keras", ".h5"):
        candidate = model_dir / f"lstm_{ticker}{suffix}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model found for {ticker}, horizon {horizon} in {model_dir}")


def evaluate_lstm(ticker: str, horizon: int, sentiment_scores: list[float] | None = None) -> dict:
    artifact = resolve_artifact(ticker, horizon)
    if not artifact["legacy"]:
        frame = pd.read_csv(artifact["model"].parent / f"backtest_h{horizon}.csv")
        metrics = evaluate_predictions(frame["current_close"], frame["predicted_close"], frame["actual_close"], sentiment_scores)
        baseline = evaluate_predictions(frame["current_close"], frame["current_close"], frame["actual_close"])
        return {"ticker": ticker, "horizon": horizon, "period": HORIZON_LABELS[horizon],
                "version": artifact["metadata"]["version"],
                "model": str(artifact["model"]), **metrics, "baseline": baseline,
                "interval_test_coverage": artifact["evaluation"]["interval"]["test_coverage"]}
    raise ValueError("Legacy model has no verified split contract; retrain before evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing LSTM model on its held-out test set.")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=1,
                        help="Forecast period: 1=Day, 5=Week, 22=Month, 88=Quarter.")
    parser.add_argument("--sentiment-json", help="JSON array of historical sentiment scores aligned to test samples.")
    args = parser.parse_args()
    sentiment_scores = None
    if args.sentiment_json:
        with open(args.sentiment_json, "r", encoding="utf-8") as handle:
            sentiment_scores = [float(value) for value in json.load(handle)]
    print(json.dumps(evaluate_lstm(args.ticker, args.horizon, sentiment_scores), indent=2))


if __name__ == "__main__":
    main()
