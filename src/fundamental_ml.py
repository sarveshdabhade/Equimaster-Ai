"""Walk-forward ML for 12-month stock excess return versus NIFTY 50."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from config import PROJECT_ROOT
from src.artifacts import atomic_json, canonical_ticker
from src.fundamental_history import SNAPSHOT_FEATURES, load_snapshots

logger = logging.getLogger(__name__)
HORIZON_SESSIONS = 252
MIN_TRAIN_SAMPLES = 300
MIN_HISTORY_MONTHS = 30
MODEL_DIR = PROJECT_ROOT / "models" / "fundamental_ml"
MODEL_PATH = MODEL_DIR / "fundamental_12m_excess.joblib"
STATUS_PATH = PROJECT_ROOT / "logs" / "fundamental_ml_status.json"
FEATURE_COLUMNS = [*SNAPSHOT_FEATURES, "financial_sector"]


def _prices(ticker: str) -> pd.Series:
    path = PROJECT_ROOT / "data" / "processed" / f"{canonical_ticker(ticker)}_processed.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        frame = pd.read_csv(path, parse_dates=["Date"])
        column = "Adj Close" if "Adj Close" in frame else "Close"
        values = pd.to_numeric(frame[column], errors="coerce")
        series = pd.Series(values.to_numpy(), index=pd.to_datetime(frame["Date"]).dt.tz_localize(None))
        return series[~series.index.duplicated(keep="last")].dropna().sort_index()
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Could not load price history for fundamental ML: %s", ticker)
        return pd.Series(dtype=float)


def _starting_index(series: pd.Series, date: pd.Timestamp) -> int:
    return int(series.index.searchsorted(date, side="right") - 1)


def _labeled_samples() -> list[dict[str, Any]]:
    snapshots = load_snapshots()
    if not snapshots:
        return []
    tickers = {str(snapshot["ticker"]) for snapshot in snapshots}
    prices = {ticker: _prices(ticker) for ticker in tickers}
    benchmark = _prices("NSEI")
    if benchmark.empty:
        return []
    result = []
    for snapshot in snapshots:
        ticker_prices = prices.get(str(snapshot["ticker"]))
        if ticker_prices is None or ticker_prices.empty:
            continue
        try:
            observed = pd.Timestamp(snapshot["observed_at"]).tz_convert(None)
        except (TypeError, ValueError):
            observed = pd.Timestamp(snapshot["observed_at"])
        stock_start = _starting_index(ticker_prices, observed)
        benchmark_start = _starting_index(benchmark, observed)
        stock_end = stock_start + HORIZON_SESSIONS
        benchmark_end = benchmark_start + HORIZON_SESSIONS
        if stock_start < 0 or benchmark_start < 0 or stock_end >= len(ticker_prices) or benchmark_end >= len(benchmark):
            continue
        stock_base, stock_final = float(ticker_prices.iloc[stock_start]), float(ticker_prices.iloc[stock_end])
        bench_base, bench_final = float(benchmark.iloc[benchmark_start]), float(benchmark.iloc[benchmark_end])
        if min(stock_base, stock_final, bench_base, bench_final) <= 0:
            continue
        stock_return = stock_final / stock_base - 1.0
        benchmark_return = bench_final / bench_base - 1.0
        features = {
            key: snapshot["features"].get(key)
            for key in SNAPSHOT_FEATURES
        }
        features["financial_sector"] = float(snapshot.get("financial_sector", False))
        result.append({
            "ticker": snapshot["ticker"],
            "snapshot_month": snapshot["snapshot_month"],
            "observed_at": snapshot["observed_at"],
            "label_end": str(ticker_prices.index[stock_end].date()),
            "features": features,
            "target": stock_return - benchmark_return,
        })
    return sorted(result, key=lambda item: (item["snapshot_month"], item["ticker"]))


def _pipeline():
    estimator = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=18,
        l2_regularization=4.0,
        random_state=41,
    )
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)


def _walk_forward_predictions(samples: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(samples)
    months = sorted(frame["snapshot_month"].unique())
    if len(months) < MIN_HISTORY_MONTHS:
        return {"passed": False, "reason": f"Need at least {MIN_HISTORY_MONTHS} matured snapshot months; have {len(months)}."}
    if len(samples) < MIN_TRAIN_SAMPLES:
        return {"passed": False, "reason": f"Need at least {MIN_TRAIN_SAMPLES} matured samples; have {len(samples)}."}

    # Three expanding-window validations. Training labels must end before the
    # validation window begins, purging overlapping 12-month outcomes.
    block = max(2, len(months) // 10)
    starts = [int(len(months) * ratio) for ratio in (0.55, 0.70, 0.85)]
    actual, predicted, baseline = [], [], []
    fold_metrics = []
    for start_idx in starts:
        start_idx = min(start_idx, len(months) - block)
        validation_start = months[start_idx]
        validation_end = months[min(len(months), start_idx + block) - 1]
        start_timestamp = pd.Timestamp(f"{validation_start}-01")
        train = frame[pd.to_datetime(frame["label_end"]) < start_timestamp]
        test = frame[frame["snapshot_month"].between(validation_start, validation_end)]
        if len(train) < 80 or len(test) < 20:
            continue
        x_train = pd.DataFrame(list(train["features"]), columns=FEATURE_COLUMNS)
        x_test = pd.DataFrame(list(test["features"]), columns=FEATURE_COLUMNS)
        model = _pipeline().fit(x_train, train["target"].to_numpy(dtype=float))
        estimates = model.predict(x_test)
        targets = test["target"].to_numpy(dtype=float)
        actual.extend(targets.tolist())
        predicted.extend(estimates.tolist())
        baseline.extend(np.zeros_like(targets).tolist())
        fold_metrics.append({
            "validation_start": validation_start,
            "validation_end": validation_end,
            "train_samples": int(len(train)),
            "validation_samples": int(len(test)),
            "model_mae": float(np.mean(np.abs(estimates - targets))),
            "zero_alpha_mae": float(np.mean(np.abs(targets))),
        })
    if len(fold_metrics) < 3:
        return {"passed": False, "reason": "Three purged, out-of-time validation folds are not available yet.", "folds": fold_metrics}
    model_mae = float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))
    baseline_mae = float(np.mean(np.abs(np.asarray(baseline) - np.asarray(actual))))
    correlation = spearmanr(actual, predicted).statistic
    correlation = float(correlation) if np.isfinite(correlation) else 0.0
    passed = model_mae < baseline_mae and correlation > 0.0
    return {
        "passed": passed,
        "reason": "Out-of-time checks passed." if passed else "The model did not beat the zero-excess-return baseline with positive rank correlation.",
        "folds": fold_metrics,
        "model_mae": model_mae,
        "zero_alpha_mae": baseline_mae,
        "spearman_rank_correlation": correlation,
    }


def train_fundamental_model() -> dict[str, Any]:
    """Train only when matured point-in-time snapshots pass purged time checks."""
    archived = load_snapshots()
    samples = _labeled_samples()
    months = sorted({sample["snapshot_month"] for sample in samples})
    status: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "model": "HistGradientBoostingRegressor",
        "target": "252-session stock return minus NIFTY 50 return",
        "samples": len(samples),
        "matured_months": len(months),
        "snapshots_archived": len(archived),
        "archived_months": len({snapshot["snapshot_month"] for snapshot in archived}),
        "feature_count": len(FEATURE_COLUMNS),
        "validation_passed": False,
        "ready": False,
    }
    if not samples:
        status["reason"] = "No 12-month labels have matured from archived point-in-time fundamentals yet."
        atomic_json(STATUS_PATH, status)
        return status
    validation = _walk_forward_predictions(samples)
    status["validation"] = validation
    status["reason"] = validation["reason"]
    if validation.get("passed"):
        frame = pd.DataFrame(samples)
        features = pd.DataFrame(list(frame["features"]), columns=FEATURE_COLUMNS)
        model = _pipeline().fit(features, frame["target"].to_numpy(dtype=float))
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        status["ready"] = True
        status["validation_passed"] = True
        status["samples"] = len(samples)
        status["sample_start"] = months[0]
        status["sample_end"] = months[-1]
        status["model_path"] = str(MODEL_PATH.relative_to(PROJECT_ROOT))
    atomic_json(STATUS_PATH, status)
    return status


def read_model_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        snapshots = load_snapshots()
        return {"ready": False, "validation_passed": False,
                "snapshots_archived": len(snapshots),
                "archived_months": len({snapshot["snapshot_month"] for snapshot in snapshots}),
                "reason": "The point-in-time ML dataset is being collected."}
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ready": False, "validation_passed": False,
                "reason": "ML readiness status could not be read."}
    if not status.get("validation_passed") or not MODEL_PATH.exists():
        status["ready"] = False
    return status


def predict_excess_return(features: Mapping[str, Any]) -> dict[str, Any]:
    status = read_model_status()
    if not status.get("ready"):
        return {"available": False, "expected_excess_return": None,
                "reason": status.get("reason", "The validated ML model is not ready."),
                "training_status": status}
    try:
        model = joblib.load(MODEL_PATH)
        row = {key: features.get(key) for key in SNAPSHOT_FEATURES}
        row["financial_sector"] = float(bool(features.get("financial_sector", False)))
        estimate = float(model.predict(pd.DataFrame([row], columns=FEATURE_COLUMNS))[0])
        return {"available": True, "expected_excess_return": estimate,
                "reason": "Validated walk-forward model estimate; not a guaranteed return.",
                "training_status": status}
    except Exception as exc:
        logger.exception("Fundamental ML prediction failed")
        return {"available": False, "expected_excess_return": None,
                "reason": f"Validated model prediction failed ({type(exc).__name__}).",
                "training_status": status}
