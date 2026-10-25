"""A first train-only Ridge model for point-in-time LSTM/news fusion."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HORIZON_CHOICES_DESCRIPTION, HORIZONS
from src.evaluate_predictions import regression_metrics
from src.fusion_dataset import (
    FUSION_DATASET_SCHEMA_VERSION,
    normalize_as_of,
    normalize_target_date,
    validate_fusion_dataset,
)
from src.fusion_evaluation import LLM_STANCE_SCORE
from src.prediction_contract import directional_accuracy

FEATURE_NAMES = (
    "lstm_log_return",
    "news_sentiment_score",
    "llm_stance_score",
    "log_horizon",
)
MIN_TRAIN_SAMPLES = 8


def _price_from_log_return(log_returns: np.ndarray, current: np.ndarray) -> np.ndarray:
    log_prices = np.log(current) + log_returns
    max_log_price = math.log(np.finfo(float).max)
    min_log_price = math.log(np.nextafter(0.0, 1.0))
    if (
        not np.isfinite(log_returns).all()
        or not np.isfinite(log_prices).all()
        or np.any(log_prices > max_log_price)
        or np.any(log_prices < min_log_price)
    ):
        raise ValueError("Fusion model produced non-finite or unrepresentable forecast returns")
    prices = np.exp(log_prices)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Fusion model produced invalid forecast prices")
    return prices


def _features(rows: list[dict[str, Any]]) -> np.ndarray:
    current = np.asarray([row["current_close"] for row in rows], dtype=float)
    predicted = np.asarray([row["lstm_predicted_close"] for row in rows], dtype=float)
    return np.column_stack((
        np.log(predicted) - np.log(current),
        [row["news_sentiment_score"] for row in rows],
        [LLM_STANCE_SCORE[row["llm_stance"]] for row in rows],
        np.log1p([row["horizon"] for row in rows]),
    ))


def _request_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def _checked_splits(split: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(split, Mapping):
        raise ValueError("A chronological split mapping is required")
    normalized = {}
    for name in ("train", "validation", "test"):
        if name not in split:
            raise ValueError(f"Chronological split is missing {name}")
        normalized[name] = validate_fusion_dataset(split[name])

    train_ids = {(row["ticker"], row["as_of"], row["horizon"]) for row in normalized["train"]}
    validation_ids = {(row["ticker"], row["as_of"], row["horizon"]) for row in normalized["validation"]}
    test_ids = {(row["ticker"], row["as_of"], row["horizon"]) for row in normalized["test"]}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Train, validation, and test partitions must not overlap")

    validation_start = min(normalize_as_of(row["as_of"]) for row in normalized["validation"])
    test_start = min(normalize_as_of(row["as_of"]) for row in normalized["test"])
    train_end = max(normalize_as_of(row["as_of"]) for row in normalized["train"])
    validation_end = max(normalize_as_of(row["as_of"]) for row in normalized["validation"])
    if train_end >= validation_start or validation_end >= test_start:
        raise ValueError("Fusion partitions must be strictly chronological")
    if any(normalize_target_date(row["target_date"]) > validation_start.date() for row in normalized["train"]):
        raise ValueError("Training labels cross the validation origin boundary")
    if any(normalize_target_date(row["target_date"]) > test_start.date() for row in normalized["validation"]):
        raise ValueError("Validation labels cross the test origin boundary")
    return normalized


def _evaluate_partition(rows: list[dict[str, Any]], fused_prices: np.ndarray) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row, fused_price in zip(rows, fused_prices):
        grouped[(row["ticker"], row["horizon"])].append((row, float(fused_price)))

    per_group = []
    for (ticker, horizon), pairs in sorted(grouped.items(), key=lambda item: item[0]):
        actual = np.asarray([row["actual_close"] for row, _ in pairs], dtype=float)
        current = np.asarray([row["current_close"] for row, _ in pairs], dtype=float)
        lstm = np.asarray([row["lstm_predicted_close"] for row, _ in pairs], dtype=float)
        fused = np.asarray([prediction for _, prediction in pairs], dtype=float)
        actual_log_returns = np.log(actual) - np.log(current)
        lstm_log_returns = np.log(lstm) - np.log(current)
        fused_log_returns = np.log(fused) - np.log(current)
        persistence_metrics = regression_metrics(actual, current)
        lstm_metrics = regression_metrics(actual, lstm)
        fused_metrics = regression_metrics(actual, fused)
        per_group.append({
            "ticker": ticker,
            "horizon": horizon,
            "sample_count": len(pairs),
            **{f"persistence_{key}": value for key, value in persistence_metrics.items()},
            **{f"lstm_{key}": value for key, value in lstm_metrics.items()},
            **{f"fused_{key}": value for key, value in fused_metrics.items()},
            "persistence_directional_accuracy": float(np.mean([
                directional_accuracy(0.0, value) for value in actual_log_returns
            ])),
            "lstm_directional_accuracy": float(np.mean([
                directional_accuracy(predicted, target)
                for predicted, target in zip(lstm_log_returns, actual_log_returns)
            ])),
            "fused_directional_accuracy": float(np.mean([
                directional_accuracy(predicted, target)
                for predicted, target in zip(fused_log_returns, actual_log_returns)
            ])),
        })
    numeric_metrics = [
        key for key, value in per_group[0].items()
        if isinstance(value, (int, float)) and key not in {"horizon", "sample_count"}
    ]
    return {
        "sample_count": len(rows),
        "group_count": len(per_group),
        "per_ticker_horizon": per_group,
        "macro_average": {
            key: float(np.mean([group[key] for group in per_group]))
            for key in numeric_metrics
        },
    }


def fit_fusion_model(
    split: Mapping[str, Any],
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit fixed-feature Ridge regression on train only and score validation/test."""
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("Ridge alpha must be positive and finite")
    partitions = _checked_splits(split)
    train = partitions["train"]
    if len(train) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"At least {MIN_TRAIN_SAMPLES} purged training samples are required")

    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    train_features = _features(train)
    train_targets = (
        np.log(np.asarray([row["actual_close"] for row in train], dtype=float))
        - np.log(np.asarray([row["current_close"] for row in train], dtype=float))
    )
    model.fit(train_features, train_targets)

    def evaluate(rows):
        predicted_log_returns = np.asarray(model.predict(_features(rows)), dtype=float)
        current = np.asarray([row["current_close"] for row in rows], dtype=float)
        predicted_prices = _price_from_log_return(predicted_log_returns, current)
        return _evaluate_partition(rows, predicted_prices)

    validation_metrics = evaluate(partitions["validation"])
    test_metrics = evaluate(partitions["test"])
    return {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "alpha": float(alpha),
        "training_sample_count": len(train),
        "validation": validation_metrics,
        "test": test_metrics,
        "limitation": (
            "Ridge fusion is an initial diagnostic model. Use validation only for "
            "selection and preserve the test partition for final evaluation; no "
            "accuracy improvement is assumed."
        ),
    }


def predict_fused_forecast(
    artifact: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Run experimental fusion inference with timestamp and model-version guards."""
    if not isinstance(artifact, Mapping) or not isinstance(request, Mapping):
        raise ValueError("Fusion artifact and request must be mappings")
    model = artifact.get("model")
    metadata = artifact.get("metadata")
    if not callable(getattr(model, "predict", None)) or not isinstance(metadata, Mapping):
        raise ValueError("A loaded fusion artifact with metadata is required")
    if (
        type(metadata.get("artifact_schema_version")) is not int
        or metadata.get("artifact_schema_version") != 1
        or type(metadata.get("fusion_dataset_schema_version")) is not int
        or metadata.get("fusion_dataset_schema_version") != FUSION_DATASET_SCHEMA_VERSION
        or metadata.get("feature_names") != list(FEATURE_NAMES)
        or metadata.get("llm_stance_mapping") != LLM_STANCE_SCORE
    ):
        raise ValueError("Fusion artifact feature contract is incompatible")

    required = {
        "ticker", "as_of", "horizon", "current_close", "lstm_predicted_close",
        "news_sentiment_score", "llm_stance", "llm_evidence_ids",
        "llm_model_id", "lstm_model_id", "evidence",
    }
    missing = required - set(request)
    if missing:
        raise ValueError(f"Fusion inference request is missing fields: {', '.join(sorted(missing))}")
    ticker = str(request["ticker"] or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    horizon = request["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of: {HORIZON_CHOICES_DESCRIPTION}")
    as_of = normalize_as_of(request["as_of"])
    current_close = _request_number(request["current_close"], "current_close", positive=True)
    lstm_predicted_close = _request_number(
        request["lstm_predicted_close"], "lstm_predicted_close", positive=True
    )
    news_score = _request_number(request["news_sentiment_score"], "news_sentiment_score")
    if not -1.0 <= news_score <= 1.0:
        raise ValueError("news_sentiment_score must be in the range -1..1")
    stance = request["llm_stance"]
    if not isinstance(stance, str) or stance not in LLM_STANCE_SCORE:
        raise ValueError("llm_stance must be bullish, bearish, neutral, or mixed")

    evidence = request["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("At least one timestamped evidence record is required")
    evidence_ids = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("Each evidence record must be an object")
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in evidence_ids:
            raise ValueError("Evidence IDs must be non-empty and unique")
        if normalize_as_of(item.get("published_at")) > as_of or normalize_as_of(item.get("observed_at")) > as_of:
            raise ValueError(f"Evidence {evidence_id} is after the requested forecast origin")
        evidence_ids.add(evidence_id)
    cited_ids = request["llm_evidence_ids"]
    if (
        not isinstance(cited_ids, list)
        or not cited_ids
        or any(not isinstance(item, str) or not item for item in cited_ids)
        or len(set(cited_ids)) != len(cited_ids)
        or not set(cited_ids).issubset(evidence_ids)
    ):
        raise ValueError("llm_evidence_ids must uniquely reference supplied evidence records")

    model_ids_by_split = metadata.get("model_ids_by_split")
    if not isinstance(model_ids_by_split, Mapping) or not isinstance(model_ids_by_split.get("train"), Mapping):
        raise ValueError("Fusion artifact is missing training model provenance")
    training_models = model_ids_by_split["train"].get(ticker)
    if not isinstance(training_models, Mapping):
        raise ValueError(f"No fusion training samples are registered for {ticker}")
    llm_model_id = str(request["llm_model_id"] or "").strip()
    lstm_model_id = str(request["lstm_model_id"] or "").strip()
    if llm_model_id not in training_models.get("llm", []):
        raise ValueError(f"LLM model {llm_model_id} was not present in fusion training for {ticker}")
    if lstm_model_id not in training_models.get("lstm", []):
        raise ValueError(f"LSTM model {lstm_model_id} was not present in fusion training for {ticker}")

    row = {
        "current_close": current_close,
        "lstm_predicted_close": lstm_predicted_close,
        "news_sentiment_score": news_score,
        "llm_stance": stance,
        "horizon": horizon,
    }
    model_output = np.asarray(model.predict(_features([row])), dtype=float).reshape(-1)
    if model_output.size != 1:
        raise ValueError("Fusion model must return exactly one log-return prediction")
    predicted_log_return = float(model_output[0])
    predicted_close = float(_price_from_log_return(
        np.asarray([predicted_log_return]), np.asarray([current_close])
    )[0])
    return {
        "status": "experimental_not_activated",
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "horizon": horizon,
        "current_close": current_close,
        "lstm_predicted_close": lstm_predicted_close,
        "fused_predicted_close": predicted_close,
        "predicted_log_return": predicted_log_return,
        "llm_stance": stance,
        "llm_evidence_ids": list(cited_ids),
        "llm_model_id": llm_model_id,
        "lstm_model_id": lstm_model_id,
        "fusion_model_version": metadata.get("version"),
    }
