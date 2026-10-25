from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from src.prediction_contract import bounded_signal, directional_accuracy, combined_signal


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("Actual and predicted values must have the same non-zero shape.")
    errors = predicted_values - actual_values
    non_zero_actual = np.where(np.abs(actual_values) > 1e-12, np.abs(actual_values), np.nan)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.nanmean(np.abs(errors) / non_zero_actual) * 100.0),
    }


def signal_metrics(
    current_close: Iterable[float],
    predicted_close: Iterable[float],
    actual_close: Iterable[float],
    sentiment_scores: Iterable[float] | None = None,
) -> dict[str, float]:
    current = np.asarray(list(current_close), dtype=float)
    predicted = np.asarray(list(predicted_close), dtype=float)
    actual = np.asarray(list(actual_close), dtype=float)
    if not (current.shape == predicted.shape == actual.shape) or current.size == 0:
        raise ValueError("Signal inputs must have the same non-zero shape.")

    predicted_returns = (predicted - current) / np.where(current == 0, 1.0, current)
    actual_returns = (actual - current) / np.where(current == 0, 1.0, current)
    lstm_accuracy = float(np.mean([
        directional_accuracy(predicted_return, actual_return)
        for predicted_return, actual_return in zip(predicted_returns, actual_returns)
    ]))

    result = {
        "lstm_directional_accuracy": lstm_accuracy,
        "lstm_return_correlation": float(np.corrcoef(predicted_returns, actual_returns)[0, 1])
        if len(predicted_returns) > 1 and np.std(predicted_returns) > 0 and np.std(actual_returns) > 0 else 0.0,
    }
    if sentiment_scores is not None:
        sentiment = np.asarray(list(sentiment_scores), dtype=float)
        if sentiment.shape != actual_returns.shape:
            raise ValueError("Sentiment scores must match prediction count.")
        combined = np.asarray([combined_signal(value, score) for value, score in zip(predicted_returns, sentiment)])
        result["combined_directional_accuracy"] = float(np.mean([
            directional_accuracy(combined_value, actual_return)
            for combined_value, actual_return in zip(combined, actual_returns)
        ]))
        result["sentiment_return_correlation"] = (
            float(np.corrcoef(sentiment, actual_returns)[0, 1])
            if np.std(sentiment) > 0 and np.std(actual_returns) > 0
            else 0.0
        )
    return result


def evaluate_predictions(
    current_close: Iterable[float],
    predicted_close: Iterable[float],
    actual_close: Iterable[float],
    sentiment_scores: Iterable[float] | None = None,
) -> dict[str, Any]:
    current = list(current_close)
    predicted = list(predicted_close)
    actual = list(actual_close)
    result = regression_metrics(actual, predicted)
    result.update(signal_metrics(current, predicted, actual, sentiment_scores))
    result["sample_count"] = len(actual)
    return result
