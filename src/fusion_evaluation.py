"""Chronological fixed-rule baselines for point-in-time fusion datasets."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluate_predictions import regression_metrics
from src.fusion_dataset import validate_fusion_dataset
from src.prediction_contract import combined_signal, directional_accuracy

LLM_STANCE_SCORE = {
    "bearish": -1.0,
    "bullish": 1.0,
    "mixed": 0.0,
    "neutral": 0.0,
}
_METRIC_KEYS = (
    "persistence_mae",
    "persistence_rmse",
    "persistence_mape",
    "lstm_mae",
    "lstm_rmse",
    "lstm_mape",
    "persistence_directional_accuracy",
    "lstm_directional_accuracy",
    "news_combined_directional_accuracy",
    "llm_stance_directional_accuracy",
    "llm_combined_directional_accuracy",
)


def _evaluate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows.sort(key=lambda row: row["as_of"])
    current = np.asarray([row["current_close"] for row in rows], dtype=float)
    lstm = np.asarray([row["lstm_predicted_close"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_close"] for row in rows], dtype=float)
    lstm_returns = (lstm - current) / current
    actual_returns = (actual - current) / current
    news_scores = [row["news_sentiment_score"] for row in rows]
    llm_scores = [LLM_STANCE_SCORE[row["llm_stance"]] for row in rows]
    persistence_metrics = regression_metrics(actual, current)
    lstm_metrics = regression_metrics(actual, lstm)

    metrics = {
        **{f"persistence_{key}": value for key, value in persistence_metrics.items()},
        **{f"lstm_{key}": value for key, value in lstm_metrics.items()},
        "persistence_directional_accuracy": float(np.mean([
            directional_accuracy(0.0, target_return) for target_return in actual_returns
        ])),
        "lstm_directional_accuracy": float(np.mean([
            directional_accuracy(predicted_return, target_return)
            for predicted_return, target_return in zip(lstm_returns, actual_returns)
        ])),
        "news_combined_directional_accuracy": float(np.mean([
            directional_accuracy(
                combined_signal(predicted_return, sentiment_score), target_return
            )
            for predicted_return, sentiment_score, target_return
            in zip(lstm_returns, news_scores, actual_returns)
        ])),
        "llm_stance_directional_accuracy": float(np.mean([
            directional_accuracy(score, target_return)
            for score, target_return in zip(llm_scores, actual_returns)
        ])),
        "llm_combined_directional_accuracy": float(np.mean([
            directional_accuracy(
                combined_signal(predicted_return, llm_score), target_return
            )
            for predicted_return, llm_score, target_return
            in zip(lstm_returns, llm_scores, actual_returns)
        ])),
    }
    metrics["news_combined_delta_vs_lstm"] = (
        metrics["news_combined_directional_accuracy"] - metrics["lstm_directional_accuracy"]
    )
    metrics["llm_combined_delta_vs_lstm"] = (
        metrics["llm_combined_directional_accuracy"] - metrics["lstm_directional_accuracy"]
    )
    return {
        "ticker": rows[0]["ticker"],
        "horizon": rows[0]["horizon"],
        "sample_count": len(rows),
        "as_of_start": rows[0]["as_of"],
        "as_of_end": rows[-1]["as_of"],
        **metrics,
    }


def evaluate_fusion_baselines(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare fixed baselines in chronological ticker/horizon groups."""
    normalized = validate_fusion_dataset(records)
    grouped = defaultdict(list)
    for row in normalized:
        grouped[(row["ticker"], row["horizon"])].append(row)
    per_group = [
        _evaluate_group(rows)
        for _, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]
    macro_average = {
        key: float(np.mean([group[key] for group in per_group]))
        for key in _METRIC_KEYS
    }
    by_horizon_groups = defaultdict(list)
    for group in per_group:
        by_horizon_groups[group["horizon"]].append(group)
    macro_by_horizon = {
        horizon: {
            key: float(np.mean([group[key] for group in groups]))
            for key in _METRIC_KEYS
        }
        for horizon, groups in sorted(by_horizon_groups.items())
    }
    return {
        "evaluation_scope": "chronological_fixed_rule_baselines",
        "sample_count": len(normalized),
        "group_count": len(per_group),
        "per_ticker_horizon": per_group,
        "macro_average": macro_average,
        "macro_by_horizon": macro_by_horizon,
        "llm_stance_mapping": dict(LLM_STANCE_SCORE),
        "limitation": (
            "These predeclared, untrained rules are diagnostic baselines on the "
            "supplied rows, not out-of-sample proof of improved forecast accuracy."
        ),
    }
