from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math


@dataclass(frozen=True)
class PredictionRecord:
    ticker: str
    as_of: str
    horizon: int
    current_close: float
    predicted_close: float
    predicted_return: float
    lstm_signal: float
    sentiment_score: float
    sentiment_label: str
    sentiment_confidence: float
    combined_score: float
    combined_label: str
    evidence_count: int = 0
    actual_close: float | None = None

    def with_actual(self, actual_close: float) -> "PredictionRecord":
        return PredictionRecord(**{**asdict(self), "actual_close": float(actual_close)})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bounded_signal(value: float) -> float:
    """Convert any model score into the shared -1..+1 signal range."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Signal must be finite")
    return max(-1.0, min(1.0, value))


def forecast_signal(predicted_return: float) -> float:
    return bounded_signal(math.tanh(float(predicted_return) * 10.0))


def combined_signal(predicted_return: float, sentiment: float) -> float:
    return 0.6 * forecast_signal(predicted_return) + 0.4 * bounded_signal(sentiment)


def classify_signal(score: float, threshold: float = 0.05) -> str:
    score = bounded_signal(score)
    if score > threshold:
        return "Bullish"
    if score < -threshold:
        return "Bearish"
    return "Neutral"


def directional_accuracy(predicted_return: float, actual_return: float) -> float:
    """Return 1.0 for a matching direction, otherwise 0.0."""
    predicted = bounded_signal(predicted_return)
    actual = bounded_signal(actual_return)
    if predicted == 0 or actual == 0:
        return float(predicted == actual)
    return float((predicted > 0) == (actual > 0))
