"""Conservative, transparent research-time fusion of forecast and context signals.

This is a display heuristic, not a trained or historically validated return model.
Fundamental snapshots are current-only, so they must never be backfilled into a
historical training set as if they were known at old forecast origins.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping

import numpy as np

from src.fundamentals import read_cached_details


_FUSION_WEIGHTS = {"lstm": 0.70, "news": 0.20, "fundamentals": 0.10}
_FUNDAMENTAL_MAX_AGE_HOURS = 72


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded(value: float) -> float:
    return math.tanh(float(value))


def _fresh(details: Mapping[str, Any], now: datetime) -> bool:
    try:
        as_of = datetime.fromisoformat(str(details["As of"]).replace("Z", "+00:00"))
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        age = (now - as_of.astimezone(timezone.utc)).total_seconds()
        return 0 <= age <= _FUNDAMENTAL_MAX_AGE_HOURS * 3600
    except (KeyError, TypeError, ValueError):
        return False


def _metric(details: Mapping[str, Any], key: str) -> float | None:
    metrics = details.get("metrics", {})
    item = metrics.get(key, {}) if isinstance(metrics, Mapping) else {}
    return _number(item.get("value")) if isinstance(item, Mapping) else None


def fundamental_signal(ticker: str, now: datetime | None = None) -> dict[str, Any]:
    """Return a bounded long-term fundamentals tilt from a fresh local snapshot."""
    now = now or datetime.now(timezone.utc)
    ticker = (ticker or "").strip().upper()
    details = read_cached_details(ticker)
    if not details:
        return {"available": False, "score": None, "as_of": None, "components": {},
                "reason": "No cached fundamentals snapshot is available."}
    if not _fresh(details, now):
        return {"available": False, "score": None, "as_of": details.get("As of"), "components": {},
                "reason": "Cached fundamentals are older than 72 hours; refresh before using them."}

    financial_sector = bool(details.get("financial_sector"))
    components: dict[str, float] = {}
    quality = []
    equity = _metric(details, "equity")
    roe = _metric(details, "roe")
    if roe is not None and (equity is None or equity > 0):
        quality.append(_bounded((roe - 12.0) / 20.0))
    if not financial_sector:
        roce = _metric(details, "roce")
        margin = _metric(details, "net_margin")
        if roce is not None:
            quality.append(_bounded((roce - 15.0) / 20.0))
        if margin is not None:
            quality.append(_bounded((margin - 10.0) / 20.0))
    if quality:
        components["business_quality"] = float(np.mean(quality))

    growth = [
        _bounded(value / 20.0)
        for value in (_metric(details, "revenue_growth"), _metric(details, "earnings_growth"))
        if value is not None
    ]
    if growth:
        components["growth"] = float(np.mean(growth))

    if not financial_sector:
        debt_equity = _metric(details, "debt_equity")
        if debt_equity is not None and debt_equity >= 0 and (equity is None or equity > 0):
            components["leverage"] = _bounded(1.0 - max(0.0, debt_equity))

    if not components:
        return {"available": False, "score": None, "as_of": details.get("As of"), "components": {},
                "reason": "The snapshot has no usable, comparable fundamentals for a directional tilt."}
    score = float(np.mean(list(components.values())))
    return {"available": True, "score": max(-1.0, min(1.0, score)),
            "as_of": details.get("As of"), "components": components,
            "reason": "Fresh public fundamentals summarized as a bounded long-term context tilt; not a price target."}


def fuse_forecast(
    current_close: float,
    lstm_prices,
    horizon: int,
    close_history,
    sentiment: Mapping[str, Any] | None,
    fundamentals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Blend point-in-time signals into a cautious research estimate.

    Context signals are converted to a small fraction of the observed horizon
    volatility. Missing signals are omitted and remaining weights are normalized.
    """
    prices = np.asarray(lstm_prices, dtype=float).reshape(-1)
    current = _number(current_close)
    if current is None or current <= 0 or not len(prices) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("A positive current close and finite positive LSTM path are required")
    if isinstance(horizon, bool) or int(horizon) < 1:
        raise ValueError("horizon must be a positive session count")
    horizon = int(horizon)
    lstm_log_return = math.log(float(prices[-1]) / current)
    history = np.asarray(close_history, dtype=float).reshape(-1)
    history = history[np.isfinite(history) & (history > 0)]
    daily_vol = 0.0
    if len(history) > 2:
        daily_vol = float(np.std(np.diff(np.log(history[-61:])), ddof=1))
    horizon_vol = min(0.25, daily_vol * math.sqrt(horizon))

    news_available = bool(
        sentiment
        and sentiment.get("status") == "available"
        and sentiment.get("evidence")
    )
    news_score = _number(sentiment.get("overall_score")) if news_available else None
    if news_score is not None:
        news_score = max(-1.0, min(1.0, news_score))
        synthesis = sentiment.get("llm_synthesis")
        stance_values = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "mixed": 0.0}
        if isinstance(synthesis, Mapping) and synthesis.get("stance") in stance_values:
            # Grounded RAG stance and article sentiment share one bounded news input.
            news_score = 0.5 * news_score + 0.5 * stance_values[synthesis["stance"]]
    fundamental_score = _number(fundamentals.get("score")) if fundamentals and fundamentals.get("available") else None
    candidates = {"lstm": lstm_log_return}
    weights = {"lstm": _FUSION_WEIGHTS["lstm"]}
    if news_score is not None:
        candidates["news"] = news_score * horizon_vol * 0.25
        weights["news"] = _FUSION_WEIGHTS["news"]
    if fundamental_score is not None:
        candidates["fundamentals"] = fundamental_score * horizon_vol * 0.20
        weights["fundamentals"] = _FUSION_WEIGHTS["fundamentals"]

    weight_total = sum(weights.values())
    normalized_weights = {name: value / weight_total for name, value in weights.items()}
    final_log_return = sum(candidates[name] * normalized_weights[name] for name in candidates)
    final_close = current * math.exp(final_log_return)
    adjustment = final_log_return - lstm_log_return
    fractions = np.linspace(1.0 / len(prices), 1.0, len(prices))
    final_prices = prices * np.exp(adjustment * fractions)
    return {
        "status": "heuristic_research_estimate",
        "current_close": current,
        "lstm_predicted_close": float(prices[-1]),
        "final_predicted_close": float(final_close),
        "lstm_log_return": lstm_log_return,
        "final_log_return": final_log_return,
        "final_return": math.exp(final_log_return) - 1.0,
        "final_prices": final_prices,
        "news_score": news_score,
        "news_evidence_count": len(sentiment.get("evidence", [])) if news_available else 0,
        "rag_synthesis": sentiment.get("llm_synthesis") if sentiment else None,
        "fundamental_score": fundamental_score,
        "fundamental_as_of": fundamentals.get("as_of") if fundamentals else None,
        "fundamental_reason": fundamentals.get("reason", "Fundamentals snapshot unavailable.") if fundamentals else "Fundamentals snapshot unavailable.",
        "fundamental_components": fundamentals.get("components", {}) if fundamentals else {},
        "weights": normalized_weights,
        "horizon_volatility": horizon_vol,
        "limitation": "The LSTM, news, and fundamentals blend is a fixed heuristic and has not been validated as more accurate than LSTM or persistence.",
    }
