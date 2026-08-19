from __future__ import annotations

from typing import Any, Dict

from src.market_context import build_market_context


def build_recommendation(ticker: str, price_trend: float = 0.0, horizon: int = 5) -> Dict[str, Any]:
    """Translate combined market context into a simple buy/hold/sell recommendation."""
    context = build_market_context(ticker, horizon=horizon)
    sentiment_score = float(context.get("sentiment", {}).get("overall_score", 0.0))
    technical_signal = float(context.get("technical_signal", 0.0))
    combined = float(context.get("combined_score", 0.0))
    confidence = float(context.get("sentiment", {}).get("confidence", 0.0))
    score = combined + (price_trend * 1.5) + (sentiment_score * 0.5)

    if score >= 0.25:
        action = "Strong Buy"
        rationale = "Momentum and sentiment are aligned in the bullish direction."
        hold_period = "2-4 weeks"
    elif score >= 0.05:
        action = "Buy"
        rationale = "The market has a mild positive bias with supportive sentiment."
        hold_period = "1-3 weeks"
    elif score <= -0.25:
        action = "Strong Sell"
        rationale = "Momentum and sentiment are both deteriorating and risk is elevated."
        hold_period = "Avoid entry / reduce exposure"
    elif score <= -0.05:
        action = "Sell"
        rationale = "The setup shows a weak technical and sentiment backdrop."
        hold_period = "1-2 weeks"
    else:
        action = "Hold"
        rationale = "Market conditions are mixed; wait for stronger confirmation."
        hold_period = "1 week or wait for confirmation"

    if confidence < 0.55:
        hold_period = "Short-term watchlist only"

    return {
        "ticker": (ticker or "").upper(),
        "action": action,
        "score": round(score, 4),
        "combined_score": round(combined, 4),
        "technical_signal": round(technical_signal, 4),
        "sentiment_score": round(sentiment_score, 4),
        "price_trend": round(price_trend, 4),
        "confidence": round(confidence, 4),
        "hold_period": hold_period,
        "rationale": rationale,
        "context": context,
    }
