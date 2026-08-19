from __future__ import annotations

from typing import Any, Dict

from src.market_context import build_market_context


def build_risk_score(ticker: str, horizon: int = 5) -> Dict[str, Any]:
    """Translate sentiment and trend volatility into a simplified risk score."""
    context = build_market_context(ticker, horizon=horizon)
    sentiment = float(context.get("sentiment", {}).get("overall_score", 0.0))
    technical = float(context.get("technical_signal", 0.0))
    confidence = float(context.get("sentiment", {}).get("confidence", 0.0))

    risk = abs(technical) * 0.6 + (1.0 - abs(sentiment)) * 0.2 + (1.0 - confidence) * 0.2
    risk = max(0.0, min(1.0, round(risk, 4)))

    if risk >= 0.7:
        level = "High"
    elif risk >= 0.4:
        level = "Medium"
    else:
        level = "Low"

    return {
        "ticker": (ticker or "").upper(),
        "risk": risk,
        "level": level,
        "technical_component": round(abs(technical), 4),
        "sentiment_component": round(1.0 - abs(sentiment), 4),
        "confidence_component": round(1.0 - confidence, 4),
        "context": context,
    }
