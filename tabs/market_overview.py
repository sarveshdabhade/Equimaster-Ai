"""Readable technical/news context, separate from measured forecast performance."""
import streamlit as st
from components.layout import section_title
from src.market_context import build_market_context
from src.recommendation import build_recommendation
from src.risk_score import build_risk_score


def render_market_overview(ticker):
    section_title("Market summary", "Technical and news heuristics · separate from LSTM backtests")
    if not ticker:
        st.info("Select a company to see market context.")
        return
    context = build_market_context(ticker, horizon=5)
    sentiment = context.get("sentiment", {})
    has_evidence = bool(sentiment.get("evidence"))
    metrics = st.columns(4)
    metrics[0].metric("Technical momentum", f"{context.get('technical_signal', 0):+.3f}")
    metrics[1].metric("News sentiment", f"{sentiment.get('overall_score', 0):+.2f}" if has_evidence else "Unavailable")
    metrics[2].metric("Combined context", f"{context.get('combined_score', 0):+.2f}" if has_evidence else "N/A")
    metrics[3].metric("Evidence articles", len(sentiment.get("evidence", [])))
    if not has_evidence:
        st.info("News sentiment and combined context are unavailable because no eligible recent news evidence was found.")
    recommendation = build_recommendation(ticker, context.get("price_trend", 0), 5, context=context)
    risk = build_risk_score(ticker, 5, context=context)
    left, right = st.columns([1.6, 1])
    with left, st.container(border=True):
        section_title(
            "Research interpretation",
            badge=context.get("label", "Neutral") if has_evidence else "News unavailable",
        )
        st.subheader(recommendation["action"])
        st.write(recommendation["rationale"])
        st.caption(f"Suggested review window: {recommendation['hold_period']}. This is a heuristic label, not a measured expected return.")
    with right, st.container(border=True):
        section_title("Risk context")
        st.metric("Heuristic risk", risk["level"], f"{risk['risk']:.2f}", delta_color="off")
        st.caption("Risk combines technical context, sentiment and evidence uncertainty; it is not a calibrated loss probability.")
    with st.container(border=True):
        section_title("What is driving the view?")
        for driver in sentiment.get("drivers", []):
            st.write(f"• {driver}")
        if not sentiment.get("drivers"):
            st.info("No recent news drivers are available for this company.")
