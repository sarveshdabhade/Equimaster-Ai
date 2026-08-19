import streamlit as st

from src.market_context import build_market_context
from src.recommendation import build_recommendation
from src.risk_score import build_risk_score


def render_market_overview(ticker):
    if not ticker:
        st.info("Select a ticker to see the unified market overview.")
        return

    context = build_market_context(ticker, horizon=5)
    sentiment = context.get("sentiment", {})
    label = context.get("label", "Neutral")

    color = "#22c55e" if label == "Bullish" else "#ef4444" if label == "Bearish" else "#f59e0b"
    st.markdown(
        f"<div style='padding:18px;border-radius:14px;background:linear-gradient(135deg, rgba(34,197,94,0.08), rgba(15,23,42,0.12)); border:1px solid rgba(255,255,255,0.09);'>"
        f"<div style='font-size:0.8rem;letter-spacing:0.12em;color:#8EA6C4;text-transform:uppercase'>Unified Summary</div>"
        f"<div style='margin-top:8px;font-size:2.1rem;font-weight:800;color:{color}'>{label}</div>"
        f"<div style='margin-top:10px;color:#dbeafe'>{context.get('summary', 'No summary available.')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Technical", f"{context.get('technical_signal', 0.0):+.4f}")
    with col2:
        st.metric("Sentiment", f"{sentiment.get('overall_score', 0.0):+.2f}", delta=sentiment.get('label', 'Neutral'))
    with col3:
        st.metric("Combined", f"{context.get('combined_score', 0.0):+.2f}", delta=context.get('label', 'Neutral'))
    with col4:
        st.metric("Confidence", f"{sentiment.get('confidence', 0.0):.2f}")

    recommendation = build_recommendation(ticker, price_trend=context.get("price_trend", 0.0), horizon=5)
    risk = build_risk_score(ticker, horizon=5)
    st.markdown("---")
    st.subheader("Decision Engine")
    rec_color = "#22c55e" if recommendation["action"] in {"Strong Buy", "Buy"} else "#ef4444" if recommendation["action"] in {"Strong Sell", "Sell"} else "#f59e0b"
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);'>"
        f"<div style='font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;color:#8EA6C4'>Recommendation</div>"
        f"<div style='margin-top:8px;font-size:1.9rem;font-weight:800;color:{rec_color}'>{recommendation['action']}</div>"
        f"<div style='margin-top:8px;color:#dbeafe'>{recommendation['rationale']}</div>"
        f"<div style='margin-top:10px;color:#8EA6C4'>Suggested hold: {recommendation.get('hold_period', 'Watchlist')}</div>"
        f"<div style='margin-top:4px;color:#8EA6C4'>Confidence: {recommendation.get('confidence', 0.0):.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    risk_color = "#ef4444" if risk["level"] == "High" else "#f59e0b" if risk["level"] == "Medium" else "#22c55e"
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);'>"
        f"<div style='font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;color:#8EA6C4'>Risk</div>"
        f"<div style='margin-top:8px;font-size:1.9rem;font-weight:800;color:{risk_color}'>{risk['level']} ({risk['risk']:.2f})</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Drivers")
    drivers = sentiment.get("drivers", [])
    if drivers:
        for driver in drivers:
            st.write(f"• {driver}")
    else:
        st.info("No explicit drivers were extracted for this ticker.")
