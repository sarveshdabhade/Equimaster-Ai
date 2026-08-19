import pandas as pd
import streamlit as st

from src.market_context import build_market_context
from src.news.sentiment import analyze_ticker_sentiment


def render_news_tab(ticker):
    st.markdown('<h2 class="section-title"><span class="accent">RAG</span> Sentiment Analysis</h2>', unsafe_allow_html=True)

    if not ticker:
        st.info("Select a ticker to fetch and analyze market sentiment.")
        return

    with st.spinner(f"Building the market context pipeline for {ticker}..."):
        market_context = build_market_context(ticker, horizon=5)
        sentiment_result = analyze_ticker_sentiment(ticker, max_articles=8)

    score = sentiment_result.get("overall_score", 0.0)
    label = sentiment_result.get("label", "Neutral")
    confidence = sentiment_result.get("confidence", 0.0)
    evidence = sentiment_result.get("evidence", [])
    drivers = sentiment_result.get("drivers", [])

    color = "#22c55e" if label == "Bullish" else "#ef4444" if label == "Bearish" else "#f59e0b"
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;background:linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.08);'>"
        f"<div style='font-size:0.8rem;letter-spacing:0.14em;color:#8EA6C4;text-transform:uppercase'>Market Context</div>"
        f"<div style='margin-top:8px;font-size:2.3rem;font-weight:800;color:{color}'>{label}</div>"
        f"<div style='margin-top:8px;color:#dbeafe'>Sentiment score: <strong>{score:.2f}</strong> · Confidence: <strong>{confidence:.2f}</strong> · Combined signal: <strong>{market_context.get('combined_score', 0.0):+.2f}</strong></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 1.8])
    with col_a:
        st.metric("Price Trend", f"{market_context.get('price_trend', 0.0):+.4f}")
        st.metric("Technical Signal", f"{market_context.get('technical_signal', 0.0):+.4f}")
        st.metric("Sentiment Signal", f"{score:+.2f}", delta=label)
        st.metric("Confidence", f"{confidence:.2f}")
        st.caption(market_context.get("summary", "No summary available."))

        if drivers:
            st.markdown("**Key drivers**")
            for driver in drivers:
                st.markdown(f"- {driver}")

    with col_b:
        if evidence:
            df = pd.DataFrame(evidence)
            st.dataframe(
                df[["title", "source", "score", "relevance"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("Headline"),
                    "source": st.column_config.TextColumn("Source"),
                    "score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
                    "relevance": st.column_config.NumberColumn("Relevance", format="%.3f"),
                },
            )
        else:
            st.info("No relevant news evidence was found for this ticker.")

    st.markdown("---")
    st.markdown("### Retrieved Evidence")
    if evidence:
        for item in evidence[:5]:
            with st.expander(f"{item.get('title', 'News item')} — {item.get('source', 'Source')}", expanded=False):
                st.write(item.get("summary", "No summary."))
                if item.get("url"):
                    st.markdown(f"[Open source]({item['url']})")
                st.caption(
                    f"Article score: {item.get('score', 0.0):.3f} | Retrieval relevance: {item.get('relevance', 0.0):.3f}"
                )
    else:
        st.info("No article-level evidence is currently available.")

    st.caption(sentiment_result.get("summary", "Architecture: data ingestion → price trend → retrieval → sentiment scoring → combined signal."))
