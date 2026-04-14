import streamlit as st


def render_news_tab(ticker):
    st.markdown('<h2 class="section-title"><span class="accent">FinBERT</span> NLP Analysis</h2>', unsafe_allow_html=True)
    st.info("Coming Soon: This module will scrape the top 10 news headlines for the selected ticker and use the Hugging Face FinBERT model to calculate a real-time polarity score (Positive/Negative sentiment).")
    st.progress(0)
    # subtle UI hint
    st.markdown('<div class="info-tooltip">ⓘ<span class="info-text">News sentiment runs on demand to avoid API rate limits — click to analyze when ready.</span></div>', unsafe_allow_html=True)
