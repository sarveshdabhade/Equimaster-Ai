"""Compact evidence-led news research view."""
from datetime import datetime
import os
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from components.layout import section_title
from src.market_context import build_market_context
from src.news.sentiment import analyze_ticker_sentiment
from src.news.fetcher import fetch_ticker_articles


def _format_published_at(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        stamp = stamp.replace(tzinfo=ZoneInfo("UTC")) if stamp.tzinfo is None else stamp
        return stamp.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y · %I:%M %p IST")
    except (TypeError, ValueError):
        return "Publication time unavailable"


def render_news_tab(ticker):
    heading, refresh = st.columns([4, 1])
    with heading:
        section_title("News & sentiment", "Yahoo Finance, Google News, publisher feeds · dates retained")
    if refresh.button("Refresh feed", width="stretch"):
        with st.spinner("Checking live news providers for this company..."):
            articles = fetch_ticker_articles(ticker, max_articles=10, force_refresh=True)
        if articles:
            st.success(f"Refreshed {len(articles)} recent verified articles.")
        else:
            st.warning("Refresh completed, but no fresh verified articles matched this company.")
    with st.spinner("Reading recent company evidence..."):
        result = analyze_ticker_sentiment(ticker, max_articles=10, include_llm=True)
        context = build_market_context(ticker, horizon=5, sentiment=result)
    evidence = result.get("evidence", [])
    synthesis_status = result.get("llm_synthesis_status", "not_requested")
    synthesis = result.get("llm_synthesis")
    if synthesis_status == "error":
        st.error(f"LLM evidence synthesis failed: {result.get('llm_synthesis_error', 'Unknown provider error')}")
    elif synthesis_status == "disabled":
        st.caption("LLM synthesis is disabled; the displayed sentiment uses the existing heuristic/FinBERT path.")
    elif synthesis_status == "no_retrieved_evidence":
        st.info("LLM synthesis was not run because retrieval returned no eligible evidence.")
    if synthesis:
        with st.container(border=True):
            st.markdown(f"**LLM evidence synthesis · {synthesis['stance'].capitalize()}**")
            st.write(synthesis["summary"])
            st.caption("This synthesis is separate from the sentiment score and forecast signal.")
            summary_sources = [
                synthesis["citation_sources"].get(evidence_id, {}).get("title") or evidence_id
                for evidence_id in synthesis["summary_evidence_ids"]
            ]
            st.caption(f"Summary sources: {', '.join(summary_sources)}")
            for claim in synthesis["claims"]:
                st.write(f"• {claim['text']}")
                source_labels = [
                    synthesis["citation_sources"].get(evidence_id, {}).get("title") or evidence_id
                    for evidence_id in claim["evidence_ids"]
                ]
                st.caption(f"Sources: {', '.join(source_labels)}")
            source_map = synthesis.get("citation_sources", {})
            if source_map:
                with st.expander("LLM evidence citations"):
                    for index, (evidence_id, source) in enumerate(source_map.items(), start=1):
                        st.markdown(f"**{source.get('source', 'Source')}** · {source.get('title') or evidence_id}")
                        url = source.get("url", "")
                        if str(url).startswith(("https://", "http://")):
                            st.link_button(
                                f"Read cited source {index}", url
                            )
    columns = st.columns(4)
    has_evidence = bool(evidence)
    columns[0].metric("News sentiment", result.get("label", "Neutral") if has_evidence else "Unavailable")
    columns[1].metric("Sentiment score", f"{result.get('overall_score', 0):+.2f}" if has_evidence else "N/A")
    columns[2].metric("Verified articles", len(evidence))
    columns[3].metric("Combined context", f"{context.get('combined_score', 0):+.2f}" if has_evidence else "N/A")
    st.caption("Confidence and impact are screening heuristics, not calibrated probabilities. Publication dates show article freshness.")
    if not evidence:
        providers = (
            "Yahoo Finance, Google News (trusted publishers including India Shipping News and ET Infra), "
            "Economic Times, CNBC, "
            "Business Standard, LiveMint, and Hindu BusinessLine"
        )
        if not os.getenv("NEWS_API_KEY"):
            providers += "; NewsAPI.org is not enabled (set NEWS_API_KEY in the app's environment for additional coverage)"
        st.info(
            f"Sentiment is unavailable because no fresh, verified company-matched news was found in the recent-news window from {providers}. "
            "Publisher feeds do not cover every company or publish a story every day, so this cannot guarantee "
            "an article for every ticker. Use Refresh feed to query providers again."
        )
        return
    frame = pd.DataFrame(evidence)
    visible = [key for key in ("title", "source", "sentiment", "score", "relevance_level", "impact_level") if key in frame]
    st.dataframe(frame[visible], hide_index=True, width="stretch")
    st.download_button("Export news evidence", frame.to_csv(index=False), f"{ticker}_news.csv", "text/csv")
    for article in evidence:
        with st.container(border=True):
            st.caption(f"{article.get('source', 'Source')} · {_format_published_at(article.get('published_at'))}")
            st.markdown(f"**{article.get('title', 'Company update')}**")
            st.write(article.get("summary") or "No summary supplied by the publisher.")
            meta = f"{article.get('sentiment', 'neutral').capitalize()} · {article.get('relevance_level', 'Unknown')} relevance · {article.get('impact_level', 'Unknown')} heuristic impact"
            st.caption(meta)
            if article.get("text"):
                with st.expander("Evidence details"):
                    st.write(article["text"])
            url = str(article.get("url") or "")
            if url.startswith(("https://", "http://")):
                st.link_button("Read source", url)
