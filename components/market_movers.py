"""Navigable gainers and losers from the application's actual stock coverage."""
from html import escape
import pandas as pd
import streamlit as st
from components.layout import section_title
from src.dashboard_data import load_dashboard_rows, data_fingerprint


def fetch_market_movers():
    frame = load_dashboard_rows(data_fingerprint())
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    return (frame[frame["Change %"] > 0].nlargest(5, "Change %"),
            frame[frame["Change %"] < 0].nsmallest(5, "Change %"))


def render_market_movers(frame=None):
    if frame is None:
        gainers, losers = fetch_market_movers()
    elif frame.empty:
        gainers, losers = pd.DataFrame(), pd.DataFrame()
    else:
        gainers = frame[frame["Change %"] > 0].nlargest(5, "Change %")
        losers = frame[frame["Change %"] < 0].nsmallest(5, "Change %")
    left, right = st.columns(2)
    for column, title, values, key in ((left, "Top gainers", gainers, "gainers_panel"), (right, "Top losers", losers, "losers_panel")):
        with column, st.container(border=True, key=key):
            section_title(title, "Last session · covered companies", f"{len(values)} stocks")
            if values.empty:
                st.caption("No matching companies in the current coverage.")
            for _, row in values.iterrows():
                name, price, change = st.columns([2.6, 1.1, 1.1])
                name.page_link("pages/stock_analyzer.py", label=str(row["Symbol"]), query_params={"ticker": row["Ticker"]})
                name.caption(str(row["Company"])[:32])
                price.markdown(f'<div class="mover-price">{row["Price"]:,.2f}</div>', unsafe_allow_html=True)
                tone = "positive" if row["Change %"] > 0 else "negative"
                change.markdown(f'<div class="mover-change {tone}">{row["Change %"]:+.2f}%</div>', unsafe_allow_html=True)
