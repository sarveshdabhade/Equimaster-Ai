"""Compact market-index cards. Missing data is never replaced with mock prices."""
from html import escape
import streamlit as st


def render_indices_ticker(indices_data=None):
    if not indices_data:
        st.caption("Index quotes are temporarily unavailable. Stock coverage below uses saved closing prices.")
        return
    cards = []
    for name, data in indices_data.items():
        change = float(data["pct"])
        tone = "positive" if change >= 0 else "negative"
        cards.append(f'<div class="index-card"><div class="index-name">{escape(name)}</div>'
                     f'<div class="index-value">{float(data["last"]):,.2f}</div>'
                     f'<div class="index-change {tone}">{float(data["delta"]):+,.2f} ({change:+.2f}%)</div></div>')
    st.markdown('<div class="index-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)
