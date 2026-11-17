"""Compact market-index cards. Missing data is never replaced with mock prices."""
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo
import streamlit as st


def render_indices_ticker(indices_data=None):
    if not indices_data:
        st.caption("Index quotes are temporarily unavailable. Stock coverage below uses saved closing prices.")
        return
    cards = []
    for name, data in indices_data.items():
        change = float(data["pct"])
        tone = "positive" if change >= 0 else "negative"
        updated = ""
        try:
            timestamp = datetime.fromisoformat(str(data.get("as_of", "")))
            timestamp = (timestamp.replace(tzinfo=ZoneInfo("Asia/Kolkata")) if timestamp.tzinfo is None
                         else timestamp.astimezone(ZoneInfo("Asia/Kolkata")))
            updated = timestamp.strftime("%d %b · %H:%M IST")
        except (TypeError, ValueError):
            pass
        update_line = f'<div class="index-update">Updated {escape(updated)}</div>' if updated else ""
        cards.append(f'<div class="index-card"><div class="index-name">{escape(name)}</div>'
                     f'<div class="index-value">{float(data["last"]):,.2f}</div>'
                     f'<div class="index-change {tone}">{float(data["delta"]):+,.2f} ({change:+.2f}%)</div>'
                     f'{update_line}</div>')
    st.markdown('<div class="index-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)
