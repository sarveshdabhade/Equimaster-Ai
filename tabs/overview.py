"""Price overview and company facts presented in compact research panels."""
from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.layout import section_title, CHART_STYLE, CHART_TEMPLATE
from src.artifacts import canonical_ticker
from src.fundamentals import read_cached_details, format_value


def render_overview_tab(df, ticker):
    details = read_cached_details(canonical_ticker(ticker)) or {}
    metrics = details.get("metrics", {})
    chart, facts = st.columns([2, 1])
    with chart:
        section_title("Price performance", "Historical closing prices · INR" if canonical_ticker(ticker) != "NSEI" else "Historical closing levels · points")
        period, style = st.columns([2, 1])
        selected = period.radio("Price history", ["1M", "3M", "6M", "1Y", "MAX"], index=2, horizontal=True, label_visibility="collapsed")
        chart_type = style.radio("Chart style", ["Line", "Candles"], horizontal=True, label_visibility="collapsed")
        window = {"1M": 22, "3M": 66, "6M": 126, "1Y": 252, "MAX": len(df)}[selected]
        history = df.tail(window)
        figure = go.Figure()
        if chart_type == "Candles":
            figure.add_trace(go.Candlestick(x=history.index, open=history["Open"], high=history["High"], low=history["Low"], close=history["Close"],
                                           increasing_line_color="#00D47E", decreasing_line_color="#FF7187", name="OHLC"))
        else:
            close = history["Close"]
            period_change = float(close.iloc[-1]) / float(close.iloc[0]) - 1 if close.iloc[0] else 0.0
            chart_color, fill_color = (
                ("#00D47E", "rgba(0,212,126,.10)")
                if period_change >= 0
                else ("#FF7187", "rgba(255,113,135,.10)")
            )
            figure.add_trace(go.Scatter(x=history.index, y=history["Close"], mode="lines", name="Close",
                                       line={"color": chart_color, "width": 2}, fill="tozeroy", fillcolor=fill_color))
        low, high = float(history["Low"].min()), float(history["High"].max())
        figure.update_layout(template=CHART_TEMPLATE, **CHART_STYLE, height=360, margin={"l": 0, "r": 8, "t": 15, "b": 0},
                             xaxis={"showgrid": False, "rangeslider_visible": False},
                             yaxis={"side": "right", "range": [low * .99, high * 1.01]}, showlegend=False, hovermode="x unified")
        st.plotly_chart(figure, width="stretch", theme=None, config={"displayModeBar": False})
        st.download_button("Export price history", history.to_csv(), f"{ticker}_price_history.csv", "text/csv")
    with facts:
        section_title("Key facts", "Published values · source dates in Fundamentals")
        for key in ("market_cap", "pe", "pb", "eps", "dividend_yield", "debt_equity"):
            metric = metrics.get(key, {})
            label = metric.get("label") or {"market_cap": "Market cap", "pe": "P/E", "pb": "P/B", "eps": "EPS", "dividend_yield": "Dividend yield", "debt_equity": "Debt / equity"}[key]
            value = format_value(metric.get("value"), metric.get("unit", "multiple"), metric.get("currency") or "INR")
            st.markdown(f'<div class="company-fact"><span class="muted">{escape(label)}</span><strong>{escape(value)}</strong></div>', unsafe_allow_html=True)
        history = df.tail(252)
        low = metrics.get("low_52w", {}).get("value") or float(history["Low"].min())
        high = metrics.get("high_52w", {}).get("value") or float(history["High"].max())
        price = float(df["Close"].iloc[-1])
        position = max(0, min(100, (price - low) / (high - low) * 100)) if high > low else 50
        section_title("52-week range")
        st.markdown(f'<div class="range-track"><span class="range-dot" style="left:{position}%"></span></div>'
                    f'<div class="range-labels"><span>{low:,.2f}<br>Low</span><span>{high:,.2f}<br>High</span></div>', unsafe_allow_html=True)
    st.divider()
    section_title("Returns at a glance", "Calculated from saved closing prices; not annualized")
    columns = st.columns(5)
    for column, label, sessions in zip(columns, ("1 week", "1 month", "3 months", "6 months", "1 year"), (5, 21, 63, 126, 252)):
        value = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-sessions - 1]) - 1) * 100 if len(df) > sessions and df["Close"].iloc[-sessions - 1] else None
        column.metric(label, format_value(value, "percent"))
    if details.get("Industry"):
        st.caption(f"{details.get('Sector') or 'Sector unavailable'} · {details['Industry']} · {len(df):,} historical observations")
