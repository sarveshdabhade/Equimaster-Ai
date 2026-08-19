import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_lottieurl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def render_technicals(df: pd.DataFrame, ticker: str):
    if df is None:
        st.warning("Data stream offline. Run data_sequencer.py.")
        return

    col_title, col_tf = st.columns([3, 1])
    with col_title:
        st.markdown(f'<h2 class="section-title"><span class="accent">Price Action & Oscillators</span> - {ticker}</h2>', unsafe_allow_html=True)

    # small animated icon + timeframe control
    with col_tf:
        timeframe = st.radio("Resolution", ["Daily", "Weekly", "Monthly"], horizontal=True, label_visibility="collapsed")

    df_plot = df.copy()
    if timeframe != "Daily":
        tf_rule = "W-FRI" if timeframe == "Weekly" else "ME"
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
        }
        for col in df_plot.columns:
            if col not in agg_dict:
                agg_dict[col] = 'last'
        df_plot = df_plot.resample(tf_rule).agg(agg_dict).dropna()

    df_plot = df_plot.tail(200)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Price',
        increasing_fillcolor="#1EBE6C", decreasing_fillcolor="#E72421",
        increasing_line_color="#0F8B51", decreasing_line_color="#F02926"
    ), row=1, col=1)

    if 'SMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'],
                                 line=dict(color='#2962FF', width=1.5), name='SMA 50'),
                      row=1, col=1)

    macd_col = [col for col in df_plot.columns if col.startswith('MACD_')]
    macdh_col = [col for col in df_plot.columns if col.startswith('MACDh_')]

    if macd_col and macdh_col:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[macd_col[0]],
                                 line=dict(color='#2962FF', width=1.5), name='MACD'),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot[macdh_col[0]],
                             marker_color='gray', name='Histogram'),
                      row=2, col=1)

    rsi_col = [col for col in df_plot.columns if col.startswith('RSI')]
    if rsi_col:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[rsi_col[0]],
                                 line=dict(color='#B289CB', width=1.5), name='RSI'),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF5350", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#26A69A", line_width=1, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        hovermode='x unified',
        showlegend=False
    )

    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="#12121B", gridwidth=1, griddash='dot', showline=False, zeroline=False, row=i, col=1)
        fig.update_yaxes(side='right', showgrid=True, gridcolor="#191925", gridwidth=1, griddash='dot', showline=False, zeroline=False, row=i, col=1)

    # wrap chart in a soft card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # small hover info
    st.markdown('<div class="info-tooltip"><span class="info-text">Hover over candles for OHLCV details. Use resolution selector to resample candles.</span></div>', unsafe_allow_html=True)

