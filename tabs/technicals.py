import streamlit as st
from components.layout import CHART_STYLE, CHART_GRID, CHART_TEMPLATE
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

    # Native resolution control keeps keyboard and screen-reader support.
    with col_tf:
        timeframe = st.radio("Resolution", ["Daily", "Weekly", "Monthly"], horizontal=True, label_visibility="collapsed")

    dates = st.slider("Chart date range", min_value=df.index.min().date(),
                      max_value=df.index.max().date(),
                      value=(df.index[max(0, len(df) - 200)].date(), df.index.max().date()),
                      key=f"technical_dates_{ticker}")
    df_plot = df.loc[str(dates[0]):str(dates[1])].copy()
    if df_plot.empty:
        st.info("No prices in this date range.")
        return
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
                agg_dict[col] = 'sum' if col == 'Volume' else 'last'
        df_plot = df_plot.resample(tf_rule).agg(agg_dict).dropna()

    with st.expander("Volatility & Performance", expanded=True):
        previous_close = df_plot["Close"].shift(1)
        true_range = pd.concat([df_plot["High"] - df_plot["Low"],
                                (df_plot["High"] - previous_close).abs(),
                                (df_plot["Low"] - previous_close).abs()], axis=1).max(axis=1)
        returns = df_plot["Close"].pct_change().dropna()
        periods = {"Daily": 252, "Weekly": 52, "Monthly": 12}[timeframe]
        a, b, c = st.columns(3)
        atr = true_range.rolling(14).mean().iloc[-1]
        a.metric("ATR (14 periods)", f"{atr:.2f}" if pd.notna(atr) else "N/A")
        b.metric("Annualized volatility", f"{returns.std() * periods ** 0.5:.1%}" if len(returns) > 1 else "N/A")
        c.metric("Selected period return", f"{df_plot['Close'].iloc[-1] / df_plot['Close'].iloc[0] - 1:+.1%}")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.18, 0.18, 0.14])

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Price',
        increasing_fillcolor="#00D47E", decreasing_fillcolor="#FF7187",
        increasing_line_color="#00D47E", decreasing_line_color="#FF7187"
    ), row=1, col=1)

    if 'SMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'],
                                 line=dict(color='#00C9FF', width=1.5), name='SMA 50'),
                      row=1, col=1)

    macd_col = [col for col in df_plot.columns if col.startswith('MACD_')]
    macdh_col = [col for col in df_plot.columns if col.startswith('MACDh_')]

    if macd_col and macdh_col:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[macd_col[0]],
                                 line=dict(color='#00C9FF', width=1.5), name='MACD'),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot[macdh_col[0]],
                             marker_color='gray', name='Histogram'),
                      row=2, col=1)

    rsi_col = [col for col in df_plot.columns if col.startswith('RSI')]
    if rsi_col:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[rsi_col[0]],
                                 line=dict(color='#C5ABFF', width=1.5), name='RSI'),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF5350", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#26A69A", line_width=1, row=3, col=1)

    if "Volume" in df_plot:
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["Volume"], name="Volume", marker_color="#6d9de1"), row=4, col=1)

    fig.update_layout(
        template=CHART_TEMPLATE,
        **CHART_STYLE,
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified',
        showlegend=False
    )

    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, gridwidth=1, griddash='dot', showline=False, zeroline=False, row=i, col=1)
        fig.update_yaxes(side='right', showgrid=True, gridcolor=CHART_GRID, gridwidth=1, griddash='dot', showline=False, zeroline=False, row=i, col=1)

    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Hover over candles for OHLCV details. Use the resolution selector to resample candles.")

