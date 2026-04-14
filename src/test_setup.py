import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go

# 1. Title
st.title("Equimaster-Ai System Check")

# 2. Fetch Data (Test yfinance)
st.subheader("Fetching Data for Reliance...")
ticker = "RELIANCE.NS"
data = yf.download(ticker, period="1mo", interval="1d")

# 3. Calculate Indicator (Test pandas_ta)
# Create a simple Moving Average to test the library
data['SMA_50'] = ta.sma(data['Close'], length=50)

# 4. Show Data Table
st.write("Recent Data:")
st.dataframe(data.tail())

# 5. Plot Interactive Chart (Test Plotly)
st.subheader("Interactive Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig)

st.success("System is READY! All AI & Finance libraries are working correctly.")