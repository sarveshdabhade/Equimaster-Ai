"""Stock Analyzer Page - Detailed stock analysis with all tabs"""
import streamlit as st
import os
from tabs import overview, technicals, lstm_model, news_sentinel, fundamentals
from utils import load_data, fetch_stock_snapshot


def _resolve_yf_symbol(ticker: str) -> str:
    """Map app-internal ticker names to valid Yahoo Finance symbols."""
    if ticker == "NIFTY 50":
        return "^NSEI"
    if ticker == "NSEI":
        return "^NSEI"
    if ".NS" in ticker or ticker.startswith("^"):
        return ticker
    return f"{ticker}.NS"

st.set_page_config(page_title="Stock Analyzer | Equimaster-Ai", page_icon=None, layout="wide")

# Get ticker from session state
ticker = st.session_state.get("selected_ticker", None)

if not ticker:
    # Show search interface when no stock selected
    st.markdown('<h2 style="color: #00C9FF; margin-bottom: 20px;">Stock Search Terminal</h2>', unsafe_allow_html=True)

    processed_dir = "data/processed"
    local_files = [f.replace("_processed.csv", "") for f in os.listdir(processed_dir) if f.endswith("_processed.csv")] if os.path.exists(processed_dir) else []
    search_options = ["Select a stock..."] + ["NIFTY 50"] + sorted(local_files)

    selected = st.selectbox("Search Stock", search_options, index=0, label_visibility="collapsed")

    if selected != "Select a stock...":
        st.session_state.selected_ticker = selected
        st.rerun()

    st.caption("Or go back to home page")
    if st.button("< Back to Home", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

# Determine paths based on ticker
if ticker == "NIFTY 50":
    DATA_PATH = f"data/processed/NSEI_processed.csv"
    candidate_keras = f"models/lstm_NSEI.keras"
    candidate_h5 = f"models/lstm_NSEI.h5"
    if os.path.exists(candidate_keras):
        MODEL_PATH = candidate_keras
    elif os.path.exists(candidate_h5):
        MODEL_PATH = candidate_h5
    else:
        MODEL_PATH = None
    SCALER_PATH = f"data/train_data/scaler_NSEI.pkl"
else:
    DATA_PATH = f"data/processed/{ticker}_processed.csv"
    candidate_keras = f"models/lstm_{ticker}.keras"
    candidate_h5 = f"models/lstm_{ticker}.h5"
    MODEL_PATH = candidate_keras if os.path.exists(candidate_keras) else candidate_h5
    SCALER_PATH = f"data/train_data/scaler_{ticker}.pkl"

# Header with Search
header_col1, header_col2 = st.columns([2, 1])
with header_col1:
    st.markdown(f'<h1 style="color: #00C9FF; margin:0;">{ticker}</h1>', unsafe_allow_html=True)

with header_col2:
    # Search bar on stock analyzer page
    processed_dir = "data/processed"
    local_files = [f.replace("_processed.csv", "") for f in os.listdir(processed_dir) if f.endswith("_processed.csv")] if os.path.exists(processed_dir) else []
    search_options = ["Search another stock..."] + ["NIFTY 50"] + local_files
    
    # Pre-select current ticker in dropdown
    current_index = search_options.index(ticker) if ticker in search_options else 0
    new_selection = st.selectbox("Search Stock", search_options, index=current_index, label_visibility="collapsed")
    
    if new_selection != "Search another stock..." and new_selection != ticker:
        st.session_state.selected_ticker = new_selection
        st.rerun()

# Show live price snapshot
try:
    snapshot = fetch_stock_snapshot(_resolve_yf_symbol(ticker))
    if snapshot:
        col_left, col_right = st.columns([4, 1])
        price_val = f"₹{snapshot['last']:,.2f}"
        price_delta = f"{snapshot['delta']:,.2f} ({snapshot['pct']:.2f}%)"
        with col_left:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:14px'>"
                f"<div style='font-size:28px;font-weight:800;color:#61E5FF'>{price_val}<div style=\"font-size:12px;color:#FFB86B;font-weight:700;\">{price_delta}</div></div>"
                f"<div>"
                f"<div style='font-size:22px;font-weight:800'>{snapshot['name']}</div>"
                f"<div style='font-size:13px;color:#C7D2E0'>{snapshot['symbol']}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_right:
            st.info("AI Brain Connected")
except Exception:
    pass

st.divider()

# Load data — use a stable fingerprint so cache isn't defeated on every rerun
if DATA_PATH and os.path.exists(DATA_PATH):
    _stat = os.stat(DATA_PATH)
    _fingerprint = f"{_stat.st_size}_{int(_stat.st_mtime) // 60}"
else:
    _fingerprint = None
df = load_data(ticker, DATA_PATH, _fingerprint)

if df is None or df.empty:
    st.error(f"Unable to load data for {ticker}")
    st.stop()

# Stock Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Technical Chart",
    "LSTM Model",
    "News Sentiment",
    "Fundamentals"
])

with tab1:
    overview.render_overview_tab(df, ticker)

with tab2:
    technicals.render_technicals(df, ticker)

with tab3:
    lstm_model.render_lstm_tab(df, ticker, MODEL_PATH, SCALER_PATH)

with tab4:
    news_sentinel.render_news_tab(ticker)

with tab5:
    fundamentals.render_fundamentals_tab(ticker)

# Back button
st.divider()
if st.button("< Back to Home", use_container_width=True):
    st.switch_page("app.py")
