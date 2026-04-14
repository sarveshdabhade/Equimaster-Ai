import os
import streamlit as st
import logging
import traceback
import subprocess
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from streamlit_lottie import st_lottie

# Import refactored helpers and tab renderers
from utils import load_lottieurl, fetch_index_snapshot, fetch_stock_snapshot, load_data
from tabs import technicals, lstm_model, news_sentinel, fundamentals

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(page_title="Equimaster-Ai", page_icon=None, layout="wide")

# Configure app logging for runtime errors
logging.basicConfig(filename='app_errors.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(0, 201, 255, 0.12), transparent 35%),
            radial-gradient(circle at 80% 100%, rgba(146, 254, 157, 0.1), transparent 35%),
            #060A14;
    }

    h1, h2, h3, [data-testid="stMetricLabel"] p {
        font-family: 'Sora', sans-serif;
    }

    p, span, div {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Gradient text for main title */
    .title-text {
        background: linear-gradient(90deg, #00C9FF, #61E5FF, #92FE9D, #00C9FF);
        background-size: 220% 220%;
        animation: gradientShift 6s ease infinite, titleReveal 0.9s ease-out;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
        line-height: 1.05;
    }

    .subtitle-text {
        color: #C7D2E0;
        font-size: clamp(1rem, 1.5vw, 1.3rem);
        margin-top: 0;
        margin-bottom: 0;
        opacity: 0;
        animation: fadeSlideUp 1s ease-out 0.3s forwards;
    }

    .section-title {
        font-size: clamp(1.5rem, 2.2vw, 2.2rem);
        line-height: 1.1;
        margin: 0.4rem 0 0.8rem 0;
        color: #F1F5FF;
        letter-spacing: 0.2px;
        animation: fadeSlideUp 0.7s ease-out;
    }

    .section-title .accent {
        color: #7CE6FF;
        text-shadow: 0 0 24px rgba(0, 201, 255, 0.35);
    }

    .market-band {
        color: #8EA6C4;
        font-size: 0.9rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        opacity: 0;
        animation: fadeSlideUp 0.8s ease-out 0.15s forwards;
    }

    /* Style the metric cards */
    div[data-testid="metric-container"] {
        background-color: #131722;
        border: 1px solid #2B2B43;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeSlideUp 0.75s ease-out;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: #3B9AC8;
        box-shadow: 0 14px 26px rgba(0, 0, 0, 0.45);
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.85rem, 2.6vw, 2.6rem);
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes titleReveal {
        from { transform: translateY(14px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes fadeSlideUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Buttons and interactive elements */
    .stButton>button {
        background: linear-gradient(90deg,#0ea5e9,#7dd3fc);
        color: #021024;
        font-weight: 700;
        border: none;
        padding: 10px 14px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(2,16,36,0.45);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }

    .stButton>button:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 14px 30px rgba(2,16,36,0.55);
        filter: brightness(1.03);
        cursor: pointer;
    }

    /* Metric animation */
    [data-testid="stMetricValue"] {
        animation: pulseValue 1.2s ease both;
    }

    @keyframes pulseValue {
        0% { transform: translateY(6px); opacity: 0; }
        60% { transform: translateY(-2px); opacity: 1; }
        100% { transform: translateY(0); opacity: 1; }
    }

    /* Card hover tooltip */
    .info-tooltip { position: relative; display: inline-block; }
    .info-tooltip .info-text {
        visibility: hidden; width: 210px; background: rgba(10,14,20,0.92); color: #dbeafe;
        text-align: left; border-radius: 8px; padding: 8px; position: absolute; z-index: 9999;
        bottom: 125%; left: 50%; transform: translateX(-50%); box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        opacity: 0; transition: opacity .18s ease, transform .18s ease;
    }
    .info-tooltip:hover .info-text { visibility: visible; opacity: 1; transform: translateX(-50%) translateY(-6px); }

    /* Soft glass card for sections */
    .glass-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: 12px; padding: 10px; border: 1px solid rgba(255,255,255,0.025); }

    /* Sidebar styles */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6,10,20,0.95), rgba(10,14,20,0.98));
        border-right: 1px solid #2A2F3E;
    }
    section[data-testid="stSidebar"] > div > div > button {
        width: 100%;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with quick actions (Phase 1)
with st.sidebar:
    st.markdown('<h3 style="color: #7CE6FF;">⚡ Quick Actions</h3>', unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared - data refreshing!")
    if st.button("🚀 Retrain Selected Ticker", use_container_width=True):
        st.info("Opening Training Manager...")
        st.switch_page("pages/training_manager.py")
    st.markdown("---")
    st.markdown('<h4 style="color: #92FE9D;">📊 Platform Stats</h4>', unsafe_allow_html=True)
    models_count = len([f for f in os.listdir("models") if f.endswith('.keras')])
    st.metric("Trained Models", models_count)
    st.metric("Stocks Covered", len(os.listdir("logs")) - 1 if os.path.exists("logs") else 0)
    st.markdown("---")
    st.caption("Equimaster v1.0")

# --- 2. LOTTIE ANIMATIONS ---
# Use refactored helper in `utils.py`
lottie_trading = load_lottieurl("https://lottie.host/80dc18df-05ba-4114-b816-5c5b0ab128ac/oGkCihKzY7.json")

# --- 4. HEADER SECTION ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="title-text">Equimaster-Ai</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Multi-Modal Deep Learning Platform for Financial Time-Series</p>', unsafe_allow_html=True)
with col2:
    if lottie_trading:
        st_lottie(lottie_trading, height=100, key="trading_anim")

st.divider()

# Hero KPIs Grid (Phase 1)
st.markdown('<p class="market-band">Platform Overview</p>', unsafe_allow_html=True)
kpi_cols = st.columns(4)
with kpi_cols[0]:
    models = len([f for f in os.listdir("models/") if f.endswith(('.keras', '.h5'))])
    st.metric("🧠 AI Models", models)
with kpi_cols[1]:
    stocks = len([f for f in os.listdir("logs/") if f.endswith('.json')]) if os.path.exists("logs/") else 0
    st.metric("📈 Stocks Trained", stocks)
with kpi_cols[2]:
    try:
        data_age = (pd.Timestamp.now() - pd.to_datetime(os.path.getmtime("data/processed") if os.path.exists("data/processed") else 0, unit='s')).days
        st.metric("📊 Data Freshness", f"{data_age} days")
    except:
        st.metric("📊 Data Freshness", "Checking...")
with kpi_cols[3]:
    st.metric("⚡ Status", "Live 🟢", delta="Online")

st.divider()

# Live Index Bar (enhanced)
indices = fetch_index_snapshot()
if indices:
    st.markdown('<p class="market-band">Live Market Indices</p>', unsafe_allow_html=True)
    cols = st.columns(len(indices))
    for col, (label, data) in zip(cols, indices.items()):
        delta_color = "normal" if data['pct'] > 0 else "inverse"
        with col:
            st.metric(label, f"₹{data['last']:,.2f}", f"{data['delta']:,.2f} ({data['pct']:.2f}%)", delta_color=delta_color)
else:
    st.info("🌐 Live indices temporarily unavailable. Click sidebar 🔄 Refresh Data.")
st.divider()

# --- 5. MAIN ASSET SEARCH BAR ---
processed_dir = "data/processed"
local_files = [f.replace("_processed.csv", "") for f in os.listdir(processed_dir) if f.endswith("_processed.csv")] if os.path.exists(processed_dir) else []

# Force NIFTY 50 to be the default first option
dropdown_options = ["NIFTY 50"] + local_files

# Enhanced Ticker Search with Preview (Phase 1)
st.markdown('<h2 class="section-title"><span class="accent">🔍 Asset Search Terminal</span></h2>', unsafe_allow_html=True)

# Search preview expander
with st.expander("Preview Available Stocks (Click to explore)", expanded=False):
    st.dataframe(pd.DataFrame({"Stocks": dropdown_options}), use_container_width=True, hide_index=True)

col_search, col_status = st.columns([3, 1])
with col_search:
    ticker = st.selectbox("Select Stock", dropdown_options, index=0, label_visibility="collapsed", help="Choose from trained models or enter custom (data auto-downloads)")
    if ticker:
        st.info(f"📂 Loaded: {ticker}")

# Dynamic Paths based on the Search Bar
if ticker == "NIFTY 50":
    # Map 'NIFTY 50' to the NSEI processed CSV and model naming used in the repo
    DATA_PATH = f"data/processed/NSEI_processed.csv"
    # prefer native Keras format, fall back to legacy h5
    candidate_keras = f"models/lstm_NSEI.keras"
    candidate_h5 = f"models/lstm_NSEI.h5"
    if os.path.exists(candidate_keras):
        MODEL_PATH = candidate_keras
    elif os.path.exists(candidate_h5):
        MODEL_PATH = candidate_h5
    else:
        MODEL_PATH = None
    SCALER_PATH = f"data/train_data/scaler_NSEI.pkl"
    with col_status:
        st.info("Market Index View Active")
else:
    DATA_PATH = f"data/processed/{ticker}_processed.csv"
    # prefer native Keras format, fall back to legacy h5
    candidate_keras = f"models/lstm_{ticker}.keras"
    candidate_h5 = f"models/lstm_{ticker}.h5"
    MODEL_PATH = candidate_keras if os.path.exists(candidate_keras) else candidate_h5
    SCALER_PATH = f"data/train_data/scaler_{ticker}.pkl"
    with col_status:
        st.success(f"AI Brain Connected: {ticker}")

    # Show the searched asset details (name + live market price) just below the search bar
    try:
        snapshot = fetch_stock_snapshot(ticker)
        if snapshot:
            # Place price left of the name, and increase the stock name size
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
                # keep space for layout consistency (empty for now)
                st.write("")
    except Exception:
        # Non-fatal: do not block the rest of the app if live fetch fails
        pass

CLOSE_COL_IDX = 1 
st.divider() # Adds a clean line before the charts start

# --- 6. DATA LOADING ---
# Implemented in `utils.load_data`

data_mtime = os.path.getmtime(DATA_PATH) if DATA_PATH and os.path.exists(DATA_PATH) else None
df = load_data(ticker, DATA_PATH, data_mtime)

# --- 7. THE 4 RESEARCH PILLARS (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Chart", "🧠 The LSTM Model", "📰 RAG News Sentiment", "⛏️ Fundamentals Miner"])

# --- TAB 1: TECHNICALS (MULTI-PANEL TRADINGVIEW UI) ---
with tab1:
    technicals.render_technicals(df, ticker)

# --- TAB 2: LSTM ---
# --- TAB 2: LSTM ---
with tab2:
    lstm_model.render_lstm_tab(df, ticker, MODEL_PATH, SCALER_PATH)

# --- TAB 3: SENTINEL (NEWS) ---
with tab3:
    news_sentinel.render_news_tab(ticker)

# --- TAB 4: MINER (FUNDAMENTALS) ---
with tab4:
    fundamentals.render_fundamentals_tab(ticker)

# Footer Status (Phase 1)
st.markdown("---")
st.markdown('<div class="glass-card" style="text-align: center; padding: 16px;">', unsafe_allow_html=True)
col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    data_exists = os.path.exists(DATA_PATH)
    st.metric("💾 Local Data", "✅ Ready" if data_exists else "❌ Missing", delta="Load now?")
with col_footer2:
    model_ready = MODEL_PATH and os.path.exists(MODEL_PATH)
    st.metric("🤖 Model Status", "🟢 Loaded" if model_ready else "🔴 Offline", delta="Retrain?")
st.markdown('  <p style="color: #8EA6C4; font-size: 0.85rem; margin-top: 8px;">Data updated recently | Models optimized for Nifty500</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
