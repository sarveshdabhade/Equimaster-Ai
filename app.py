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

# Import refactored helpers
from views import render_home_view

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
    /* Sidebar navigation - bold and uppercase */
    section[data-testid="stSidebar"] a,
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a,
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] span {
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 14px !important;
    }
    section[data-testid="stSidebar"] a span,
    section[data-testid="stSidebar"] div[role="listitem"] span {
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 14px !important;
    }
    /* Target Streamlit navigation links specifically */
    .st-emotion-cache-1cypcdb a,
    .st-emotion-cache-1cypcdb span {
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    /* Universal catch-all for sidebar nav items */
    section[data-testid="stSidebar"] * {
        text-transform: uppercase !important;
    }
    section[data-testid="stSidebar"] div:has(> a) span,
    section[data-testid="stSidebar"] nav span {
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with quick actions (Phase 1)
with st.sidebar:
    st.markdown('<h3 style="color: #7CE6FF;">Quick Actions</h3>', unsafe_allow_html=True)
    if st.button("Refresh Data", use_container_width=True):
        from utils import fetch_index_snapshot, fetch_stock_snapshot
        fetch_index_snapshot.clear()
        fetch_stock_snapshot.clear()
        st.success("Cache cleared - data refreshing!")

    st.markdown("---")
    st.markdown('<h4 style="color: #92FE9D;">Platform Stats</h4>', unsafe_allow_html=True)
    st.caption("Visit Training Manager page for model details and retraining.")
    st.markdown("---")
    st.caption("Equimaster v1.0")

# --- 2. HEADER SECTION ---
st.markdown('<h1 class="title-text">Equimaster-Ai</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Multi-Modal Deep Learning Platform for Financial Time-Series</p>', unsafe_allow_html=True)

st.divider()

# --- 3. HOME VIEW (Tickers, Search, Market Movers) ---
# This will handle search and navigation to stock_analyzer page
render_home_view()


