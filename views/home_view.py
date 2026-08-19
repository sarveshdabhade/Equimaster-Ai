"""Home View - Landing page with tickers, search, and market overview"""
import streamlit as st
import os
from components.indices_ticker import render_indices_ticker
from components.sensex_ticker import render_sensex_ticker
from components.market_movers import render_market_movers
from utils import fetch_index_snapshot

def render_home_view():
    """Render the homepage view with tickers, search, and market movers"""
    
    # Live Market Tickers - Indices + Sensex 30 Stocks
    indices = fetch_index_snapshot()
    if indices:
        st.markdown('<p style="font-size: 20px; color: #888; margin: 0; padding: 5px 0 0 10px;">Live Indices</p>', unsafe_allow_html=True)
        render_indices_ticker(indices)
    else:
        render_indices_ticker()
        st.info("Live indices temporarily unavailable. Click sidebar Refresh Data.")
    
    # Sensex 30 Companies Ticker
    st.markdown('<p style="font-size: 16px; color: #888; margin: 0; padding: 5px 0 0 10px;">Sensex 30</p>', unsafe_allow_html=True)
    render_sensex_ticker()
    st.divider()
    
    # Asset Search Terminal
    st.markdown('<h2 class="section-title"><span class="accent">Asset Search Terminal</span></h2>', unsafe_allow_html=True)
    
    processed_dir = "data/processed"
    local_files = [f.replace("_processed.csv", "") for f in os.listdir(processed_dir) if f.endswith("_processed.csv")] if os.path.exists(processed_dir) else []
    
    # Add placeholder as first option
    dropdown_options = ["Search a stock..."] + ["NIFTY 50"] + local_files
    
    col_search, col_status = st.columns([3, 1])
    with col_search:
        selected_option = st.selectbox(
            "Select Stock", 
            dropdown_options, 
            index=0, 
            label_visibility="collapsed", 
            help="Choose from trained models or enter custom (data auto-downloads)"
        )
        ticker = None if selected_option == "Search a stock..." else selected_option
        
        if ticker:
            st.info(f"Opening {ticker} analyzer...")
            # Store ticker in session state and navigate
            st.session_state.selected_ticker = ticker
            st.switch_page("pages/stock_analyzer.py")
    
    # Show hint when no stock selected
    if not ticker:
        st.info("Select a stock to open detailed analysis in a new page")
    
    # Market Movers Section
    render_market_movers()
