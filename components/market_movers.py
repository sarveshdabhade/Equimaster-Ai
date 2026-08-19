"""Market Movers Component - Reusable widget for top gainers/losers"""
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import MARKET_MOVERS_TICKERS as MAJOR_STOCKS

def _fetch_single_ticker(symbol_name):
    """Fetch data for a single ticker."""
    symbol, name = symbol_name
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d", interval="1d")
        
        if not hist.empty and len(hist) >= 1:
            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0
            
            return {
                'Symbol': symbol.replace('.NS', ''),
                '_yf_symbol': symbol,
                'Company': name,
                'Price': current,
                'Change': change,
                'Change %': change_pct,
                'Volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            }
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_market_movers():
    """Fetch real-time data for major stocks using parallel processing."""
    data = []
    
    # Parallel fetch using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_single_ticker, item): item for item in MAJOR_STOCKS.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)
    
    if not data:
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(data)
    df = df.sort_values('Change %', ascending=False)
    
    # Top 5 gainers and losers
    gainers = df.head(5).copy()
    losers = df.tail(5).copy()
    
    return gainers, losers

def format_change_pct(value):
    """Format change percentage with color indicator"""
    if value >= 0:
        return f"▲ +{value:.2f}%"
    else:
        return f"▼ {value:.2f}%"

def render_market_movers():
    """Render the market movers widget"""
    
    # Custom CSS for colored buttons
    st.markdown("""
        <style>
        /* All market mover buttons same size */
        div[data-testid="stHorizontalBlock"] button {
            width: 100% !important;
            min-height: 40px !important;
        }
        /* Gainers buttons - green background with white text */
        div[data-testid="stHorizontalBlock"] button[data-testid*="gainer"],
        button[data-testid*="gainer_"] {
            background: linear-gradient(90deg, #00D47E, #00B894) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 600 !important;
        }
        div[data-testid="stHorizontalBlock"] button[data-testid*="gainer"]:hover,
        button[data-testid*="gainer_"]:hover {
            background: linear-gradient(90deg, #00C9A7, #00D47E) !important;
            color: white !important;
        }
        /* Losers buttons - red background with white text */
        div[data-testid="stHorizontalBlock"] button[data-testid*="loser"],
        button[data-testid*="loser_"] {
            background: linear-gradient(90deg, #FF4757, #EE5A6F) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 600 !important;
        }
        div[data-testid="stHorizontalBlock"] button[data-testid*="loser"]:hover,
        button[data-testid*="loser_"]:hover {
            background: linear-gradient(90deg, #FF2E43, #FF4757) !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 style="color: white; margin-bottom: 5px;">Market Movers</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #888; margin-bottom: 20px;">Real-time top gainers and losers</p>', unsafe_allow_html=True)
    
    # Fetch market data
    with st.spinner("Fetching live market data..."):
        gainers, losers = fetch_market_movers()
    
    if gainers.empty and losers.empty:
        st.warning("No market data available. Check internet connection or try Refresh.")
        st.caption("Tips: 1) Click Refresh button 2) Check if market is open 3) Wait a moment and retry")
        return
    elif gainers.empty:
        st.info("Gainers data temporarily unavailable. Showing losers only.")
    elif losers.empty:
        st.info("Losers data temporarily unavailable. Showing gainers only.")
    
    col_status, col_refresh = st.columns([3, 1])
    with col_status:
        # Check actual NSE market hours (Mon-Fri, 9:15-15:30 IST)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        is_weekday = now_ist.weekday() < 5
        market_open = is_weekday and (
            (now_ist.hour == 9 and now_ist.minute >= 15) or
            (now_ist.hour > 9 and now_ist.hour < 15) or
            (now_ist.hour == 15 and now_ist.minute <= 30)
        )
        if market_open:
            st.markdown('<span style="color: #00D47E; font-size: 14px;">[OPEN] Market Open</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #FF4757; font-size: 14px;">[CLOSED] Market Closed</span>', unsafe_allow_html=True)
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            fetch_market_movers.clear()
            st.rerun()
    
    # Show data freshness
    ist = pytz.timezone('Asia/Kolkata')
    st.caption(f"Last updated: {datetime.now(ist).strftime('%H:%M:%S IST')}")
    
    st.divider()
    
    col_gainers, col_losers = st.columns(2)
    
    # Top Gainers
    with col_gainers:
        st.markdown(f'<h4 style="color: white; margin-bottom: 15px;">Top gainers <span style="background: #00D47E; color: black; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{len(gainers)}</span></h4>', unsafe_allow_html=True)
        
        for _, row in gainers.iterrows():
            symbol = row['Symbol']
            price = f"{row['Price']:,.2f}"
            change_val = row['Change']
            change_pct = row['Change %']
            change_str = f"+{change_val:,.2f} (+{change_pct:.2f}%)" if change_val >= 0 else f"{change_val:,.2f} ({change_pct:.2f}%)"
            
            # Clickable row layout
            cols = st.columns([3, 1, 2])
            with cols[0]:
                # Green styled button for gainers
                # Store the full Yahoo Finance symbol (with .NS) in session state
                full_symbol = row.get('_yf_symbol', symbol + '.NS')
                if st.button(symbol, key=f"gainer_{symbol}", use_container_width=True):
                    st.session_state.selected_ticker = full_symbol
                    st.switch_page("pages/stock_analyzer.py")
            with cols[1]:
                st.markdown(f"<div style='text-align: right; color: white; font-weight: 500;'>{price}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='text-align: right; color: #00D47E; font-size: 13px;'>{change_str}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 8px 0; border-color: #333;'>", unsafe_allow_html=True)
    
    # Top Losers
    with col_losers:
        st.markdown(f'<h4 style="color: white; margin-bottom: 15px;">Top losers <span style="background: #FF4757; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{len(losers)}</span></h4>', unsafe_allow_html=True)
        
        for _, row in losers.iterrows():
            symbol = row['Symbol']
            price = f"{row['Price']:,.2f}"
            change_val = row['Change']
            change_pct = row['Change %']
            change_str = f"{change_val:,.2f} ({change_pct:.2f}%)"
            
            # Clickable row layout
            cols = st.columns([3, 1, 2])
            with cols[0]:
                # Red styled button for losers
                full_symbol = row.get('_yf_symbol', symbol + '.NS')
                if st.button(symbol, key=f"loser_{symbol}", use_container_width=True):
                    st.session_state.selected_ticker = full_symbol
                    st.switch_page("pages/stock_analyzer.py")
            with cols[1]:
                st.markdown(f"<div style='text-align: right; color: white; font-weight: 500;'>{price}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='text-align: right; color: #FF4757; font-size: 13px;'>{change_str}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 8px 0; border-color: #333;'>", unsafe_allow_html=True)
