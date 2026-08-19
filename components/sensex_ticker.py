"""
Continuous scrolling BSE Sensex 30 stock ticker for Streamlit.
Fetches live data from Yahoo Finance via yfinance.
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import SENSEX_30_TICKERS


def _fetch_single_sensex_stock(ticker_item):
    """Fetch data for a single Sensex stock."""
    ticker_sym, company_name = ticker_item
    try:
        import yfinance as yf
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info

        # Get current price and previous close
        current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('lastPrice')
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')

        if current_price and prev_close:
            change_pct = ((current_price - prev_close) / prev_close) * 100
            return {
                "symbol": ticker_sym.replace('.NS', ''),
                "name": company_name,
                "price": current_price,
                "change_pct": change_pct
            }
    except Exception:
        pass
    return None

# Cache with minimal TTL for near real-time data
@st.cache_data(ttl=30)
def fetch_sensex_data():
    """
    Fetch live data for all 30 Sensex stocks from Yahoo Finance in parallel.
    Returns list of dicts with symbol, name, price, change_pct.
    Falls back to mock data if API fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    # All 30 Sensex constituents with Yahoo Finance ticker symbols (.NS for NSE)
    sensex_tickers = SENSEX_30_TICKERS

    live_data = []
    
    # Parallel fetch using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_single_sensex_stock, item): item for item in sensex_tickers.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                live_data.append(result)

    return live_data if live_data else None


def render_sensex_ticker():
    """
    Renders a smooth, horizontally scrolling ticker for all 30 BSE Sensex companies.
    Uses CSS animation for continuous marquee effect. Pauses on hover.
    Fetches live data from yfinance with 30s cache.
    """

    # Fetch live data (30s cache for minimal delay)
    live_data = fetch_sensex_data()

    # ===================== FALLBACK MOCK DATA =====================
    if live_data is None:
        sensex_stocks = [
            {"symbol": "RELIANCE", "name": "Reliance Industries", "price": 2456.80, "change_pct": 1.25},
            {"symbol": "TCS", "name": "Tata Consultancy", "price": 3892.45, "change_pct": -0.85},
            {"symbol": "HDFCBANK", "name": "HDFC Bank", "price": 1634.20, "change_pct": 0.65},
            {"symbol": "INFY", "name": "Infosys", "price": 1823.55, "change_pct": -1.20},
            {"symbol": "ICICIBANK", "name": "ICICI Bank", "price": 1245.30, "change_pct": 1.85},
            {"symbol": "HUL", "name": "Hindustan Unilever", "price": 2345.60, "change_pct": -0.45},
            {"symbol": "SBIN", "name": "State Bank of India", "price": 765.80, "change_pct": 2.15},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "price": 1456.90, "change_pct": 0.95},
            {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "price": 7234.50, "change_pct": -0.75},
            {"symbol": "ITC", "name": "ITC Ltd", "price": 423.45, "change_pct": 0.55},
            {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "price": 1876.30, "change_pct": -0.35},
            {"symbol": "LT", "name": "Larsen & Toubro", "price": 3654.20, "change_pct": 1.45},
            {"symbol": "AXISBANK", "name": "Axis Bank", "price": 1123.65, "change_pct": 0.25},
            {"symbol": "ASIANPAINT", "name": "Asian Paints", "price": 2987.40, "change_pct": -0.65},
            {"symbol": "MARUTI", "name": "Maruti Suzuki", "price": 11234.80, "change_pct": 0.85},
            {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical", "price": 1567.90, "change_pct": 1.35},
            {"symbol": "TATAMOTORS", "name": "Tata Motors", "price": 987.60, "change_pct": 2.55},
            {"symbol": "M&M", "name": "Mahindra & Mahindra", "price": 2876.45, "change_pct": -0.25},
            {"symbol": "ULTRACEMCO", "name": "UltraTech Cement", "price": 11245.30, "change_pct": 0.45},
            {"symbol": "TITAN", "name": "Titan Company", "price": 3456.70, "change_pct": -1.05},
            {"symbol": "TECHM", "name": "Tech Mahindra", "price": 1234.50, "change_pct": 0.15},
            {"symbol": "NESTLEIND", "name": "Nestle India", "price": 23456.80, "change_pct": -0.55},
            {"symbol": "POWERGRID", "name": "Power Grid Corp", "price": 312.45, "change_pct": 0.75},
            {"symbol": "NTPC", "name": "NTPC Ltd", "price": 356.80, "change_pct": 1.05},
            {"symbol": "INDUSINDBK", "name": "IndusInd Bank", "price": 1456.30, "change_pct": -0.95},
            {"symbol": "TATASTEEL", "name": "Tata Steel", "price": 134.80, "change_pct": 1.65},
            {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv", "price": 1654.20, "change_pct": -0.15},
            {"symbol": "ADANIPORTS", "name": "Adani Ports", "price": 1456.70, "change_pct": 2.25},
            {"symbol": "JSWSTEEL", "name": "JSW Steel", "price": 987.40, "change_pct": 0.35},
            {"symbol": "DRREDDY", "name": "Dr. Reddy's Labs", "price": 6543.20, "change_pct": -0.45},
        ]
    else:
        sensex_stocks = live_data
    # =============================================================

    # Build ticker items HTML using Python for-loop and f-strings
    ticker_items = []
    for stock in sensex_stocks:
        color_class = "positive" if stock["change_pct"] >= 0 else "negative"
        arrow = "▲" if stock["change_pct"] >= 0 else "▼"

        ticker_items.append(f"""
            <div class="ticker-item">
                <span class="stock-symbol">{stock['symbol']}</span>
                <span class="stock-price">₹{stock['price']:,.2f}</span>
                <span class="stock-change {color_class}">
                    {arrow} {abs(stock['change_pct']):.2f}%
                </span>
            </div>
        """)

    # Duplicate items for seamless infinite scroll
    all_items = ticker_items + ticker_items

    # Generate complete HTML/CSS - Bigger size, consistent fonts
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            .ticker-container {{
                width: 100%;
                height: 65px;
                background: transparent;
                overflow: hidden;
                position: relative;
                border-top: 1px solid rgba(128, 128, 128, 0.2);
                border-bottom: 1px solid rgba(128, 128, 128, 0.2);
            }}

            .ticker-track {{
                display: flex;
                width: max-content;
                height: 100%;
                align-items: center;
                animation: marquee 45s linear infinite;
            }}

            .ticker-container:hover .ticker-track {{
                animation-play-state: paused;
            }}

            @keyframes marquee {{
                0% {{ transform: translateX(0); }}
                100% {{ transform: translateX(-50%); }}
            }}

            .ticker-item {{
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 0 45px;
                white-space: nowrap;
                border-right: 1px solid rgba(128, 128, 128, 0.15);
            }}

            .stock-symbol {{
                font-family: 'Inter', sans-serif;
                font-size: 16px;
                font-weight: 700;
                color: #00C9FF;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }}

            .stock-price {{
                font-family: 'Inter', sans-serif;
                font-size: 17px;
                font-weight: 600;
                color: #FFFFFF;
            }}

            .stock-change {{
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 6px;
            }}

            .positive {{
                color: #00D47E;
                background: rgba(0, 212, 126, 0.15);
            }}

            .negative {{
                color: #FF4757;
                background: rgba(255, 71, 87, 0.15);
            }}
        </style>
    </head>
    <body>
        <div class="ticker-container">
            <div class="ticker-track">
                {''.join(all_items)}
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=65)
