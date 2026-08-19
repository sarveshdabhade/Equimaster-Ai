"""
Stock Overview Tab - 52 Week Range and Company Summary
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


def _resolve_yf_symbol(ticker: str) -> str:
    """Map app-internal ticker names to valid Yahoo Finance symbols."""
    if ticker == "NIFTY 50":
        return "^NSEI"
    if ticker == "NSEI":
        return "^NSEI"
    if ".NS" in ticker or ticker.startswith("^"):
        return ticker
    return f"{ticker}.NS"


def render_overview_tab(df, ticker):
    """
    Renders the stock overview page with 52-week range and company summary.
    
    Args:
        df: DataFrame with historical price data
        ticker: Stock ticker symbol
    """
    st.markdown('<h2 class="section-title"><span class="accent">Stock</span> Overview</h2>', unsafe_allow_html=True)
    
    # Fetch 52-week data and company info
    info = None
    try:
        # Get ticker info from yfinance
        yf_ticker = yf.Ticker(_resolve_yf_symbol(ticker))
        info = yf_ticker.info
        
        # 52-week high/low
        week_52_high = info.get('fiftyTwoWeekHigh', 0)
        week_52_low = info.get('fiftyTwoWeekLow', 0)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice', 0)
        
        # Company info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 0)
        
        # Calculate position in 52-week range
        if week_52_high > week_52_low and current_price > 0:
            range_position = ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100
        else:
            range_position = 50
            
    except Exception as e:
        st.warning(f"Could not fetch live data for {ticker}. Using available data.")
        # Fallback to dataframe stats
        if df is not None and not df.empty:
            week_52_high = df['Close'].max() if 'Close' in df.columns else 0
            week_52_low = df['Close'].min() if 'Close' in df.columns else 0
            current_price = df['Close'].iloc[-1] if 'Close' in df.columns else 0
            range_position = 50
            company_name = ticker
            sector = industry = "N/A"
            market_cap = 0
            pe_ratio = "N/A"
            dividend_yield = 0
        else:
            st.error("No data available for this stock.")
            return
    
    # Layout: 52-week range on left, company info on right
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<h3 style="font-size: 22px; margin-bottom: 15px;">52-Week Price Range</h3>', unsafe_allow_html=True)
        
        # Current price display with larger font
        st.markdown(f"""
            <div style="margin: 15px 0;">
                <span style="font-size: 14px; color: #888;">Current Price</span><br>
                <span style="font-size: 32px; font-weight: 700; color: #00C9FF;">₹{current_price:,.2f}</span><br>
                <span style="font-size: 16px; color: #92FE9D;">{range_position:.1f}% of 52W range</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Clean 52-week range bar like reference image - dark theme optimized
        fig = go.Figure()

        # Light blue background bar (full range)
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='rgba(0, 201, 255, 0.25)', width=10),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Current position - green dot
        fig.add_trace(go.Scatter(
            x=[range_position],
            y=[0.5],
            mode='markers',
            marker=dict(
                size=18,
                color='#00D47E',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"LTP: ₹{current_price:,.2f}"
        ))

        # Vertical line from dot to price label
        fig.add_trace(go.Scatter(
            x=[range_position, range_position],
            y=[0.5, 0.82],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Current price with LTP label above
        fig.add_trace(go.Scatter(
            x=[range_position],
            y=[0.88],
            mode='text',
            text=[f"₹{current_price:,.1f} (LTP)"],
            textposition='top center',
            textfont=dict(size=15, color='white', weight='bold'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Low price with (Low) label - same format as LTP
        fig.add_trace(go.Scatter(
            x=[8],
            y=[0.15],
            mode='text',
            text=[f"₹{week_52_low:,.0f} (Low)"],
            textposition='middle center',
            textfont=dict(size=15, color='white', weight='bold'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # High price with (High) label - same format as LTP
        fig.add_trace(go.Scatter(
            x=[92],
            y=[0.15],
            mode='text',
            text=[f"₹{week_52_high:,.0f} (High)"],
            textposition='middle center',
            textfont=dict(size=15, color='white', weight='bold'),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            height=140,
            xaxis=dict(
                range=[-10, 110],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                range=[-0.15, 1.15],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                fixedrange=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=30, t=45, b=20),
            showlegend=False,
            dragmode=False
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
        
        # Range statistics
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("52W Low", f"₹{week_52_low:,.2f}")
        with stat_cols[1]:
            st.metric("Current", f"₹{current_price:,.2f}")
        with stat_cols[2]:
            st.metric("52W High", f"₹{week_52_high:,.2f}")
    
    with col_right:
        st.markdown('<h3 style="font-size: 22px; margin-bottom: 15px;">Company Profile</h3>', unsafe_allow_html=True)
        
        # Company name and sector - larger fonts
        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">{company_name}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 15px; color: #888;">{sector} | {industry}</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Key metrics
        st.markdown('<h4 style="font-size: 18px; margin: 15px 0 10px 0;">Key Metrics</h4>', unsafe_allow_html=True)
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            if market_cap > 0:
                if market_cap >= 1e12:
                    mc_display = f"₹{market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_display = f"₹{market_cap/1e9:.2f}B"
                else:
                    mc_display = f"₹{market_cap/1e6:.2f}M"
                st.metric("Market Cap", mc_display)
            
            if pe_ratio != 'N/A':
                st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        
        with metrics_col2:
            if dividend_yield:
                st.metric("Div Yield", f"{dividend_yield*100:.2f}%")
            
            st.metric("Position", f"{range_position:.1f}%")
    
    st.divider()
    
    # Company summary
    st.markdown('<h3 style="font-size: 22px; margin: 20px 0 15px 0;">About</h3>', unsafe_allow_html=True)
    
    if info is None:
        st.info("Company summary not available for this stock.")
    else:
        try:
            summary = info.get('longBusinessSummary', '')
            if summary:
                # Truncate if too long
                if len(summary) > 800:
                    summary = summary[:800] + "..."
                st.markdown(f"""
                    <div style="
                        background: rgba(0, 201, 255, 0.05);
                        border-left: 3px solid #00C9FF;
                        padding: 20px 25px;
                        border-radius: 0 8px 8px 0;
                        line-height: 1.8;
                        color: #E0E0E0;
                        font-size: 16px;
                    ">
                        {summary}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Company summary not available.")
        except Exception:
            st.info("Company summary not available for this stock.")
