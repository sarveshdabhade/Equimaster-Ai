"""
Continuous scrolling stock market ticker strip for Streamlit.
"""

import streamlit.components.v1 as components


def render_indices_ticker(indices_data=None):
    """
    Renders a smooth, horizontally scrolling stock ticker strip.

    Args:
        indices_data: Dict with index names as keys, containing 'last', 'delta', 'pct'.
                      Format: {"NIFTY 50": {"last": 22545.32, "delta": 125.45, "pct": 0.56}, ...}
                      If None, uses mock data.
    """

    # ===================== MOCK DATA SECTION =====================
    # REPLACE THIS BLOCK WITH LIVE DATA from fetch_index_snapshot()
    # Required indices: 1.NIFTY 50, 2.SENSEX, 3.BANK NIFTY, 4.NIFTY 100, 5.NIFTY IT, 6.NIFTY PHARMA
    if indices_data is None:
        indices_data = {
            "NIFTY 50": {"last": 22545.32, "delta": 125.45, "pct": 0.56},
            "SENSEX": {"last": 74123.55, "delta": 342.12, "pct": 0.46},
            "BANK NIFTY": {"last": 48234.15, "delta": -234.80, "pct": -0.48},
            "NIFTY 100": {"last": 24567.80, "delta": 98.50, "pct": 0.40},
            "NIFTY IT": {"last": 35456.80, "delta": 89.30, "pct": 0.25},
            "NIFTY PHARMA": {"last": 18456.22, "delta": -45.60, "pct": -0.25},
        }
    # =============================================================

    # Build ticker items HTML from indices_data dict
    ticker_items = []
    for name, data in indices_data.items():
        # Determine color based on change direction
        color_class = "positive" if data["delta"] >= 0 else "negative"
        arrow = "▲" if data["delta"] >= 0 else "▼"

        ticker_items.append(f"""
            <div class="ticker-item">
                <span class="index-name">{name}</span>
                <span class="index-price">₹{data['last']:,.2f}</span>
                <span class="index-change {color_class}">
                    {arrow} {abs(data['delta']):,.2f} ({abs(data['pct']):.2f}%)
                </span>
            </div>
        """)

    # Duplicate items for seamless infinite scroll
    all_items = ticker_items + ticker_items

    # Generate HTML/CSS - Bigger size, Inter font, slower speed
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

            .ticker-wrapper {{
                width: 100%;
                height: 65px;
                background: transparent;
                overflow: hidden;
                position: relative;
                border-bottom: 1px solid rgba(128, 128, 128, 0.2);
            }}

            .ticker-track {{
                display: flex;
                width: max-content;
                animation: scroll 60s linear infinite;
                height: 100%;
                align-items: center;
            }}

            .ticker-wrapper:hover .ticker-track {{
                animation-play-state: paused;
            }}

            @keyframes scroll {{
                0% {{ transform: translateX(0); }}
                100% {{ transform: translateX(-50%); }}
            }}

            .ticker-item {{
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 0 45px;
                white-space: nowrap;
                border-right: 1px solid rgba(128, 128, 128, 0.2);
            }}

            .index-name {{
                font-family: 'Inter', sans-serif;
                font-size: 16px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }}

            .index-price {{
                font-family: 'Inter', sans-serif;
                font-size: 17px;
                font-weight: 600;
                color: #f0f0f0;
            }}

            .index-change {{
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
        <div class="ticker-wrapper">
            <div class="ticker-track">
                {''.join(all_items)}
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=65)


# Example usage for standalone testing
if __name__ == "__main__":
    import streamlit as st

    st.title("Indices Ticker Demo")
    st.write("Hover over the ticker to pause scrolling")
    render_indices_ticker()
