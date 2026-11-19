"""Company research workspace with a compact quote header and lazy section navigation."""
from html import escape
from pathlib import Path
import streamlit as st
import pandas as pd
from components.layout import render_shell, render_footer
from utils import load_data, fetch_stock_snapshot
from config import resolve_app_path, provider_symbol, ALL_DOWNLOAD_TICKERS
from src.artifacts import canonical_ticker, resolve_artifact
from src.dashboard_data import data_fingerprint, load_dashboard_rows
from src.fundamentals import read_cached_details

st.set_page_config(page_title="Stock analysis | Equimaster AI", page_icon="E", layout="wide", initial_sidebar_state="auto")
render_shell("Stock analysis")
options = ["NIFTY 50"] + list(ALL_DOWNLOAD_TICKERS)
requested = st.query_params.get("ticker")
ticker = None
if requested:
    requested = "NIFTY 50" if requested.upper() in ("NSEI", "NIFTY 50", "^NSEI") else requested.upper()
    if requested in options:
        st.session_state.selected_ticker = requested
        ticker = requested
    else:
        st.warning("That symbol is outside the current research coverage. Select a company below.")
        st.session_state.pop("selected_ticker", None)
else:
    # Sidebar navigation opens the analyzer landing page. Detail views are
    # entered only through a ticker link or an explicit in-page selection.
    st.session_state.pop("selected_ticker", None)
if ticker and ticker not in options:
    st.session_state.pop("selected_ticker", None)
    ticker = None

st.markdown(
    '<section class="home-hero">'
    '<h1 class="home-brand">Equimaster-Ai</h1>'
    '<p>Multi-Modal Deep Learning Platform for Financial Time-Series</p>'
    '</section><div class="home-divider"></div>',
    unsafe_allow_html=True,
)

if not ticker:
    coverage = load_dashboard_rows(data_fingerprint())
    records = coverage.set_index("Ticker", drop=False).to_dict("index") if not coverage.empty else {}
    stocks = [
        records.get(option, {
            "Ticker": option,
            "Symbol": option.removesuffix(".NS"),
            "Company": option.removesuffix(".NS"),
            "Sector": "Unclassified",
            "Price": None,
            "Change %": None,
            "As of": None,
        })
        for option in options if option != "NIFTY 50"
    ]
    sectors = sorted({stock["Sector"] for stock in stocks if stock.get("Sector") and stock["Sector"] != "Unclassified"})

    st.markdown(
        '<section class="research-start-hero">'
        '<div class="eyebrow">COMPANY RESEARCH · NSE COVERAGE</div>'
        '<h1>Turn a ticker into a research brief.</h1>'
        '<p>Search a company or symbol to explore its price history, fundamentals, news and model forecasts.</p>'
        '</section>',
        unsafe_allow_html=True,
    )
    count, sector_count, date = st.columns(3)
    count.metric("COMPANIES IN COVERAGE", len(stocks))
    sector_count.metric("SECTORS", len(sectors))
    latest_date = coverage["As of"].max() if not coverage.empty else "No saved prices"
    date.metric("LATEST SAVED CLOSE", latest_date)

    search_col, sector_col = st.columns([2.2, 1])
    with search_col:
        search_term = st.text_input(
            "Search companies",
            placeholder="Try “Tata”, “TCS” or another NSE ticker",
            key="research_company_search",
        ).strip().casefold()
    with sector_col:
        selected_sector = st.selectbox("Filter by sector", ["All sectors"] + sectors, key="research_sector_filter")

    matches = [
        stock for stock in stocks
        if (selected_sector == "All sectors" or stock.get("Sector") == selected_sector)
        and (not search_term or search_term in f"{stock.get('Company', '')} {stock.get('Ticker', '')} {stock.get('Sector', '')}".casefold())
    ]
    if search_term:
        matches.sort(key=lambda stock: str(stock.get("Company", "")).casefold())
        section_title = f"Matching companies · {len(matches)}"
    else:
        matches.sort(key=lambda stock: abs(float(stock.get("Change %") or 0)), reverse=True)
        section_title = "Worth a closer look"
    st.markdown(
        f'<div class="research-results-heading"><h2>{escape(section_title)}</h2>'
        f'<span>{"Search by company or ticker" if search_term else "Largest saved-session moves · a quick place to start"}</span></div>',
        unsafe_allow_html=True,
    )
    if not matches:
        st.info("No covered companies match those filters. Try a broader search or choose another sector.")
    else:
        if not search_term:
            matches = matches[:9]
        for start in range(0, len(matches), 3):
            columns = st.columns(3)
            for column, stock in zip(columns, matches[start:start + 3]):
                with column, st.container(border=True, key=f"research_pick_{stock['Ticker'].replace('.', '_')}"):
                    ticker_label = escape(str(stock.get("Symbol") or stock["Ticker"]))
                    company_label = escape(str(stock.get("Company") or ticker_label))
                    sector_label = escape(str(stock.get("Sector") or "Sector unavailable"))
                    st.markdown(
                        f'<div class="research-stock-symbol">{ticker_label}</div>'
                        f'<div class="research-stock-name">{company_label}</div>'
                        f'<div class="research-stock-sector">{sector_label}</div>',
                        unsafe_allow_html=True,
                    )
                    price = stock.get("Price")
                    change = stock.get("Change %")
                    if price is not None:
                        tone = "positive" if float(change or 0) >= 0 else "negative"
                        st.markdown(
                            f'<div class="research-stock-price">₹{float(price):,.2f}'
                            f'<span class="{tone}"> {float(change or 0):+.2f}%</span></div>'
                            f'<div class="research-stock-date">Saved close · {escape(str(stock.get("As of") or "Date unavailable"))}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Price history unavailable")
                    if st.button("Open research →", key=f"open_research_{stock['Ticker']}", use_container_width=True):
                        st.session_state.selected_ticker = stock["Ticker"]
                        st.query_params["ticker"] = stock["Ticker"]
                        st.rerun()
    st.page_link("app.py", label="← Back to markets")
    st.stop()
symbol = canonical_ticker(ticker)
path = Path(resolve_app_path("data", "processed", f"{symbol}_processed.csv"))
fingerprint = f"{path.stat().st_size}_{path.stat().st_mtime_ns}" if path.exists() else None
df = load_data(ticker, str(path), fingerprint)
if df is None or df.empty:
    st.error(f"Historical data is unavailable for {ticker}.")
    st.stop()
details = read_cached_details(symbol) or {}
try:
    snapshot = fetch_stock_snapshot(provider_symbol(symbol))
except Exception:
    snapshot = None
close = float(df["Close"].iloc[-1])
previous = float(df["Close"].iloc[-2]) if len(df) > 1 else close
price = float(snapshot["last"]) if snapshot else close
change = float(snapshot["delta"]) if snapshot else close - previous
pct = float(snapshot["pct"]) if snapshot else (close / previous - 1) * 100 if previous else 0
as_of = snapshot.get("as_of") if snapshot else str(df.index[-1])[:10]
company = details.get("Company") or (snapshot.get("name") if snapshot else None) or ticker
tone = "positive" if change >= 0 else "negative"
try:
    artifact = resolve_artifact(symbol, 1)
    model_path, scaler_path = str(artifact["model"]), str(artifact["scaler"])
    model_label = "Model available"
except (FileNotFoundError, KeyError, ValueError):
    artifact = None
    model_path, scaler_path, model_label = None, None, "Model unavailable"
with st.container(border=True, key="company_quote"):
    heading, search = st.columns([3.6, 2])
    with heading:
        st.markdown(f'<div class="stock-symbol">NSE · {escape(symbol.removesuffix(".NS"))} <span class="pill">{escape(model_label)}</span></div>'
                    f'<div class="company-name">{escape(company)}</div>'
                    f'<span class="quote-price">{price:,.2f}</span><span class="quote-change {tone}">{change:+,.2f} ({pct:+.2f}%)</span>'
                    f'<div class="stock-meta">{"Latest provider quote" if snapshot else "Saved closing price"} · {escape(str(as_of or "Date unavailable"))} · {escape(details.get("Sector") or "Indian equities")}</div>', unsafe_allow_html=True)
    with search:
        selected = st.selectbox("Search Stock", options, index=options.index(ticker) if ticker in options else 0, label_visibility="collapsed", key="company_search")
        if selected != ticker:
            st.session_state.selected_ticker = selected
            st.query_params["ticker"] = selected
            st.rerun()
        st.caption("INR" if symbol != "NSEI" else "Index points")
with st.container(key="research_nav"):
    section = st.radio("Research sections", ["Overview", "Technical Chart", "LSTM Model", "News Sentiment", "Fundamentals"],
                       horizontal=True, label_visibility="collapsed", key="research_section")
if section == "News Sentiment":
    from tabs import news_sentinel
    news_sentinel.render_news_tab(ticker)
else:
    with st.container(border=True, key="research_content"):
        if section == "Overview":
            from tabs import overview
            overview.render_overview_tab(df, ticker)
        elif section == "Technical Chart":
            from tabs import technicals
            technicals.render_technicals(df, ticker)
        elif section == "LSTM Model":
            from tabs import lstm_model
            lstm_model.render_lstm_tab(df, ticker, model_path, scaler_path)
        elif section == "Fundamentals":
            from tabs import fundamentals
            fundamentals.render_fundamentals_tab(ticker, current_price=price, price_history=df)
training_end = artifact.get("metadata", {}).get("data_end") if artifact else None
training_label = pd.Timestamp(training_end).strftime("%d %b %Y") if training_end else "unavailable"
st.caption(f"Price history through {df.index[-1]:%d %b %Y}. Model training data through {training_label}. Quote and financial reporting dates may differ.")
render_footer()
