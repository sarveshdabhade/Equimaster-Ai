"""Market dashboard, screener and recent news."""
from datetime import datetime, time
from html import escape
from urllib.parse import quote
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from components.indices_ticker import render_indices_ticker
from components.sensex_ticker import render_sensex_ticker  # Compatibility for existing component users.
from components.market_movers import render_market_movers
from components.layout import section_title, CHART_STYLE, CHART_TEMPLATE
from src.dashboard_data import data_fingerprint, load_dashboard_rows, index_history, apply_screen, cached_news
from utils import fetch_index_snapshot

INDIA_TZ = ZoneInfo("Asia/Kolkata")


def market_session_status(now=None):
    local_now = now or datetime.now(INDIA_TZ)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=INDIA_TZ)
    else:
        local_now = local_now.astimezone(INDIA_TZ)
    local_time = local_now.time().replace(tzinfo=None)
    is_open = local_now.weekday() < 5 and time(9, 15) <= local_time < time(15, 30)
    return ("OPEN", "market-open") if is_open else ("CLOSED", "market-closed")


def _render_market_clock(status, status_class):
    components.html(
        f"""
        <style>
          body {{ margin: 0; background: transparent; color: #BAC7DA;
                 font: 650 16px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
          .clock {{ height: 56px; display: flex; justify-content: flex-end;
                    align-items: center; gap: 12px; white-space: nowrap; }}
          .status {{ display: inline-flex; align-items: center; gap: 6px;
                     font-weight: 750; letter-spacing: .04em; color: #BAC7DA; }}
          .status.open {{ color: #69C58B; }}
          .dot {{ width: 9px; height: 9px; border-radius: 50%; background: #8491A5; }}
          .status.open .dot {{ background: #4ADE80; box-shadow: 0 0 8px #4ADE8070; }}
          .time {{ font-variant-numeric: tabular-nums; }}
        </style>
        <div class="clock" title="Regular NSE/BSE hours, weekdays; exchange holidays are not checked">
          <span id="market-status" class="status { 'open' if status_class == 'market-open' else '' }">
            <span class="dot"></span><span id="market-label">MARKET {status}</span>
          </span>
          <span id="india-time" class="time"></span>
        </div>
        <script>
          const formatter = new Intl.DateTimeFormat('en-IN', {{
            timeZone: 'Asia/Kolkata', weekday: 'short', hour: '2-digit',
            minute: '2-digit', second: '2-digit', hourCycle: 'h23'
          }});
          function updateMarketClock() {{
            const parts = Object.fromEntries(formatter.formatToParts(new Date()).map(part => [part.type, part.value]));
            const isWeekday = !['Sat', 'Sun'].includes(parts.weekday);
            const isOpen = isWeekday &&
              (Number(parts.hour) > 9 || (Number(parts.hour) === 9 && Number(parts.minute) >= 15)) &&
              (Number(parts.hour) < 15 || (Number(parts.hour) === 15 && Number(parts.minute) < 30));
            const status = document.getElementById('market-status');
            status.classList.toggle('open', isOpen);
            document.getElementById('market-label').textContent = `MARKET ${{isOpen ? 'OPEN' : 'CLOSED'}}`;
            document.getElementById('india-time').textContent =
              `${{parts.hour}}:${{parts.minute}}:${{parts.second}} IST`;
          }}
          updateMarketClock();
          setInterval(updateMarketClock, 1000);
        </script>
        """,
        height=58,
        scrolling=False,
    )


def _company_table(frame, key):
    if frame.empty:
        st.info("No companies match this view.")
        return
    table = frame.copy()
    base = str(st.get_option("server.baseUrlPath") or "").strip("/")
    prefix = f"/{base}" if base else ""
    table["Research"] = table["Ticker"].map(lambda ticker: f"{prefix}/stock_analyzer?ticker={quote(ticker)}")
    columns = ["Research", "Symbol", "Company", "Price", "Change %", "P/E", "P/B", "ROE %", "Sector", "As of"]
    st.dataframe(table[columns], hide_index=True, width="stretch", height=min(450, len(table) * 35 + 40), key=key,
                 column_config={"Research": st.column_config.LinkColumn("Research", display_text="Open"),
                                "Price": st.column_config.NumberColumn("Close (INR)", format="%.2f"),
                                "Change %": st.column_config.NumberColumn("1D change (%)", format="%+.2f"),
                                "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
                                "P/B": st.column_config.NumberColumn("P/B", format="%.2f"),
                                "ROE %": st.column_config.NumberColumn("ROE (%)", format="%.2f")})


@st.fragment(run_every="30s")
def _render_live_index_cards(fingerprint):
    indices = _live_index_snapshot()
    live_available = bool(indices)
    if not indices:
        history = index_history(fingerprint)
        if len(history) >= 2:
            close, previous = float(history["Close"].iloc[-1]), float(history["Close"].iloc[-2])
            indices = {"NIFTY 50": {"last": close, "delta": close - previous,
                                    "pct": (close / previous - 1) * 100,
                                    "as_of": f"Saved close · {history.index[-1]:%d %b %Y}"}}
    render_indices_ticker(indices)
    if not live_available:
        st.caption("Live index quotes are unavailable; the NIFTY 50 card shows the latest saved close.")


@st.fragment(run_every="30s")
def _benchmark(history):
    with st.container(border=True, key="benchmark_panel"):
        heading, controls = st.columns([2, 3])
        with heading:
            section_title("NIFTY 50", "Benchmark performance · latest quote + saved history")
        with controls:
            period = st.radio("Benchmark period", ["1M", "3M", "6M", "1Y"], index=1, horizontal=True, label_visibility="collapsed")
        if history.empty:
            st.info("Benchmark history is unavailable.")
            return
        subset = history.tail({"1M": 22, "3M": 66, "6M": 126, "1Y": 252}[period])
        subset = subset[["Close"]].copy()
        quote = _live_index_snapshot().get("NIFTY 50")
        quote_used = False
        quote_label = None
        if quote and quote.get("last") is not None and quote.get("as_of"):
            quote_time = pd.to_datetime(quote["as_of"], errors="coerce")
            if not pd.isna(quote_time):
                if quote_time.tzinfo is not None:
                    quote_time = quote_time.tz_convert(INDIA_TZ).tz_localize(None)
                quote_date = quote_time.normalize()
                history_dates = pd.DatetimeIndex(pd.to_datetime(subset.index))
                if history_dates.tz is not None:
                    history_dates = history_dates.tz_convert(INDIA_TZ).tz_localize(None)
                latest_history_date = history_dates[-1].normalize()
                current_time = datetime.now(INDIA_TZ)
                is_open = market_session_status(current_time)[0] == "OPEN"
                quote_is_current = not is_open or quote_date.date() == current_time.date()
                if quote_date >= latest_history_date and quote_is_current:
                    live_close = float(quote["last"])
                    if live_close > 0:
                        quote_day_matches_history = quote_date == latest_history_date
                        point_time = quote_time
                        if quote_day_matches_history:
                            subset.index = history_dates
                            subset.iloc[-1, subset.columns.get_loc("Close")] = live_close
                            if point_time > subset.index[-1]:
                                subset.index = subset.index[:-1].append(pd.DatetimeIndex([point_time]))
                        else:
                            subset.index = history_dates
                            subset.loc[point_time, "Close"] = live_close
                        subset = subset.sort_index()
                        quote_used = True
                        quote_label = quote_time.strftime("%d %b %H:%M IST")
        subset["Close"] = pd.to_numeric(subset["Close"], errors="coerce")
        subset = subset.dropna(subset=["Close"])
        close = subset["Close"]
        if close.empty:
            st.info("Benchmark history is unavailable.")
            return
        change = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100 if close.iloc[0] else 0
        tone = "positive" if change >= 0 else "negative"
        chart_color, fill_color = ("#00D47E", "rgba(0,212,126,.10)") if change >= 0 else ("#FF7187", "rgba(255,113,135,.10)")
        st.markdown(f'<span class="quote-price">{close.iloc[-1]:,.2f}</span><span class="quote-change {tone}">{change:+.2f}% <span class="muted">{period}</span></span>', unsafe_allow_html=True)
        figure = go.Figure(go.Scatter(x=subset.index, y=close, mode="lines", line={"color": chart_color, "width": 2},
                                     fill="tozeroy", fillcolor=fill_color, name="NIFTY 50"))
        figure.update_layout(template=CHART_TEMPLATE, **CHART_STYLE, height=230, margin={"l": 0, "r": 5, "t": 8, "b": 0},
                             xaxis={"showgrid": False}, yaxis={"side": "right", "range": [float(close.min()) * .994, float(close.max()) * 1.006]},
                             showlegend=False, hovermode="x unified")
        st.plotly_chart(figure, width="stretch", theme=None, config={"displayModeBar": False})
        if quote_used:
            st.caption(f"Latest provider quote: {quote_label} · polled every 30 seconds; earlier points use saved daily closes.")
        else:
            st.caption("Live quote unavailable or stale; showing saved daily closes.")


def _breadth(frame):
    with st.container(border=True, key="breadth_panel"):
        section_title("Coverage breadth", "Last session · companies tracked here", f"{len(frame)} stocks")
        if frame.empty:
            st.info("Refresh the local price datasets to populate coverage.")
            return
        up, down, flat = int((frame["Change %"] > 0).sum()), int((frame["Change %"] < 0).sum()), int((frame["Change %"] == 0).sum())
        total = max(1, up + down + flat)
        st.markdown(f'<div class="breadth-stats"><div>Advancing<strong class="positive">{up}</strong></div>'
                    f'<div>Declining<strong class="negative">{down}</strong></div><div>Unchanged<strong>{flat}</strong></div></div>'
                    f'<div class="breadth-bar" role="img" aria-label="{up} advancing, {down} declining, {flat} unchanged">'
                    f'<span style="width:{up / total * 100}%;background:#00D47E"></span>'
                    f'<span style="width:{down / total * 100}%;background:#FF7187"></span></div>', unsafe_allow_html=True)
        sectors = frame.groupby("Sector", as_index=False)["Change %"].mean().sort_values("Change %", ascending=True)
        figure = go.Figure(go.Bar(x=sectors["Change %"], y=sectors["Sector"], orientation="h",
                                 marker_color=["#00D47E" if value >= 0 else "#FF7187" for value in sectors["Change %"]]))
        figure.update_layout(template=CHART_TEMPLATE, **CHART_STYLE, height=212, margin={"l": 0, "r": 10, "t": 5, "b": 0},
                             xaxis={"ticksuffix": "%"}, yaxis={"tickfont": {"size": 10}}, showlegend=False)
        st.plotly_chart(figure, width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("Equal-weight sector moves within this coverage, not exchange-wide indices.")


def _open_latest_news():
    st.session_state["home_section"] = "Latest news"


def _news_panel(limit=5, show_all=True):
    with st.container(border=True, key=f"news_panel_{limit}"):
        heading, action = st.columns([2.1, 1])
        with heading:
            section_title("Market updates", "Recent verified source articles", "NEWS")
        if show_all:
            with action:
                st.button("View all →", key="open_latest_news", width="stretch", on_click=_open_latest_news)
        articles = cached_news(limit)
        if not articles:
            st.caption("No recent verified articles in the cache.")
        for article in articles:
            title = escape(str(article.get("title") or "Market update"))
            url = escape(str(article["url"]), quote=True)
            source = escape(str(article.get("source") or "Source"))
            date = escape(str(article.get("published_at") or "")[:10])
            st.markdown(f'<div class="news-row"><div class="news-source">{source} · {date}</div>'
                        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></div>', unsafe_allow_html=True)


def _live_index_snapshot():
    cached = st.session_state.get("_live_index_snapshot")
    if cached is None:
        current = fetch_index_snapshot()
        st.session_state["_live_index_snapshot"] = current
        return current
    if market_session_status()[0] == "OPEN":
        current = fetch_index_snapshot()
        if current:
            st.session_state["_live_index_snapshot"] = current
            return current
        return {}
    return cached or {}


def render_home_view():
    fingerprint = data_fingerprint()
    frame = load_dashboard_rows(fingerprint)
    names = dict(zip(frame["Ticker"], frame["Company"])) if not frame.empty else {}
    hero_title, market_clock = st.columns([3, 2])
    with hero_title:
        st.markdown(
            '<section class="home-hero">'
            '<h1 class="home-brand">Equimaster-Ai</h1>'
            '<p>Multi-Modal Deep Learning Platform for Financial Time-Series</p>'
            '</section>',
            unsafe_allow_html=True,
        )
    with market_clock:
        _render_market_clock(*market_session_status())
    st.markdown(
        '<div class="home-divider"></div>'
        '<div class="home-section-label">LIVE MARKET INDICES</div>',
        unsafe_allow_html=True,
    )

    _render_live_index_cards(fingerprint)
    render_sensex_ticker()

    search, refresh = st.columns([3, 1])
    with search:
        query = st.text_input("Search a stock", placeholder="Type a symbol or company name", label_visibility="collapsed")
        matches = []
        if query:
            lowered = query.strip().lower()
            matches = [
                ticker for ticker in names
                if lowered in ticker.lower() or lowered in names[ticker].lower()
            ][:8]
            if not matches:
                st.caption("No symbol or company matches that search. Try a broader term or use the screener.")
        if matches:
            for ticker in matches:
                if st.button(f"{ticker} · {names[ticker][:20]}", key=f"search_match_{ticker}", use_container_width=True):
                    st.session_state.selected_ticker = ticker
                    st.switch_page("pages/stock_analyzer.py", query_params={"ticker": ticker})
    with refresh:
        if st.button("Refresh", key="market_refresh", width="stretch"):
            fetch_index_snapshot.clear()
            load_dashboard_rows.clear()
            index_history.clear()
            st.session_state.pop("_live_index_snapshot", None)
            st.rerun()
    section = st.radio(
        "Dashboard section",
        ["Markets", "Stock screener", "Latest news"],
        index=0,
        key="home_section",
        horizontal=True,
        label_visibility="collapsed",
    )
    if section == "Markets":
        main, side = st.columns([1.7, 1])
        with main:
            _benchmark(index_history(fingerprint))
        with side:
            _breadth(frame)
        movers, updates = st.columns([2.1, 1])
        with movers:
            render_market_movers(frame)
        with updates:
            _news_panel()
    elif section == "Stock screener":
        with st.container(border=True, key="stock_screener_panel"):
            section_title("Discover companies", "Filter actual coverage using the parameters that matter to you.")
            sector, preset, query = st.columns([1.1, 1.4, 1.4])
            sectors = sorted(frame["Sector"].dropna().unique()) if not frame.empty else []
            selected_sector = sector.selectbox("Sector", ["All sectors"] + sectors)
            selected_preset = preset.selectbox("Screen", ["All companies", "Positive P/E under 20", "ROE at least 15%", "Debt/equity under 0.5x", "Positive 1-month return"])
            text = query.text_input("Company or symbol", placeholder="Search your coverage")
            filtered = apply_screen(frame, selected_sector, selected_preset, text)
            st.caption(f"{len(filtered)} companies · prices and fundamentals retain their own reporting dates")
            _company_table(filtered, "screener_table")
            st.download_button("Export screener", filtered.drop(columns=["Financial sector"], errors="ignore").to_csv(index=False), "equimaster_screener.csv", "text/csv")
    else:
        _news_panel(20, show_all=False)
