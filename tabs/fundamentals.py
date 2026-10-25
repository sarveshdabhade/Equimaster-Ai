"""Comprehensive company fundamentals with transparent ratio ranges."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import ALL_DOWNLOAD_TICKERS
from components.layout import CHART_STYLE, CHART_TEMPLATE
from src.artifacts import canonical_ticker
from src.fundamentals import (
    compare_peers, valuation_summary, read_cached_details, snapshot_from_details, details_from_snapshot,
    assess_metric, peer_reference, format_value, SOURCES, CACHE_HOURS,
)

COLORS = {"favorable": ("#10382C", "#92FE9D"), "concern": ("#3D1F2B", "#FFAFBB"),
          "caution": ("#3A301C", "#FFE0A0"), "neutral": ("#133147", "#9AEAFF")}


def _cached_frame(tickers):
    return pd.DataFrame([snapshot_from_details(details) for ticker in tickers
                         if (details := read_cached_details(ticker)) is not None])


def _metric_rows(details, peers):
    rows, tones = [], []
    for key, metric in details["metrics"].items():
        assessment = assess_metric(key, metric, details, peer_reference(peers, details, key))
        rows.append({"Category": metric["group"], "Parameter": metric["label"],
                     "Value": format_value(metric["value"], metric["unit"], metric.get("currency") or details.get("Currency") or "INR"),
                     "Assessment": assessment["assessment"], "Reference range": assessment["range"],
                     "Period": metric.get("period") or "N/A", "Source": metric.get("source") or "Unavailable",
                     "Basis": metric.get("basis") or "N/A", "Explanation / formula": metric.get("note") or assessment["reason"],
                     "Raw value": metric["value"], "Unit": metric.get("currency") or metric["unit"]})
        tones.append(assessment["tone"])
    return pd.DataFrame(rows), tones


def _statement_view(table, name, ticker):
    st.caption(f"{table['source']} · {table['basis']} · Amounts: {table['unit']}. EPS is per share; % rows remain percentages.")
    rows = [{"Parameter": row["label"], "Unit": row["unit"], **dict(zip(table["periods"], row["values"]))}
            for row in table["rows"]]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(f"Export {name}", frame.to_csv(index=False), f"{ticker}_{name}.csv", "text/csv", key=f"statement_{ticker}_{name}")
    return frame


def render_fundamentals_tab(ticker):
    st.subheader("Company Fundamentals & Peer Comparison")
    ticker = canonical_ticker(ticker)
    if ticker == "NSEI":
        st.info("Company valuation metrics apply to individual stocks. Select a company to compare peers.")
        return
    details = read_cached_details(ticker)
    options = [t for t in ALL_DOWNLOAD_TICKERS if t != ticker]
    defaults = []
    if details and details.get("Industry"):
        for candidate in options:
            cached = read_cached_details(candidate)
            if cached and cached.get("Industry") == details["Industry"]:
                defaults.append(candidate)
            if len(defaults) == 3:
                break
    peers = st.multiselect("Compare companies", options, default=defaults, max_selections=8, key=f"peers_{ticker}")
    selection = (ticker, tuple(peers))
    if st.button("Refresh Company Data", key=f"fundamentals_{ticker}"):
        with st.spinner("Loading company ratios, statements and ownership..."):
            frame, errors = compare_peers([ticker] + peers, refresh=True)
        st.session_state["fundamentals_data"] = (frame, errors)
        st.session_state["fundamentals_selection"] = selection
    if st.session_state.get("fundamentals_selection") == selection:
        frame, errors = st.session_state["fundamentals_data"]
        for error in errors:
            st.warning(error)
        if not frame.empty and ticker in frame["Ticker"].values:
            details = details_from_snapshot(frame.loc[frame["Ticker"] == ticker].iloc[0].to_dict())
    else:
        frame = _cached_frame([ticker] + peers)
    if not details:
        st.info("Refresh Company Data to fetch company ratios, financial statements and shareholding history.")
        return
    st.markdown(f"**{details['Company']}** · {details.get('Sector') or 'Sector unavailable'} · {details.get('Industry') or 'Industry unavailable'}")
    try:
        fetched = datetime.fromisoformat(details["As of"])
        fetched = fetched.replace(tzinfo=timezone.utc) if fetched.tzinfo is None else fetched
        st.caption(f"Fetched {fetched.astimezone(ZoneInfo('Asia/Kolkata')):%d %b %Y, %I:%M %p IST}. Financial reporting periods are shown per parameter.")
        if (datetime.now(timezone.utc) - fetched).total_seconds() > CACHE_HOURS * 3600:
            st.warning("This saved snapshot is over 24 hours old. Refresh to check for new disclosures.")
    except (ValueError, TypeError):
        st.caption("Snapshot retrieval time unavailable; check reporting periods below.")
    for warning in details.get("warnings", []):
        st.warning(warning)
    available = sum(metric["value"] is not None for metric in details["metrics"].values())
    st.caption(f"{available}/{len(details['metrics'])} parameters available. Missing disclosures remain N/A; fetched today does not mean the financial statements were published today.")
    links = st.columns(2)
    if details["sources"].get("screener"):
        links[0].link_button("Open Screener source", details["sources"]["screener"])
    links[1].link_button("Open Yahoo source", details["sources"]["yahoo"])
    keys = ["market_cap", "pe", "pb", "dividend_yield", "roe", "roce", "debt_equity", "eps"]
    for offset in (0, 4):
        columns = st.columns(4)
        for column, key in zip(columns, keys[offset:offset + 4]):
            metric = details["metrics"][key]
            column.metric(metric["label"], format_value(metric["value"], metric["unit"], metric.get("currency") or details.get("Currency") or "INR"))
    st.caption("Ranges are screening heuristics, not universal good/bad rules. With two suitable peers, valuation bands use 80–120% of their median. Low P/E or P/B alone does not establish good value.")
    overview, statements, ownership, comparison, methodology = st.tabs([
        "All Parameters & Ranges", "Financial Statements", "Shareholding", "Peer Comparison", "Definitions & Sources"
    ])
    data, tones = _metric_rows(details, frame)
    with overview:
        groups = list(dict.fromkeys(data["Category"]))
        # Banking disclosures are useful for lenders, and otherwise omitted from the default view.
        default_groups = [g for g in groups if g != "Financial-sector disclosures" or details.get("financial_sector")]
        selected = st.multiselect("Parameter categories", groups, default=default_groups, key=f"fundamental_groups_{ticker}")
        show_missing = st.checkbox("Show unavailable parameters", value=True, key=f"fundamental_missing_{ticker}")
        for category in selected:
            subset = data[data["Category"] == category]
            if not show_missing:
                subset = subset[subset["Raw value"].notna()]
            with st.expander(category, expanded=category in ("Valuation", "Profitability")):
                if subset.empty:
                    st.info("No values available for this category.")
                    continue
                shown = subset[["Parameter", "Value", "Assessment", "Reference range", "Period", "Source", "Basis"]]
                styled = shown.style.apply(lambda row: [f"background-color: {COLORS[tones[row.name]][0]}; color: {COLORS[tones[row.name]][1]}" if name == "Assessment" else "" for name in shown.columns], axis=1)
                st.dataframe(styled, hide_index=True, width="stretch", column_config={
                    "Parameter": st.column_config.TextColumn(width=260),
                    "Value": st.column_config.TextColumn(width=180),
                    "Assessment": st.column_config.TextColumn(width=170),
                    "Reference range": st.column_config.TextColumn(width=340),
                    "Period": st.column_config.TextColumn(width=170),
                    "Source": st.column_config.TextColumn(width=200),
                    "Basis": st.column_config.TextColumn(width=180),
                })
        st.download_button("Export All Fundamentals", data.to_csv(index=False), f"{ticker}_all_fundamentals.csv", "text/csv")
        st.download_button("Export Full Company JSON", json.dumps(details, indent=2), f"{ticker}_fundamentals.json", "application/json")
    with statements:
        names = {"quarters": "Quarterly results", "profit-loss": "Annual profit and loss", "balance-sheet": "Balance sheet", "cash-flow": "Cash flow", "ratios": "Historical ratios"}
        for section, name in names.items():
            with st.expander(name, expanded=section == "quarters"):
                table = details["statements"].get(section)
                if table:
                    _statement_view(table, name, ticker)
                else:
                    st.info("This statement was not supplied by the available sources.")
    with ownership:
        subset = data[data["Category"] == "Ownership"]
        st.dataframe(subset[["Parameter", "Value", "Period", "Source"]], hide_index=True, width="stretch")
        st.caption("Promoters, FII and DII are Screener disclosure categories. Yahoo insiders/institutions use different definitions and are not substitutes. Pledged shares stay N/A unless explicitly disclosed.")
        table = details["statements"].get("shareholding")
        if table:
            _statement_view(table, "Shareholding history", ticker)
            chart = go.Figure()
            for row in table["rows"]:
                if row["unit"] == "percent":
                    chart.add_trace(go.Scatter(x=table["periods"], y=row["values"], mode="lines+markers", name=row["label"]))
            chart.update_layout(yaxis_title="Holding (%)", template=CHART_TEMPLATE, **CHART_STYLE)
            st.plotly_chart(chart, width="stretch", theme=None)
    with comparison:
        st.write(valuation_summary(frame, ticker))
        if not frame.empty:
            visible = frame.drop(columns=["_details"], errors="ignore").copy()
            if "_details" in frame:
                for field, label in (("source", "P/E source"), ("period", "P/E period"), ("basis", "Reporting basis")):
                    visible[label] = frame["_details"].map(lambda value: value["metrics"]["pe"].get(field) if isinstance(value, dict) else "Unknown")
            for name in ("ROE", "ROCE", "Profit margin", "Dividend yield"):
                if name in visible:
                    visible[name] = visible[name].map(lambda value: format_value(value * 100, "percent") if pd.notna(value) else "N/A")
            st.dataframe(visible, hide_index=True, width="stretch")
            st.download_button("Export Peer Comparison", visible.to_csv(index=False), "peer_comparison.csv", "text/csv")
        st.caption("Only same-industry peers with matching metric sources, reported-period labels and consolidated/standalone basis contribute to valuation ranges. Two valid peers are required.")
    with methodology:
        st.dataframe(data[["Parameter", "Unit", "Period", "Source", "Basis", "Explanation / formula"]], hide_index=True, width="stretch")
        if details.get("financial_sector"):
            st.info("Banks, insurers and lenders need sector-specific analysis. Generic industrial-company leverage, liquidity, margins, ROCE and cash-flow ratings are disabled.")
        st.write("TTM means trailing twelve months. Forward metrics are estimates. Monetary totals use INR crore where applicable (1 crore = 10,000,000); EPS and book value are per share.")
        st.write("Each monetary value identifies its currency. USD financial data is not labeled INR, and unverified cross-currency valuation multiples are withheld. No guessed exchange-rate conversion is applied.")
        st.write("CAGR requires positive endpoints and the actual fiscal-year span; TTM is excluded. Derived annual ratios use matching fiscal dates. Published ROCE and calculated ROCE can differ in definition.")
        for label, url in SOURCES.items():
            st.markdown(f"[{label}]({url})")
