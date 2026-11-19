"""Comprehensive company fundamentals with transparent ratio ranges."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import ALL_DOWNLOAD_TICKERS
from components.layout import CHART_STYLE, CHART_TEMPLATE, research_card
from src.artifacts import canonical_ticker
from src.fundamentals import (
    compare_peers, valuation_summary, read_cached_details, snapshot_from_details, details_from_snapshot,
    assess_metric, peer_reference, format_value, SOURCES, CACHE_HOURS,
)
from src.fundamental_swot import analyze_fundamental_swot, load_valuation_assumptions
from src.fundamental_ml import read_model_status

COLORS = {"favorable": ("#10382C", "#92FE9D"), "concern": ("#3D1F2B", "#FFAFBB"),
          "caution": ("#3A301C", "#FFE0A0"), "neutral": ("#133147", "#9AEAFF")}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fundamentals_by_ticker():
    return {ticker: read_cached_details(ticker) for ticker in ALL_DOWNLOAD_TICKERS}


def _cached_frame(tickers, details_by_ticker):
    rows = []
    for ticker in tickers:
        details = details_by_ticker.get(ticker)
        if details is None:
            details = read_cached_details(ticker)
        if details is not None:
            rows.append(snapshot_from_details(details))
    return pd.DataFrame(rows)


def _parse_peer_symbols(raw, subject_ticker):
    peers, invalid = [], []
    for token in re.split(r"[,;\s]+", raw or ""):
        if not token:
            continue
        symbol = canonical_ticker(token.strip())
        if "." not in symbol:
            symbol += ".NS"
        if not re.fullmatch(r"[A-Z0-9&_-]+\.(?:NS|BO)", symbol):
            invalid.append(token)
        elif symbol != subject_ticker and symbol not in peers:
            peers.append(symbol)
    return peers, invalid


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


def _render_swot_view(ticker, details, peers, current_price):
    st.subheader("Fundamental SWOT & Valuation")
    st.caption("Evidence-based SWOT, sector-aware fair value, and a point-in-time return model.")
    company_beta = details.get("metrics", {}).get("beta", {}).get("value")
    try:
        company_beta = float(company_beta) if company_beta is not None else None
    except (TypeError, ValueError):
        company_beta = None
    assumptions = load_valuation_assumptions(ticker, company_beta)

    with st.spinner("Building SWOT from published fundamentals, selected peers and verified news..."):
        from src.news.sentiment import analyze_ticker_sentiment
        news = analyze_ticker_sentiment(ticker, max_articles=20, include_llm=False)
        result = analyze_fundamental_swot(ticker, details, peers, current_price, news, assumptions)

    fair_value = result.get("fair_value")
    upside = result.get("upside")
    score = result.get("swot_score")
    model = result.get("ml", {})
    expected_alpha = model.get("expected_excess_return") if model.get("available") else None
    metrics = st.columns(4)
    metrics[0].metric("Fair value", format_value(fair_value, "money", details.get("Currency") or "INR") if fair_value is not None else "Unavailable",
                      result.get("valuation_status", "unavailable").capitalize())
    metrics[1].metric("Value status", result.get("value_status", "Unavailable"),
                      f"{upside:+.1%} vs price" if upside is not None else "No supported fair-value estimate")
    metrics[2].metric("Raw SWOT balance", f"{score:+.0f}/100" if score is not None else "N/A",
                      result.get("swot_label", "Insufficient evidence"))
    metrics[3].metric("12-month excess return", f"{expected_alpha:+.1%}" if expected_alpha is not None else "Warming up",
                      "vs NIFTY 50" if expected_alpha is not None else "No validated model yet")
    if score is not None:
        st.caption(
            f"Raw evidence balance across {result.get('factor_count', 0)} factors and "
            f"{result.get('score_family_count', 0)} families; related factors are averaged within each family. "
            "It is a screening summary, not a calibrated rating or return prediction."
        )

    if result["action"] == "PENDING ML DATA":
        status = model.get("training_status", read_model_status())
        snapshots = int(status.get("snapshots_archived", 0) or 0)
        months = int(status.get("archived_months", 0) or 0)
        month_word = "month" if months == 1 else "months"
        st.info(
            f"**Recommendation pending ML data.** {result['action_reason']} "
            f"The point-in-time archive currently has {snapshots} dated company snapshots "
            f"across {months} {month_word}; recommendations unlock only after 12-month outcomes "
            "mature and the return model passes walk-forward validation."
        )
    elif result["action"] == "BUY":
        st.success(f"**BUY** · {result['action_reason']}")
    elif result["action"] == "SELL-TO-EXIT":
        st.error(f"**SELL-TO-EXIT** · {result['action_reason']}")
    else:
        st.warning(f"**HOLD** · {result['action_reason']}")

    st.caption(
        f"Valuation coverage: {result['valuation_status']}. A full estimate blends DCF at {result['dcf_weight']:.0%} "
        f"and peers at {result['peer_weight']:.0%}; partial coverage cannot trigger BUY."
    )
    st.caption(
        f"DCF: {result['dcf'].get('method', 'unavailable')} — {result['dcf'].get('reason', '')} "
        f"Peer value: {result['peer_value'].get('method', 'unavailable')} — {result['peer_value'].get('reason', '')}"
    )
    st.markdown("#### SWOT evidence")
    quadrants = st.columns(4)
    for column, label in zip(quadrants, ("Strengths", "Weaknesses", "Opportunities", "Threats")):
        with column:
            st.markdown(f"**{label}**")
            items = result["swot"].get(label, [])
            if not items:
                st.caption("No supported factor found from current inputs.")
            for item in items:
                st.markdown(f"- {item['title']}")
                st.caption(f"{item['evidence']} Source: {item['source']}.")

    st.markdown("#### Valuation & profitability parameters considered")
    review = result.get("valuation_profitability_review", [])
    if review:
        review_frame = pd.DataFrame(review)
        currency = details.get("Currency") or "INR"
        for column in ("Value", "Peer median"):
            review_frame[column] = review_frame.apply(
                lambda row: format_value(row[column], row["Unit"], currency)
                if pd.notna(row[column]) else "N/A", axis=1)
        st.dataframe(review_frame.drop(columns=["Unit"]), hide_index=True, width="stretch")
        st.caption("Each available valuation and profitability metric is reviewed. Strengths and weaknesses reflect a material same-industry peer difference where possible; otherwise the displayed application screening guide is used. Enterprise value is shown as scale context, not scored on size alone.")
    else:
        st.caption("No valuation or profitability parameters are currently available to review.")

    model_status = model.get("training_status", read_model_status())
    with st.expander("ML readiness and scoring method"):
        st.write({
            "model": model_status.get("model", "HistGradientBoostingRegressor"),
            "target": model_status.get("target", "252-session stock return minus NIFTY 50 return"),
            "archived snapshots": model_status.get("snapshots_archived", "Collecting monthly observations"),
            "matured labeled samples": model_status.get("samples", 0),
            "matured months": model_status.get("matured_months", 0),
            "status": model_status.get("reason", "Training data is still accumulating."),
        })
        st.write("Raw SWOT balance (−100 to +100) is calculated by averaging directional factors within evidence families, then averaging the available family balances equally. It is not a calibrated rating or probability. Generic screening guides may still be used when peers are missing, so treat the balance as a triage summary. BUY requires a full DCF/peer value, at least 15% discount, and a validated model estimate of at least +5% 12-month excess return. SELL-TO-EXIT requires a validated model estimate at or below −5%, or full-value downside of at least 15%.")
        st.write("The model trains only on monthly archived fundamentals whose 252-session outcome has matured. Validation uses expanding time windows with overlapping 12-month outcomes purged; it must beat a zero-excess-return baseline and show positive rank correlation before recommendations are enabled.")


def _build_universe_swot(details_by_ticker):
    from src.news.fetcher import read_cached_ticker_articles
    from src.news.sentiment import analyze_ticker_sentiment

    available = {ticker: details_by_ticker.get(ticker) for ticker in ALL_DOWNLOAD_TICKERS
                 if details_by_ticker.get(ticker)}
    peer_frame = _cached_frame(list(available), details_by_ticker)
    rows = []
    for stock in ALL_DOWNLOAD_TICKERS:
        details = available.get(stock)
        if not details:
            rows.append({"Ticker": stock, "Company": "Unavailable", "Industry": "N/A",
                         "Current price": None, "Fair value": None, "Value status": "Unavailable",
                         "Upside": None, "Raw SWOT balance": None, "Evidence balance label": "Unavailable",
                         "Recommendation": "Unavailable", "Valuation coverage": "Unavailable",
                         "News articles": 0, "News sentiment": "N/A"})
            continue
        price = details.get("metrics", {}).get("price", {}).get("value")
        beta = details.get("metrics", {}).get("beta", {}).get("value")
        try:
            beta = float(beta) if beta is not None else None
        except (TypeError, ValueError):
            beta = None
        cached_articles = read_cached_ticker_articles(stock)
        news = analyze_ticker_sentiment(stock, max_articles=20, articles=cached_articles, include_llm=False)
        result = analyze_fundamental_swot(
            stock, details, peer_frame, price, news, load_valuation_assumptions(stock, beta)
        )
        rows.append({
            "Ticker": stock,
            "Company": details.get("Company", stock),
            "Industry": details.get("Industry") or "N/A",
            "Current price": result.get("current_price"),
            "Fair value": result.get("fair_value"),
            "Value status": result.get("value_status"),
            "Upside": result.get("upside"),
            "Raw SWOT balance": result.get("swot_score"),
            "Evidence balance label": result.get("swot_label"),
            "Recommendation": result.get("action"),
            "Valuation coverage": result.get("valuation_status"),
            "News articles": len(news.get("articles", [])),
            "News sentiment": news.get("label", "No evidence"),
        })
    return pd.DataFrame(rows)


def _render_universe_swot(details_by_ticker):
    st.subheader("All-stock SWOT & Valuation")
    st.caption(f"Run the same SWOT, peer valuation, and recommendation rules across all {len(ALL_DOWNLOAD_TICKERS)} configured stocks. This uses cached fundamentals and cached news; it does not initiate live provider requests.")
    if st.button("Calculate all-stock SWOT & valuation", key="calculate_all_stock_swot"):
        with st.spinner(f"Calculating SWOT and valuation across {len(ALL_DOWNLOAD_TICKERS)} stocks..."):
            st.session_state["all_stock_swot_results"] = _build_universe_swot(details_by_ticker)
    results = st.session_state.get("all_stock_swot_results")
    if results is None:
        st.info("Select Calculate to build the all-stock comparison.")
        return
    shown = results.copy()
    for column in ("Current price", "Fair value"):
        shown[column] = shown[column].map(lambda value: format_value(value, "money") if pd.notna(value) else "N/A")
    shown["Upside"] = shown["Upside"].map(lambda value: f"{value:+.1%}" if pd.notna(value) else "N/A")
    shown["Raw SWOT balance"] = shown["Raw SWOT balance"].map(lambda value: f"{value:+.0f}/100" if pd.notna(value) else "N/A")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Price and fundamentals come from saved company snapshots. The raw SWOT balance averages signals within evidence families and is not calibrated. News sentiment uses only recent cached articles. Missing model data keeps recommendations pending; the table does not manufacture inputs.")


def _render_snapshot_cards(details, price_history=None):
    metrics = details.get("metrics", {})
    roe, margin, pe, pb = (metrics.get(key, {}) for key in ("roe", "net_margin", "pe", "pb"))
    quality, valuation, momentum = st.columns(3)
    with quality:
        research_card("Financial snapshot", f"{format_value(roe.get('value'), 'percent')} ROE",
                      f"Net margin {format_value(margin.get('value'), 'percent')} | published fundamentals", "green")
    with valuation:
        assessment = assess_metric("pe", pe, details)
        research_card("Valuation context", f"{format_value(pe.get('value'), 'multiple')} P/E",
                      f"{assessment['assessment']} | P/B {format_value(pb.get('value'), 'multiple')}", "amber")
    return_1m = None
    rsi = None
    if price_history is not None and len(price_history) >= 22:
        prior_close = price_history["Close"].iloc[-22]
        latest_close = price_history["Close"].iloc[-1]
        if pd.notna(prior_close) and float(prior_close) != 0:
            return_1m = (float(latest_close) / float(prior_close) - 1) * 100
    if price_history is not None and "RSI" in price_history and len(price_history):
        rsi_value = pd.to_numeric(price_history["RSI"], errors="coerce").iloc[-1]
        rsi = float(rsi_value) if pd.notna(rsi_value) else None
    with momentum:
        research_card("Price momentum", format_value(return_1m, "percent"),
                      f"21-session return | RSI {f'{rsi:.1f}' if rsi is not None else 'N/A'}")


def render_fundamentals_tab(ticker, current_price=None, price_history=None):
    st.subheader("Company Fundamentals & Peer Comparison")
    ticker = canonical_ticker(ticker)
    if ticker == "NSEI":
        st.info("Company valuation metrics apply to individual stocks. Select a company to compare peers.")
        return
    details_by_ticker = _cached_fundamentals_by_ticker()
    details = details_by_ticker.get(ticker) or read_cached_details(ticker)
    options = [t for t in ALL_DOWNLOAD_TICKERS if t != ticker]
    defaults = []
    if details and details.get("Industry"):
        for candidate in options:
            cached = details_by_ticker.get(candidate)
            if cached and cached.get("Industry") == details["Industry"]:
                defaults.append(candidate)
            if len(defaults) == 3:
                break
    selected_peers = st.multiselect("Compare companies", options, default=defaults, max_selections=8, key=f"peers_{ticker}")
    if details and details.get("Industry") and not defaults:
        st.info(
            f"No cached company in the configured universe shares the exact industry '{details['Industry']}'. "
            "Add public peer symbols below and refresh to fetch their fundamentals. Peer estimates still require at least two eligible matches."
        )
    extra_peer_text = st.text_input(
        "Additional peer symbols (comma-separated; .NS is assumed when omitted)",
        placeholder="NYKAA.NS, SWIGGY.NS",
        key=f"extra_peers_{ticker}",
    )
    extra_peers, invalid_peers = _parse_peer_symbols(extra_peer_text, ticker)
    if invalid_peers:
        st.warning(f"Ignored invalid peer symbol(s): {', '.join(invalid_peers)}")
    peers = list(dict.fromkeys(selected_peers + extra_peers))
    if len(peers) > 8:
        st.warning("At most eight peer symbols are used; the extra symbols were ignored.")
        peers = peers[:8]
    selection = (ticker, tuple(peers))
    if st.button("Refresh Company Data", key=f"fundamentals_{ticker}"):
        with st.spinner("Loading company ratios, statements and ownership..."):
            frame, errors = compare_peers([ticker] + peers, refresh=True)
        _cached_fundamentals_by_ticker.clear()
        st.session_state["fundamentals_data"] = (frame, errors)
        st.session_state["fundamentals_selection"] = selection
    if st.session_state.get("fundamentals_selection") == selection:
        frame, errors = st.session_state["fundamentals_data"]
        for error in errors:
            st.warning(error)
        if not frame.empty and ticker in frame["Ticker"].values:
            details = details_from_snapshot(frame.loc[frame["Ticker"] == ticker].iloc[0].to_dict())
    else:
        frame = _cached_frame([ticker] + peers, details_by_ticker)
    if not details:
        st.info("Refresh Company Data to fetch company ratios, financial statements and shareholding history.")
        return
    _render_snapshot_cards(details, price_history)
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
    _render_swot_view(ticker, details, frame, current_price)
    _render_universe_swot(details_by_ticker)
    st.divider()
    st.caption("Ranges are screening heuristics, not universal good/bad rules. With two suitable peers, valuation bands use 80–120% of their median. Low P/E or P/B alone does not establish good value.")
    view = st.radio(
        "Fundamentals section",
        ["Parameters", "Statements", "Shareholding", "Peer Comparison", "Methodology"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"fundamentals_view_{ticker}",
    )
    if view == "Parameters":
        data, tones = _metric_rows(details, frame)
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
    elif view == "Statements":
        names = {"quarters": "Quarterly results", "profit-loss": "Annual profit and loss", "balance-sheet": "Balance sheet", "cash-flow": "Cash flow", "ratios": "Historical ratios"}
        for section, name in names.items():
            with st.expander(name, expanded=section == "quarters"):
                table = details["statements"].get(section)
                if table:
                    _statement_view(table, name, ticker)
                else:
                    st.info("This statement was not supplied by the available sources.")
    elif view == "Shareholding":
        data, _ = _metric_rows(details, frame)
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
    elif view == "Peer Comparison":
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
    else:
        data, _ = _metric_rows(details, frame)
        st.dataframe(data[["Parameter", "Unit", "Period", "Source", "Basis", "Explanation / formula"]], hide_index=True, width="stretch")
        if details.get("financial_sector"):
            st.info("Banks, insurers and lenders need sector-specific analysis. Generic industrial-company leverage, liquidity, margins, ROCE and cash-flow ratings are disabled.")
        st.write("TTM means trailing twelve months. Forward metrics are estimates. Monetary totals use INR crore where applicable (1 crore = 10,000,000); EPS and book value are per share.")
        st.write("Each monetary value identifies its currency. USD financial data is not labeled INR, and unverified cross-currency valuation multiples are withheld. No guessed exchange-rate conversion is applied.")
        st.write("CAGR requires positive endpoints and the actual fiscal-year span; TTM is excluded. Derived annual ratios use matching fiscal dates. Published ROCE and calculated ROCE can differ in definition.")
        for label, url in SOURCES.items():
            st.markdown(f"[{label}]({url})")
