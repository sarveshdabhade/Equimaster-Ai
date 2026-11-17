"""Cached comprehensive fundamentals and comparable peer references."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import pandas as pd
import yfinance as yf
from config import resolve_app_path, provider_symbol
from src.artifacts import atomic_json, canonical_ticker
from src.fundamental_sources import number, fetch_screener, statement_currencies
from src.fundamental_calculations import build_details, CRORE
from src.fundamental_catalog import SPECS, RANGES, FINANCIAL_EXCLUSIONS
from src.fundamental_history import archive_snapshot

SCHEMA_VERSION = 4
CACHE_HOURS = 24
logger = logging.getLogger(__name__)
SOURCES = {
    "Ratio interpretation": "https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/financial-analysis-techniques",
    "Industry context": "https://www.fidelity.com/learning-center/trading-investing/fundamentals",
    "ROCE methodology": "https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f7/technical-articles/performance-appraisal.html",
}
yf.set_tz_cache_location(resolve_app_path("data", ".yf_cache"))


def read_cached_details(ticker):
    path = Path(resolve_app_path("data", "fundamentals", f"{canonical_ticker(ticker)}.json"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        missing_metrics = {key for key in ("net_borrowing", "fcfe") if key not in payload.get("metrics", {})}
        if missing_metrics:
            payload = dict(payload)
            payload["metrics"] = dict(payload.get("metrics", {}))
            for key in missing_metrics:
                payload["metrics"][key] = {
                    **SPECS[key], "value": None, "source": None, "period": None, "basis": None,
                    "note": "Refresh this snapshot to calculate same-period net borrowing and FCFE inputs.",
                }
        return payload
    except (OSError, ValueError):
        return None


def company_details(ticker, refresh=False):
    ticker = canonical_ticker(ticker)
    if ticker == "NSEI":
        raise ValueError("Company fundamentals do not apply to an index")
    cached = read_cached_details(ticker)
    if cached and not refresh:
        return cached
    warnings, info, screen, annual = [], {}, None, {}
    client = yf.Ticker(provider_symbol(ticker))

    def fetch_yahoo():
        result = client.get_info()
        if not result or not result.get("longName"):
            raise ValueError("No company profile returned")
        return result

    jobs = {"profile": fetch_yahoo, "screener": lambda: fetch_screener(ticker),
            "balance": lambda: client.get_balance_sheet(), "income": lambda: client.get_income_stmt(),
            "cashflow": lambda: client.get_cash_flow()}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {name: executor.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            try:
                result = future.result()
                if name == "profile":
                    info = result
                elif name == "screener":
                    screen = result
                else:
                    annual[name] = result.sort_index(axis=1, ascending=False) if not result.empty else result
            except Exception as exc:
                warnings.append(f"{name}: {type(exc).__name__}: data unavailable")
    if not info and not screen:
        if cached:
            return {**cached, "warnings": cached.get("warnings", []) + ["Refresh failed; showing the previously saved snapshot"]}
        raise ValueError(f"No fundamentals returned for {ticker}")
    if info.get("currency") and info.get("financialCurrency") and info["currency"] != info["financialCurrency"]:
        try:
            annual["currencies"] = statement_currencies(client, ticker)
        except Exception:
            warnings.append("Statement currency could not be verified; mixed-currency calculations are withheld")
    details = build_details(ticker, info, screen, annual, warnings)
    # A missing public table can fall back to Yahoo, with its original units and basis.
    for name, key in (("income", "profit-loss"), ("balance", "balance-sheet"), ("cashflow", "cash-flow")):
        frame = annual.get(name)
        if key not in details["statements"] and frame is not None and not frame.empty:
            details["statements"][key] = {
                "periods": [str(x.date()) for x in frame.columns],
                "rows": [{"label": str(label), "values": [number(x) for x in row],
                          "unit": "money_per_share" if "EPS" in str(label) else "number" if "Shares" in str(label) else "base currency"}
                         for label, row in frame.iterrows()],
                "unit": "base currency", "basis": "Yahoo annual statements", "source": "Yahoo Finance"}
    atomic_json(Path(resolve_app_path("data", "fundamentals", f"{ticker}.json")), details)
    try:
        archive_snapshot(details)
    except Exception:
        logger.exception("Could not archive point-in-time fundamentals for %s", ticker)
    return details


def snapshot_from_details(details):
    metrics = details["metrics"]
    flat = {key: details.get(key) for key in ("Ticker", "Company", "Industry", "Sector", "Currency", "As of")}
    for key, label, fraction in (
        ("pe", "P/E", False), ("forward_pe", "Forward P/E", False), ("pb", "P/B", False),
        ("roe", "ROE", True), ("roce", "ROCE", True), ("net_margin", "Profit margin", True),
        ("debt_equity", "Debt/equity", False), ("market_cap", "Market cap", False), ("dividend_yield", "Dividend yield", True),
    ):
        value = metrics[key]["value"]
        flat[label] = value / 100 if value is not None and fraction else value
    flat["_details"] = details
    return flat


def details_from_snapshot(snapshot):
    """Support callers supplying the previous flat snapshot contract."""
    if isinstance(snapshot.get("_details"), dict):
        return snapshot["_details"]
    details = build_details(snapshot.get("Ticker", ""), {})
    for field in ("Ticker", "Company", "Industry", "Sector", "Currency", "As of"):
        if snapshot.get(field) is not None:
            details[field] = snapshot[field]
    for label, key, fraction in (("P/E", "pe", False), ("P/B", "pb", False), ("ROE", "roe", True),
                                  ("ROCE", "roce", True), ("Debt/equity", "debt_equity", False)):
        value = number(snapshot.get(label))
        if value is not None:
            details["metrics"][key].update(value=value * 100 if fraction else value, source="Legacy snapshot", period="Unknown", basis="Unknown")
    return details


def company_snapshot(ticker, refresh=False):
    return snapshot_from_details(company_details(ticker, refresh))


def compare_peers(tickers, refresh=False):
    rows, errors = [], []

    def fetch(ticker):
        try:
            return company_snapshot(ticker, refresh), None
        except Exception as exc:
            return None, f"{ticker}: {exc}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        for row, error in executor.map(fetch, dict.fromkeys(tickers)):
            if row:
                rows.append(row)
            if error:
                errors.append(error)
    return pd.DataFrame(rows), errors


def peer_reference(frame, details, key):
    if frame.empty or not details.get("Industry") or key not in details.get("metrics", {}):
        return []
    subject, values = details["metrics"][key], []
    if subject.get("basis") not in ("Consolidated", "Standalone"):
        return []
    for _, row in frame.iterrows():
        peer = row.get("_details")
        if not isinstance(peer, dict) or row.get("Ticker") == details["Ticker"] or peer.get("Industry") != details["Industry"]:
            continue
        try:
            if (datetime.now(timezone.utc) - datetime.fromisoformat(peer["As of"])).total_seconds() > CACHE_HOURS * 3600:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        metric = peer.get("metrics", {}).get(key, {})
        if any(metric.get(field) != subject.get(field) for field in ("source", "period", "basis")):
            continue
        value = number(metric.get("value"))
        if value is not None and value > 0:
            values.append(value)
    return values


def valuation_summary(frame, ticker):
    if frame.empty or ticker not in frame["Ticker"].values:
        return "No company snapshot available."
    subject = frame.loc[frame["Ticker"] == ticker].iloc[0]
    if pd.isna(subject.get("Industry")):
        return "Industry unavailable; a same-industry comparison cannot be calculated."
    if isinstance(subject.get("_details"), dict):
        values = pd.Series(peer_reference(frame, subject["_details"], "pe"), dtype=float)
    else:
        values = pd.to_numeric(frame.loc[(frame["Ticker"] != ticker) & (frame["Industry"] == subject["Industry"]), "P/E"], errors="coerce")
        values = values[values > 0]
    pe = number(subject.get("P/E"))
    if pe is None or pe <= 0 or len(values) < 2:
        return "Select at least two profitable same-industry peers with matching sources and reporting basis for a P/E comparison."
    median = float(values.median())
    gap = (pe / median - 1) * 100
    return f"P/E is {abs(gap):.1f}% {'above' if gap > 0 else 'below'} the same-industry peer median ({median:.1f}). This compares multiples, not intrinsic value."


def format_value(value, unit, currency="INR"):
    value = number(value)
    if value is None:
        return "N/A"
    if unit == "percent":
        return f"{value:,.2f}%"
    if unit == "multiple":
        return f"{value:,.2f}x"
    if unit == "days":
        return f"{value:,.1f} days"
    if unit == "number":
        return f"{value:,.0f}"
    if unit == "money":
        if currency == "INR" and abs(value) >= CRORE:
            return f"INR {value / CRORE:,.2f} Cr"
        return f"{currency} {value:,.2f}"
    return f"{value:,.2f}"


def assess_metric(key, metric, details=None, peer_values=None):
    value, rule = number(metric.get("value")), SPECS.get(key, {}).get("rule")
    result = {"assessment": "Context needed", "range": "Compare history and suitable peers", "tone": "neutral",
              "reason": "No universal good/bad threshold"}
    if value is None:
        return {**result, "assessment": "Unavailable", "range": "N/A", "reason": metric.get("note") or "Not disclosed by provider"}
    details = details or {}
    if key in ("pe", "forward_pe", "pb", "peg", "ev_ebitda", "price_cfo", "price_fcf") and value <= 0:
        return {**result, "assessment": "Not meaningful", "range": "Requires a positive denominator",
                "reason": "Non-positive earnings, book value, cash flow or growth prevents a cheap/expensive interpretation"}
    denominator_key = "eps" if key == "pe" else "book_value" if key == "pb" else "equity" if key == "debt_equity" else ""
    denominator = details.get("metrics", {}).get(denominator_key, {}).get("value")
    if denominator is not None and denominator <= 0:
        return {**result, "assessment": "Not meaningful", "range": "Requires positive equity / earnings",
                "reason": "A non-positive denominator prevents a normal ratio interpretation"}
    if details.get("financial_sector") and key in FINANCIAL_EXCLUSIONS:
        return {**result, "assessment": "Sector-specific", "range": "No generic cutoff for banks / insurers / lenders",
                "reason": "Use sector disclosures and peers; industrial-company thresholds are unsuitable"}
    peers = [x for x in (peer_values or []) if number(x) is not None and x > 0]
    if key in ("pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales", "price_cfo", "price_fcf") and len(peers) >= 2:
        median = float(pd.Series(peers).median())
        low, high = .8 * median, 1.2 * median
        return {"assessment": "Below peers" if value < low else "Above peers" if value > high else "Near peers",
                "range": f"Selected-peer band {low:.2f}–{high:.2f}x; median {median:.2f}x (n={len(peers)})",
                "tone": "caution" if value > high else "neutral",
                "reason": "80–120% of matching-source, same-industry peer median; lower multiples are not automatically better"}
    if not rule:
        return result
    thresholds, labels, guide = RANGES[rule]
    label = labels[sum(value >= limit for limit in thresholds)]
    concern = any(word in label.lower() for word in ("negative", "weak", "concern", "declining", "very high leverage", "above earnings", "very high multiple", "loss"))
    strong = any(word in label.lower() for word in ("strong", "adequate", "net cash", "low leverage"))
    return {"assessment": label, "range": guide,
            "tone": "concern" if concern else "favorable" if strong else "caution" if "high" in label.lower() or "thin" in label.lower() else "neutral",
            "reason": "Application screening default, not an industry standard or buy/sell recommendation. Growth, business model and accounting basis matter."}
