"""Read public financial facts without executing page code or accessing private data."""
import math
import re
import time
from urllib.parse import quote
import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import provider_symbol


def number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("−", "-").replace("\xa0", " ").strip()
        if value.lower() in ("", "-", "--", "n/a", "na", "none", "nan"):
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not match:
            return None
        value = match.group()
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def ratio(numerator, denominator):
    a, b = number(numerator), number(denominator)
    return a / b if a is not None and b is not None and b > 0 else None


def parse_table(table, unit="INR crore", basis="Consolidated"):
    headers = table.select("thead tr")
    periods = [cell.get_text(" ", strip=True) for cell in headers[-1].find_all(["th", "td"])[1:]] if headers else []
    rows = []
    for row in table.select("tbody tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 2:
            continue
        label = re.sub(r"\s+", " ", cells[0].get_text(" ", strip=True)).rstrip(" +")
        values = [number(cell.get_text(" ", strip=True)) for cell in cells[1:]]
        if len(values) != len(periods):
            continue
        row_unit = ("number" if "shareholders" in label.lower() else
                    "percent" if "%" in label or unit == "percent" else
                    "money_per_share" if "EPS" in label else "days" if unit == "days / percent" else unit)
        rows.append({"label": label, "values": values, "unit": row_unit})
    return {"periods": periods, "rows": rows, "unit": unit, "basis": basis, "source": "Screener.in"}


def parse_screener_html(html, url):
    soup = BeautifulSoup(html, "html.parser")
    top = {}
    for item in soup.select("#top-ratios li"):
        name, value = item.select_one(".name"), item.select_one(".value")
        if name and value:
            top[name.get_text(" ", strip=True)] = value.get_text(" ", strip=True)
    basis = "Consolidated" if "/consolidated/" in url else "Standalone"
    statements = {}
    for section in ("quarters", "profit-loss", "balance-sheet", "cash-flow", "ratios", "shareholding"):
        element = soup.select_one(f"#{section}")
        table = element.select_one("table.data-table") if element else None
        if table:
            statements[section] = parse_table(table, "percent" if section == "shareholding" else "days / percent" if section == "ratios" else "INR crore", basis)
    title = soup.select_one("h1")
    return {"url": url, "basis": basis, "top": top, "statements": statements,
            "company": title.get_text(" ", strip=True) if title else None}


def fetch_screener(ticker):
    symbol = provider_symbol(ticker).removesuffix(".NS").removesuffix(".BO")
    for suffix in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{quote(symbol, safe='')}/{suffix}"
        response = requests.get(url, timeout=20, headers={"User-Agent": "Equimaster-Ai/1.0 (personal research)"})
        if response.status_code == 404:
            continue
        response.raise_for_status()
        result = parse_screener_html(response.text, url)
        if result["top"] or result["statements"]:
            return result
        raise ValueError("Public company financial tables were not returned")
    raise ValueError("No public company page found")


def statement_currencies(client, ticker):
    """Read currencyCode discarded by yfinance's DataFrame conversion (pinned 1.1.0)."""
    keys = {"annualTotalAssets": "balance", "annualTotalRevenue": "income", "annualOperatingCashFlow": "cashflow"}
    symbol = quote(provider_symbol(ticker), safe="")
    url = (f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"
           f"?symbol={symbol}&type={','.join(keys)}&period1=1483142400&period2={int(time.time())}")
    response = client._data.cache_get(url=url, timeout=20)
    response.raise_for_status()
    result = {key: {} for key in keys.values()}
    for series in response.json().get("timeseries", {}).get("result", []):
        for field, category in keys.items():
            for point in series.get(field, []):
                if point.get("asOfDate") and point.get("currencyCode"):
                    result[category][point["asOfDate"]] = point["currencyCode"]
    return result


def cell(frame, keys, date=None):
    if frame is None or frame.empty:
        return None
    date = frame.columns[0] if date is None else date
    if date not in frame.columns:
        return None
    for key in keys if isinstance(keys, tuple) else (keys,):
        if key in frame.index:
            value = number(frame.loc[key, date])
            if value is not None:
                return value
    return None


def find_row(table, names):
    for row in table.get("rows", []):
        if row["label"].lower().strip(" +%") in {name.lower() for name in names}:
            return row
    return None


def latest(table, names):
    row = find_row(table, names)
    if row:
        for period, value in reversed(list(zip(table["periods"], row["values"]))):
            if value is not None and period != "TTM":
                return value, period
    return None, None


def cagr(table, names, years):
    row = find_row(table, names)
    if not row:
        return None
    observations = {}
    for period, value in zip(table["periods"], row["values"]):
        try:
            if period == "TTM":
                continue
            stamp = pd.to_datetime(period)
            if value is not None:
                observations[stamp.year] = (stamp, value)
        except (ValueError, TypeError):
            continue
    if not observations:
        return None
    year = max(observations)
    if year - years not in observations:
        return None
    end, final = observations[year]
    start, initial = observations[year - years]
    duration = (end - start).days / 365.25
    if initial <= 0 or final <= 0 or not years - .1 <= duration <= years + .1:
        return None
    return (final / initial) ** (1 / duration) * 100 - 100
