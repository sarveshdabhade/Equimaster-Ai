from datetime import datetime, timezone
from unittest.mock import Mock
import json
import pandas as pd
import pytest
from src.fundamental_sources import number, parse_screener_html, cagr
from src.fundamental_calculations import build_details
from src.fundamentals import assess_metric, format_value, peer_reference, snapshot_from_details


HTML = """<h1>Example Company</h1><ul id='top-ratios'>
<li><span class='name'>Market Cap</span><span class='value'>₹ 1,000 Cr.</span></li>
<li><span class='name'>Current Price</span><span class='value'>₹ 100</span></li>
<li><span class='name'>Stock P/E</span><span class='value'>16.8</span></li>
<li><span class='name'>Book Value</span><span class='value'>₹ 20</span></li></ul>
<section id='profit-loss'><table class='data-table'><thead><tr><th></th><th>Mar 2023</th><th>Mar 2026</th><th>TTM</th></tr></thead>
<tbody><tr><td>Sales +</td><td>100</td><td>133.1</td><td>999</td></tr>
<tr><td>Net Profit +</td><td>10</td><td>20</td><td>25</td></tr>
<tr><td>EPS in Rs</td><td>2</td><td>4</td><td>5</td></tr>
<tr><td>OPM %</td><td>20%</td><td>22%</td><td>-</td></tr></tbody></table></section>
<section id='shareholding'><table class='data-table'><thead><tr><th></th><th>Sep 2026</th></tr></thead>
<tbody><tr><td>Promoters +</td><td>71.77%</td></tr><tr><td>No. of Shareholders</td><td>12,345</td></tr></tbody></table></section>"""


def annual_frames():
    dates = pd.to_datetime(["2026-03-31", "2025-03-31"])
    balance = pd.DataFrame({dates[0]: {"TotalAssets": 1000, "CurrentLiabilities": 100, "CurrentAssets": 200,
                                      "StockholdersEquity": 500, "TotalDebt": 50, "CashAndCashEquivalents": 200, "Inventory": 20},
                            dates[1]: {"TotalAssets": 800, "CurrentLiabilities": 80}})
    income = pd.DataFrame({dates[0]: {"EBIT": 81, "EBITDA": 100, "NetIncome": 50,
                                     "TotalRevenue": 500, "InterestExpense": 9, "TaxProvision": 10, "PretaxIncome": 50}})
    cash = pd.DataFrame({dates[0]: {"OperatingCashFlow": 80, "CapitalExpenditure": -20, "CashDividendsPaid": -30}})
    return {"balance": balance, "income": income, "cashflow": cash}


def test_numeric_parser_preserves_missing_and_zero():
    assert number("₹ 1,234.5 Cr.") == 1234.5
    assert number("−2.5%") == -2.5
    assert number("0") == 0
    for value in ("-", "N/A", None, float("inf"), float("nan")):
        assert number(value) is None


def test_public_tables_preserve_crore_eps_percent_and_share_counts():
    parsed = parse_screener_html(HTML, "https://www.screener.in/company/TEST/consolidated/")
    annual = parsed["statements"]["profit-loss"]
    assert annual["periods"] == ["Mar 2023", "Mar 2026", "TTM"]
    assert annual["rows"][2]["unit"] == "money_per_share"
    assert annual["rows"][3]["values"][-1] is None
    assert parsed["statements"]["shareholding"]["rows"][1]["unit"] == "number"
    details = build_details("TEST", {}, parsed)
    assert details["metrics"]["market_cap"]["value"] == 10_000_000_000
    assert details["metrics"]["profit"]["value"] == 250_000_000
    assert details["metrics"]["eps"]["value"] == 5
    assert details["metrics"]["pb"]["value"] == 5
    assert details["metrics"]["promoters"]["value"] == 71.77


def test_yahoo_percentages_normalize_to_percent_or_multiple():
    details = build_details("TEST", {"returnOnEquity": .2, "debtToEquity": 9.83, "payoutRatio": .3,
                                      "dividendRate": 2, "currentPrice": 100})
    assert details["metrics"]["roe"]["value"] == 20
    assert details["metrics"]["debt_equity"]["value"] == pytest.approx(.0983)
    assert details["metrics"]["payout"]["value"] == 30
    assert details["metrics"]["dividend_yield"]["value"] == 2


def test_annual_ratios_use_matched_fiscal_dates_and_capex_sign():
    metrics = build_details("TEST", {}, annual=annual_frames())["metrics"]
    assert metrics["roce_annual"]["value"] == 10
    assert metrics["current_ratio"]["value"] == 2
    assert metrics["quick_ratio"]["value"] == 1.8
    assert metrics["net_debt_ebitda"]["value"] == -1.5
    assert metrics["interest_cover"]["value"] == 9
    assert metrics["fcf"]["value"] == 60
    assert metrics["cfo_profit"]["value"] == 1.6


def test_mismatched_fiscal_years_are_not_combined():
    annual = annual_frames()
    annual["income"].columns = pd.to_datetime(["2025-03-31"])
    metrics = build_details("TEST", {}, annual=annual)["metrics"]
    assert metrics["roce_annual"]["value"] is None
    assert metrics["cfo_profit"]["value"] is None


def test_cagr_excludes_ttm_and_requires_positive_matching_endpoints():
    table = parse_screener_html(HTML, "https://www.screener.in/company/TEST/")["statements"]["profit-loss"]
    assert cagr(table, ("Sales",), 3) == pytest.approx(10, abs=.02)
    assert cagr(table, ("Sales",), 5) is None
    table["rows"][0]["values"][0] = -100
    assert cagr(table, ("Sales",), 3) is None


def test_sector_specific_ratings_and_invalid_denominators():
    details = build_details("HDFCBANK.NS", {"sector": "Financial Services", "debtToEquity": 500})
    assert assess_metric("debt_equity", details["metrics"]["debt_equity"], details)["assessment"] == "Sector-specific"
    negative = build_details("TEST", {"trailingPE": 5, "trailingEps": -2, "priceToBook": 1, "bookValue": -10})
    assert assess_metric("pe", negative["metrics"]["pe"], negative)["assessment"] == "Not meaningful"
    assert assess_metric("pb", negative["metrics"]["pb"], negative)["assessment"] == "Not meaningful"


def test_threshold_boundaries_and_missing_values_are_explicit():
    assert assess_metric("pe", {"value": 15})["assessment"] == "Moderate multiple"
    assert assess_metric("pe", {"value": 25})["assessment"] == "High multiple"
    assert assess_metric("pe", {"value": 40})["assessment"] == "Very high multiple"
    assert assess_metric("pe", {"value": -5})["assessment"] == "Not meaningful"
    assert assess_metric("pe", {"value": None})["assessment"] == "Unavailable"
    assert assess_metric("current_ratio", {"value": .9})["assessment"] == "Liquidity concern"


def test_peer_bands_require_matching_industry_source_and_basis():
    def company(ticker, industry="IT", basis="Consolidated"):
        result = build_details(ticker, {"industry": industry}, parse_screener_html(HTML, f"https://www.screener.in/company/{ticker}/consolidated/"))
        result["metrics"]["pe"].update(value=20, basis=basis)
        return result
    subject = company("A")
    others = [company("B"), company("C"), company("D", "Bank"), company("E", basis="Standalone")]
    frame = pd.DataFrame([snapshot_from_details(x) for x in [subject] + others])
    values = peer_reference(frame, subject, "pe")
    assert values == [20, 20]
    result = assess_metric("pe", {"value": 30}, subject, values)
    assert result["assessment"] == "Above peers"
    assert "16.00–24.00x" in result["range"]


def test_currency_mismatch_does_not_produce_cash_valuation_multiple():
    metrics = build_details("TEST", {"currency": "INR", "financialCurrency": "USD", "marketCap": 1000}, annual=annual_frames())["metrics"]
    assert metrics["price_cfo"]["value"] is None
    assert metrics["fcf_yield"]["value"] is None


def test_verified_statement_currency_overrides_ambiguous_profile_currency():
    annual = annual_frames()
    annual["currencies"] = {name: {"2026-03-31": "INR"} for name in ("balance", "income", "cashflow")}
    metrics = build_details("TEST", {"currency": "INR", "financialCurrency": "USD", "marketCap": 1000}, annual=annual)["metrics"]
    assert metrics["cfo"]["currency"] == "INR"
    assert metrics["cfo"]["value"] == 80
    assert metrics["price_cfo"]["value"] == 12.5


def test_usd_statements_are_not_labeled_as_inr_and_bad_provider_multiples_are_withheld():
    annual = annual_frames()
    annual["currencies"] = {name: {"2026-03-31": "USD"} for name in ("balance", "income", "cashflow")}
    metrics = build_details("TEST", {"currency": "INR", "financialCurrency": "USD", "marketCap": 1000,
                                      "enterpriseToEbitda": 1000, "priceToSalesTrailing12Months": 200}, annual=annual)["metrics"]
    assert metrics["cfo"]["currency"] == "USD"
    assert metrics["equity"]["currency"] == "USD"
    assert metrics["ev_ebitda"]["value"] is None
    assert metrics["ps"]["value"] is None


def test_statement_currency_metadata_parser(monkeypatch):
    from src.fundamental_sources import statement_currencies
    client = Mock()
    client._data.cache_get.return_value.json.return_value = {"timeseries": {"result": [
        {"annualTotalRevenue": [{"asOfDate": "2026-03-31", "currencyCode": "USD"}]},
        {"annualTotalAssets": [{"asOfDate": "2026-03-31", "currencyCode": "INR"}]},
    ]}}
    result = statement_currencies(client, "TEST.NS")
    assert result["income"]["2026-03-31"] == "USD"
    assert result["balance"]["2026-03-31"] == "INR"


def test_failed_refresh_retains_last_saved_snapshot(monkeypatch, tmp_path):
    import src.fundamentals as module
    cached = build_details("TEST", {"longName": "Saved company"})
    path = tmp_path / "TEST.json"
    path.write_text(json.dumps(cached))
    monkeypatch.setattr(module, "resolve_app_path", lambda *args: str(path))
    client = Mock()
    for name in ("get_info", "get_balance_sheet", "get_income_stmt", "get_cash_flow"):
        getattr(client, name).side_effect = RuntimeError("Provider down")
    monkeypatch.setattr(module.yf, "Ticker", lambda *args: client)
    monkeypatch.setattr(module, "fetch_screener", lambda *args: (_ for _ in ()).throw(RuntimeError("Provider down")))
    before = path.read_bytes()
    result = module.company_details("TEST", refresh=True)
    assert result["Company"] == "Saved company"
    assert any("Refresh failed" in warning for warning in result["warnings"])
    assert path.read_bytes() == before


def test_user_display_units_are_unambiguous():
    assert format_value(15, "percent") == "15.00%"
    assert format_value(.1, "multiple") == "0.10x"
    assert format_value(10_000_000, "money") == "INR 1.00 Cr"
    assert format_value(None, "multiple") == "N/A"


def test_fundamentals_page_displays_cached_ranges_and_statement_exports(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import tabs.fundamentals as tab
    details = build_details("TCS.NS", {"longName": "Test Company", "industry": "IT"},
                            parse_screener_html(HTML, "https://www.screener.in/company/TCS/consolidated/"))
    monkeypatch.setattr(tab, "read_cached_details", lambda ticker: details if ticker == "TCS.NS" else None)
    wrapper = tmp_path / "fundamentals_preview.py"
    wrapper.write_text("from tabs.fundamentals import render_fundamentals_tab\nrender_fundamentals_tab('TCS.NS')")
    app = AppTest.from_file(str(wrapper), default_timeout=30).run()
    assert not app.exception
    assert any(metric.label == "P/E (trailing)" and metric.value == "16.80x" for metric in app.metric)
    assert any(button.label == "Export All Fundamentals" for button in app.get("download_button"))
    assert any(button.label == "Export Quarterly results" for button in app.get("download_button")) is False
    assert any(button.label == "Export Annual profit and loss" for button in app.get("download_button"))
