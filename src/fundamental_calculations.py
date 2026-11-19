"""Normalize published metrics and derive ratios only from compatible inputs."""
from datetime import datetime, timezone
from config import BANK_NIFTY_TICKERS, provider_symbol
from urllib.parse import quote
from src.fundamental_catalog import SPECS
from src.fundamental_sources import number, ratio, cell, find_row, latest, cagr

CRORE = 10_000_000


def build_details(ticker, info, screener=None, annual=None, warnings=None):
    annual = annual or {}
    screener = screener or {"top": {}, "statements": {}, "basis": "Unavailable", "url": None}
    metrics = {key: {**spec, "value": None, "source": None, "period": None, "basis": None,
                     "note": "Not returned by the available public sources"} for key, spec in SPECS.items()}
    quote_currency = info.get("currency") or "INR"
    financial_currency = info.get("financialCurrency") or quote_currency
    quote_money = {"price", "market_cap", "enterprise_value", "eps", "forward_eps", "book_value", "high_52w", "low_52w", "dividend_ps"}

    def put(key, value, source="Yahoo Finance", period="TTM / latest reported", basis="Provider reported", note=None):
        value = number(value)
        if value is not None:
            metrics[key].update(value=value, source=source, period=period, basis=basis, note=note)
            if metrics[key]["unit"] == "money":
                metrics[key]["currency"] = "INR" if source and source.startswith("Screener") else quote_currency if key in quote_money else financial_currency
        elif note and metrics[key]["value"] is None:
            metrics[key].update(source=source, period=period, basis=basis, note=note)

    mappings = {
        "price": "currentPrice", "market_cap": "marketCap", "enterprise_value": "enterpriseValue",
        "pe": "trailingPE", "forward_pe": "forwardPE", "pb": "priceToBook", "ps": "priceToSalesTrailing12Months",
        "ev_ebitda": "enterpriseToEbitda", "ev_sales": "enterpriseToRevenue", "peg": "trailingPegRatio",
        "current_ratio": "currentRatio", "quick_ratio": "quickRatio", "eps": "trailingEps", "forward_eps": "forwardEps",
        "book_value": "bookValue", "cash_per_share": "totalCashPerShare", "shares": "sharesOutstanding",
        "float_shares": "floatShares", "beta": "beta", "high_52w": "fiftyTwoWeekHigh", "low_52w": "fiftyTwoWeekLow",
        "revenue": "totalRevenue", "profit": "netIncomeToCommon", "ebitda": "ebitda", "debt": "totalDebt",
        "cash": "totalCash", "dividend_ps": "dividendRate", "dividend_avg_5y": "fiveYearAvgDividendYield",
    }
    for key, field in mappings.items():
        put(key, info.get(field))
    for key, field in {
        "roe": "returnOnEquity", "roa": "returnOnAssets", "gross_margin": "grossMargins",
        "operating_margin": "operatingMargins", "ebitda_margin": "ebitdaMargins", "net_margin": "profitMargins",
        "revenue_growth": "revenueGrowth", "earnings_growth": "earningsGrowth", "quarterly_earnings_growth": "earningsQuarterlyGrowth",
        "payout": "payoutRatio", "insiders": "heldPercentInsiders", "institutions": "heldPercentInstitutions",
    }.items():
        value = number(info.get(field))
        put(key, value * 100 if value is not None else None)
    value = number(info.get("debtToEquity"))
    put("debt_equity", value / 100 if value is not None else None,
        note="Yahoo debtToEquity is a percentage; displayed as a multiple")
    dy = ratio(info.get("dividendRate"), info.get("currentPrice"))
    put("dividend_yield", dy * 100 if dy is not None else None,
        note="Annual indicated dividend per share / current price")
    balance, income, cashflow = (annual.get(name) for name in ("balance", "income", "cashflow"))
    anchor = next((frame for frame in (balance, income, cashflow) if frame is not None and not frame.empty), None)
    date = anchor.columns[0] if anchor is not None else None
    period = str(date.date()) if date is not None else None
    currency_map = annual.get("currencies", {})
    def annual_currency(category):
        return currency_map.get(category, {}).get(period) or (financial_currency if financial_currency == quote_currency else None)
    anchor_category = "balance" if balance is not None and not balance.empty else "income" if income is not None and not income.empty else "cashflow"
    matched_currency = annual_currency(anchor_category)
    b = lambda keys: cell(balance, keys, date)
    i = lambda keys: cell(income, keys, date) if annual_currency("income") == matched_currency and matched_currency else None
    c = lambda keys: cell(cashflow, keys, date) if annual_currency("cashflow") == matched_currency and matched_currency else None
    base = {"assets": b("TotalAssets"), "liabilities": b("TotalLiabilitiesNetMinorityInterest"),
            "equity": b(("StockholdersEquity", "CommonStockEquity")), "receivables": b("AccountsReceivable"), "inventory": b("Inventory")}
    for key, value in base.items():
        put(key, value, period=period, basis="Annual balance sheet")
        if metrics[key]["unit"] == "money":
            metrics[key]["currency"] = annual_currency("balance") or "Unverified currency"
    ca, cl, inventory = b("CurrentAssets"), b("CurrentLiabilities"), b("Inventory")
    cash = b(("CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents"))
    debt, equity, assets = b("TotalDebt"), base["equity"], base["assets"]
    ebit, ebitda = i("EBIT"), i(("EBITDA", "NormalizedEBITDA"))

    def derived(key, value, formula):
        put(key, value, "Yahoo Finance (calculated)", period, "Annual; matched fiscal dates", formula)
        if metrics[key]["unit"] == "money":
            metrics[key]["currency"] = matched_currency or "Unverified currency"

    derived("working_capital", ca - cl if ca is not None and cl is not None else None, "Current assets - current liabilities")
    derived("cash_ratio", ratio(cash, cl), "Cash and short-term investments / current liabilities")
    if metrics["current_ratio"]["value"] is None:
        derived("current_ratio", ratio(ca, cl), "Current assets / current liabilities")
    if metrics["quick_ratio"]["value"] is None:
        derived("quick_ratio", ratio(ca - inventory if ca is not None and inventory is not None else None, cl), "(Current assets - inventory) / current liabilities")
    derived("debt_assets", ratio(debt, assets) * 100 if ratio(debt, assets) is not None else None, "Annual debt / assets")
    derived("net_debt", debt - cash if debt is not None and cash is not None else None, "Annual debt - cash and short-term investments")
    derived("net_debt_ebitda", ratio(metrics["net_debt"]["value"], ebitda), "Annual net debt / same-year EBITDA")
    interest = i(("InterestExpense", "InterestExpenseNonOperating"))
    derived("interest_cover", ratio(ebit, abs(interest)) if interest is not None else None,
            "Annual EBIT / absolute interest expense; zero interest yields no finite multiple")
    if metrics["debt_equity"]["value"] is None:
        derived("debt_equity", ratio(debt, equity), "Annual debt / shareholders' equity")
    previous = balance.columns[1] if balance is not None and len(balance.columns) > 1 else None
    capital = assets - cl if assets is not None and cl is not None else None
    prev_assets = cell(balance, "TotalAssets", previous) if previous is not None else None
    prev_cl = cell(balance, "CurrentLiabilities", previous) if previous is not None else None
    avg_capital = (capital + prev_assets - prev_cl) / 2 if capital is not None and prev_assets is not None and prev_cl is not None else None
    derived("roce_annual", ratio(ebit, avg_capital) * 100 if ratio(ebit, avg_capital) is not None else None,
            "EBIT / average of current and previous annual (total assets - current liabilities)")
    tax_rate = ratio(i("TaxProvision"), i("PretaxIncome"))
    invested = debt + equity - cash if all(v is not None for v in (debt, equity, cash)) else None
    nopat = ebit * (1 - tax_rate) if ebit is not None and tax_rate is not None and 0 <= tax_rate <= 1 else None
    derived("roic", ratio(nopat, invested) * 100 if ratio(nopat, invested) is not None else None,
            "EBIT x (1 - effective tax rate) / (year-end debt + equity - cash); year-end capital basis")
    cfo, fcf, capex = c("OperatingCashFlow"), c("FreeCashFlow"), c("CapitalExpenditure")
    if fcf is None and cfo is not None and capex is not None:
        fcf = cfo - abs(capex)
    # FCFE requires net borrowing in addition to CFO less capital investment.
    # Withhold it when the same-period aggregate is not reported; total financing
    # cash flow also includes dividends and equity transactions, so it is not a proxy.
    net_borrowing = c("NetIssuancePaymentsOfDebt")
    fcfe = cfo - abs(capex) + net_borrowing if cfo is not None and capex is not None and net_borrowing is not None else None
    derived("net_borrowing", net_borrowing,
            "Same-period reported net debt issuance less repayments; requires verified annual statement currency")
    for key, value in (("cfo", cfo), ("fcf", fcf), ("capex", abs(capex) if capex is not None else None),
                       ("investing_cf", c("InvestingCashFlow")), ("financing_cf", c("FinancingCashFlow"))):
        derived(key, value, "Annual cash flow statement; FCF = operating cash flow - capital expenditure outflow")
    derived("fcfe", fcfe,
            "Annual CFO - capital expenditure outflow + same-period reported net debt borrowing; withheld if any input is unavailable")
    derived("cfo_profit", ratio(cfo, i("NetIncome")), "Same-year operating cash flow / net profit; requires positive net profit")
    dividends = c("CashDividendsPaid")
    derived("dividend_cover", ratio(fcf, abs(dividends)) if dividends is not None else None, "Annual free cash flow / absolute cash dividends paid")
    derived("fcf_margin", ratio(fcf, i("TotalRevenue")) * 100 if ratio(fcf, i("TotalRevenue")) is not None else None, "Annual FCF / same-year revenue")
    avg_assets = (assets + prev_assets) / 2 if assets is not None and prev_assets is not None else None
    derived("asset_turnover", ratio(i("TotalRevenue"), avg_assets), "Annual revenue / average total assets")
    derived("fcf_per_share", ratio(fcf, info.get("sharesOutstanding")), "Annual FCF / latest shares; dates differ")
    top, statements = screener["top"], screener["statements"]
    for label, key in {"Market Cap": "market_cap", "Current Price": "price", "Stock P/E": "pe", "Book Value": "book_value",
                       "Dividend Yield": "dividend_yield", "ROCE": "roce", "ROE": "roe", "Face Value": "face_value"}.items():
        value = number(top.get(label))
        if key == "market_cap" and value is not None:
            value *= CRORE
        put(key, value, "Screener.in", "Latest published", screener["basis"], "Published value; reporting convention may differ from Yahoo")
    if "High / Low" in top:
        values = top["High / Low"].split("/")
        if len(values) == 2:
            for key, value in zip(("high_52w", "low_52w"), values):
                put(key, number(value), "Screener.in", "Latest published", screener["basis"])
    if number(top.get("Current Price")) is not None and number(top.get("Book Value")) is not None:
        put("pb", ratio(number(top["Current Price"]), number(top["Book Value"])), "Screener.in (calculated)", "Latest published", screener["basis"], "Published price / book value per share")
    annual_table = statements.get("profit-loss", {})
    for key, names in (("revenue", ("Sales", "Revenue", "Revenue from operations")), ("profit", ("Net Profit",)), ("eps", ("EPS in Rs", "EPS"))):
        row = find_row(annual_table, names)
        if row and "TTM" in annual_table["periods"]:
            value = row["values"][annual_table["periods"].index("TTM")]
            put(key, value * CRORE if value is not None and key != "eps" else value, "Screener.in", "TTM", screener["basis"])
    if financial_currency != quote_currency:
        for key in ("ps", "ev_ebitda", "ev_sales"):
            metrics[key].update(value=None, note="Provider multiple mixes financial and quote currencies; withheld")
        for key in ("pe", "pb", "forward_pe", "enterprise_value"):
            if metrics[key]["source"] == "Yahoo Finance":
                metrics[key].update(value=None, note="Quote and profile financial currencies differ; provider valuation is not verified")
        for key in ("forward_eps", "eps", "book_value"):
            if metrics[key]["source"] == "Yahoo Finance":
                metrics[key]["currency"] = "Unverified currency"
    if metrics["revenue"]["source"] == "Screener.in" and metrics["market_cap"]["source"] == "Screener.in":
        put("ps", ratio(metrics["market_cap"]["value"], metrics["revenue"]["value"]), "Screener.in (calculated)",
            "Latest market cap / TTM sales", screener["basis"], "Same-source INR market capitalisation / TTM sales")
    for kind, names in (("sales", ("Sales", "Revenue", "Revenue from operations")), ("profit", ("Net Profit",)), ("eps", ("EPS in Rs", "EPS"))):
        for years in (3, 5, 10):
            put(f"{kind}_cagr_{years}", cagr(annual_table, names, years), "Screener.in (calculated)",
                f"{years} fiscal years to latest annual report", screener["basis"], "Positive endpoints required; TTM excluded; actual elapsed years used")
    for names, key in ((("Debtor Days",), "debtor_days"), (("Inventory Days",), "inventory_days"),
                       (("Days Payable",), "payable_days"), (("Cash Conversion Cycle",), "cash_cycle"), (("Working Capital Days",), "working_capital_days")):
        value, fiscal = latest(statements.get("ratios", {}), names)
        put(key, value, "Screener.in", fiscal, screener["basis"])
    for names, key in ((("Promoters",), "promoters"), (("FIIs",), "fii"), (("DIIs",), "dii"), (("Public",), "public"), (("Government",), "government")):
        value, fiscal = latest(statements.get("shareholding", {}), names)
        put(key, value, "Screener.in", fiscal, screener["basis"])
    pe = metrics["pe"]["value"]
    put("earnings_yield", 100 / pe if pe is not None and pe > 0 else None,
        metrics["pe"]["source"], metrics["pe"]["period"], metrics["pe"]["basis"], "100 / positive trailing P/E")
    # Cross-date multiples explicitly use Yahoo's market cap and financial currency.
    currencies_match = annual_currency("cashflow") == quote_currency
    if currencies_match:
        for key, flow in (("price_cfo", cfo), ("price_fcf", fcf)):
            put(key, ratio(info.get("marketCap"), flow), "Yahoo Finance (calculated)",
                f"Latest market cap / annual cash flow {period}", "Mixed dates; explicitly identified")
        put("fcf_yield", ratio(fcf, info.get("marketCap")) * 100 if ratio(fcf, info.get("marketCap")) is not None else None,
            "Yahoo Finance (calculated)", f"Annual {period} / latest market cap", "Mixed dates; explicitly identified")
    sector = info.get("sector")
    financial = str(sector or "").lower() in ("financial services", "financials") or ticker in BANK_NIFTY_TICKERS or ticker in (
        "LICI.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS")
    summary = info.get("longBusinessSummary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
    return {"schema_version": 4, "Ticker": ticker, "Company": info.get("longName") or screener.get("company") or ticker,
            "Industry": info.get("industry"), "Sector": sector, "Currency": info.get("currency") or "INR",
            "Financial currency": info.get("financialCurrency"), "Website": info.get("website"), "Summary": summary,
            "As of": datetime.now(timezone.utc).isoformat(), "financial_sector": financial,
            "sources": {"screener": screener.get("url"), "yahoo": f"https://finance.yahoo.com/quote/{quote(provider_symbol(ticker))}/"},
            "metrics": metrics, "statements": statements, "statement_currencies": currency_map, "warnings": warnings or []}
