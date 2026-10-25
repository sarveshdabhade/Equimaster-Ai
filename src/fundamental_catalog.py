"""Metric display units and transparent application screening defaults."""

# Percent values use 15 for 15%; multiples use 1 for 1x; money uses base currency.
METRICS = []


def group(name, unit, rows):
    for row in rows:
        key, label, *rule = row
        METRICS.append((key, label, name, unit, rule[0] if rule else None))


group("Market & per share", "money", [
    ("price", "Current price"), ("market_cap", "Market capitalisation"),
    ("eps", "EPS (trailing)", "positive"), ("forward_eps", "Forward EPS"),
    ("book_value", "Book value / share"), ("face_value", "Face value"),
    ("cash_per_share", "Cash / share"), ("fcf_per_share", "Free cash flow / share (annual)"),
    ("high_52w", "52-week high"), ("low_52w", "52-week low"),
])
group("Market & per share", "number", [("shares", "Shares outstanding"), ("float_shares", "Float shares")])
group("Market & per share", "multiple", [("beta", "Beta")])
group("Valuation", "money", [("enterprise_value", "Enterprise value")])
group("Valuation", "multiple", [
    ("pe", "P/E (trailing)", "pe"), ("forward_pe", "Forward P/E", "pe"),
    ("pb", "P/B", "pb"), ("ps", "Price / sales", "ps"),
    ("ev_ebitda", "EV / EBITDA", "ev_ebitda"), ("ev_sales", "EV / sales", "ps"),
    ("price_cfo", "Price / operating cash flow", "cash_multiple"),
    ("price_fcf", "Price / free cash flow", "cash_multiple"), ("peg", "PEG (provider estimate)", "peg"),
])
group("Valuation", "percent", [("earnings_yield", "Earnings yield"), ("fcf_yield", "Free cash flow yield", "yield")])
group("Profitability", "percent", [
    ("roe", "Return on equity (ROE)", "return"), ("roa", "Return on assets (ROA)", "roa"),
    ("roce", "ROCE (published)", "return"), ("roce_annual", "ROCE (annual, average capital)", "return"),
    ("roic", "ROIC (derived, annual)", "return"), ("gross_margin", "Gross profit margin", "margin"),
    ("operating_margin", "Operating margin", "margin"), ("ebitda_margin", "EBITDA margin", "margin"),
    ("net_margin", "Net profit margin", "margin"), ("fcf_margin", "Free cash flow margin", "margin"),
])
group("Growth", "percent", [
    ("revenue_growth", "Revenue growth YoY", "growth"), ("earnings_growth", "Earnings growth YoY", "growth"),
    ("quarterly_earnings_growth", "Quarterly earnings growth YoY", "growth"),
    *[(f"{kind}_cagr_{years}", f"{label} CAGR ({years} years)", "growth")
      for kind, label in (("sales", "Sales"), ("profit", "Net profit"), ("eps", "EPS")) for years in (3, 5, 10)],
])
group("Debt & liquidity", "multiple", [
    ("debt_equity", "Debt / equity", "debt"), ("net_debt_ebitda", "Net debt / EBITDA (annual)", "net_debt"),
    ("interest_cover", "Interest coverage (annual EBIT)", "interest"),
    ("current_ratio", "Current ratio", "current"), ("quick_ratio", "Quick ratio", "quick"),
    ("cash_ratio", "Cash ratio", "cash"),
])
group("Debt & liquidity", "percent", [("debt_assets", "Debt / assets", "debt_assets")])
group("Balance sheet", "money", [
    ("assets", "Total assets"), ("liabilities", "Total liabilities"), ("equity", "Shareholders' equity"),
    ("debt", "Total debt"), ("cash", "Cash & short-term investments"), ("net_debt", "Net debt"),
    ("working_capital", "Working capital"), ("receivables", "Trade receivables"), ("inventory", "Inventory"),
])
group("Cash flow", "money", [
    ("revenue", "Revenue (TTM)"), ("profit", "Net profit (TTM)"), ("ebitda", "EBITDA (TTM)"),
    ("cfo", "Operating cash flow (annual)", "positive"), ("fcf", "Free cash flow (annual)", "positive"),
    ("capex", "Capital expenditure (annual outflow)"), ("investing_cf", "Investing cash flow (annual)"),
    ("financing_cf", "Financing cash flow (annual)"),
])
group("Cash flow", "multiple", [
    ("cfo_profit", "Operating cash flow / net profit (annual)", "conversion"),
    ("dividend_cover", "Free cash flow / dividends paid (annual)", "conversion"),
])
group("Efficiency", "days", [
    ("debtor_days", "Debtor days"), ("inventory_days", "Inventory days"),
    ("payable_days", "Days payable"), ("cash_cycle", "Cash conversion cycle"),
    ("working_capital_days", "Working capital days"),
])
group("Efficiency", "multiple", [("asset_turnover", "Asset turnover (annual)")])
group("Dividends", "percent", [
    ("dividend_yield", "Dividend yield", "dividend_yield"), ("payout", "Dividend payout ratio", "payout"),
    ("dividend_avg_5y", "5-year average dividend yield"),
])
group("Dividends", "money", [("dividend_ps", "Annual dividend / share")])
group("Ownership", "percent", [
    ("promoters", "Promoter holding"), ("fii", "FII holding"), ("dii", "DII holding"),
    ("public", "Public holding"), ("government", "Government holding"),
    ("insiders", "Insider holding (Yahoo definition)"),
    ("institutions", "Institutional holding (Yahoo definition)"), ("pledged", "Promoter shares pledged"),
])
group("Financial-sector disclosures", "percent", [
    ("nim", "Net interest margin"), ("gnpa", "Gross NPA ratio"), ("nnpa", "Net NPA ratio"),
    ("casa", "CASA ratio"), ("capital_adequacy", "Capital adequacy ratio"),
])
SPECS = {key: {"label": label, "group": category, "unit": unit, "rule": rule}
         for key, label, category, unit, rule in METRICS}

# These are application heuristics, not universal industry standards.
RANGES = {
    "pe": ([15, 25, 40], ["Low multiple", "Moderate multiple", "High multiple", "Very high multiple"], "<15x low; 15–25x moderate; 25–40x high; >=40x very high"),
    "pb": ([1, 3, 5], ["Low multiple", "Moderate multiple", "High multiple", "Very high multiple"], "<1x low; 1–3x moderate; 3–5x high; >=5x very high"),
    "ps": ([1, 3, 6], ["Low multiple", "Moderate multiple", "High multiple", "Very high multiple"], "<1x low; 1–3x moderate; 3–6x high; >=6x very high"),
    "ev_ebitda": ([8, 15, 25], ["Low multiple", "Moderate multiple", "High multiple", "Very high multiple"], "<8x low; 8–15x moderate; 15–25x high; >=25x very high"),
    "cash_multiple": ([10, 20, 30], ["Low multiple", "Moderate multiple", "High multiple", "Very high multiple"], "<10x low; 10–20x moderate; 20–30x high; >=30x very high"),
    "peg": ([1, 2], ["Low vs growth", "Moderate vs growth", "High vs growth"], "Positive PEG: <1x low; 1–2x moderate; >=2x high"),
    "return": ([0, 10, 15, 25], ["Negative return", "Weak return", "Moderate return", "Strong return", "Very high return"], "<0% negative; 0–10% weak; 10–15% moderate; 15–25% strong; >=25% very high"),
    "roa": ([0, 2, 5], ["Negative return", "Low return", "Moderate return", "Strong return"], "<0% negative; 0–2% low; 2–5% moderate; >=5% strong"),
    "margin": ([0, 10, 20], ["Loss / negative margin", "Thin margin", "Moderate margin", "High margin"], "<0% negative; 0–10% thin; 10–20% moderate; >=20% high; compare with industry"),
    "growth": ([0, 10, 20], ["Declining", "Slow growth", "Moderate growth", "Strong growth"], "<0% declining; 0–10% slow; 10–20% moderate; >=20% strong"),
    "debt": ([.5, 1, 2], ["Low leverage", "Moderate leverage", "High leverage", "Very high leverage"], "<0.5x low; 0.5–1x moderate; 1–2x high; >=2x very high"),
    "debt_assets": ([20, 50, 70], ["Low leverage", "Moderate leverage", "High leverage", "Very high leverage"], "<20% low; 20–50% moderate; 50–70% high; >=70% very high"),
    "net_debt": ([0, 2, 4], ["Net cash", "Lower leverage", "Elevated leverage", "High leverage"], "<0x net cash; 0–2x lower; 2–4x elevated; >=4x high"),
    "interest": ([1, 3, 5], ["Weak coverage", "Thin coverage", "Adequate coverage", "Strong coverage"], "<1x weak; 1–3x thin; 3–5x adequate; >=5x strong"),
    "current": ([1, 1.5, 3], ["Liquidity concern", "Tight liquidity", "Adequate liquidity", "High liquidity"], "<1x concern; 1–1.5x tight; 1.5–3x adequate; >=3x high (check idle assets)"),
    "quick": ([1, 2], ["Tight liquidity", "Adequate liquidity", "High liquidity"], "<1x tight; 1–2x adequate; >=2x high"),
    "cash": ([.2, 1], ["Low cash buffer", "Moderate cash buffer", "High cash buffer"], "<0.2x low; 0.2–1x moderate; >=1x high"),
    "conversion": ([0, 1, 1.5], ["Negative cash coverage", "Weak cash coverage", "Adequate cash coverage", "Strong cash coverage"], "<0x negative; 0–1x weak; 1–1.5x adequate; >=1.5x strong"),
    "positive": ([0], ["Negative", "Positive / non-negative"], "Negative values need review; positive values alone do not establish quality"),
    "yield": ([0, 3, 6], ["Negative cash yield", "Low cash yield", "Moderate cash yield", "High cash yield"], "<0% negative; 0–3% low; 3–6% moderate; >=6% high; review sustainability"),
    "dividend_yield": ([0, 2, 5, 8], ["Invalid", "Low / no yield", "Moderate yield", "High yield", "Very high yield: review"], "0–2% low; 2–5% moderate; 5–8% high; >=8% review sustainability"),
    "payout": ([0, 30, 60, 100], ["Invalid / loss basis", "Low payout", "Moderate payout", "High payout", "Above earnings: review"], "0–30% low; 30–60% moderate; 60–100% high; >=100% review coverage"),
}
FINANCIAL_EXCLUSIONS = {
    "debt_equity", "debt_assets", "net_debt_ebitda", "interest_cover", "current_ratio", "quick_ratio", "cash_ratio",
    "roa", "roce", "roce_annual", "roic", "gross_margin", "operating_margin", "ebitda_margin", "net_margin", "fcf_margin",
    "cfo", "fcf", "cfo_profit", "dividend_cover", "ev_ebitda", "ev_sales", "ps", "price_cfo", "price_fcf", "fcf_yield",
}
