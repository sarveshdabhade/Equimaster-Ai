# Company fundamentals

Open Stock Analyzer > Fundamentals. All 58 configured companies have public-source
snapshots loaded; use Refresh Company Data to recheck the selected company and peers.
The page supports 98 parameters. A supported slot does not imply the provider supplies
that number for every company: missing, inappropriate or unverified values are identified.

The top cards show market cap, P/E, P/B, dividend yield, ROE, published ROCE, debt/equity
and trailing EPS. The remaining tables cover forward valuation, EV multiples, margins,
ROA, annual ROCE/ROIC, growth/CAGR, liquidity, debt, assets, liabilities, cash generation,
working capital, per-share values, efficiency and ownership.

Financial Statements contains quarterly results, annual profit/loss, balance sheets,
cash flows and historical ratios. Shareholding includes promoters, FII, DII and public
holdings when disclosed, a historical table and a trend chart. Yahoo insider/institutional
holdings are identified separately; they are not relabeled promoters/FII/DII.

## Reading the assessments

Every rated metric displays its exact reference range. The default examples below are
**application screening heuristics, not universally accepted industry standards**:

| Parameter | Default interpretation |
|---|---|
| Positive trailing P/E | <15x low; 15?25x moderate; 25?40x high; >=40x very high |
| Positive P/B | <1x low; 1?3x moderate; 3?5x high; >=5x very high |
| ROE / ROCE | <0% negative; 0?10% weak; 10?15% moderate; 15?25% strong; >=25% very high |
| Debt/equity, non-financial companies | <0.5x low; 0.5?1x moderate; 1?2x high; >=2x very high |
| Current ratio, non-financial companies | <1x concern; 1?1.5x tight; 1.5?3x adequate; >=3x high |

When at least two selected peers have the same industry, known accounting basis,
metric source and reported-period label, valuation ranges use 80?120% of their median.
Stale peer snapshots are excluded. A low multiple alone is not a good-company or buy signal.
Negative earnings/book value and non-positive denominator ratios are not labeled cheap.
Financial-company leverage, liquidity, ROA/ROCE, margins and cash flows require different
analysis, so generic industrial thresholds are disabled. NPA, NIM, CASA, capital adequacy
and pledged shares remain N/A if the current sources do not supply them.

## Units and source conventions

Monetary totals from Screener are INR crore (1 crore = 10,000,000). EPS/book value/face
value are per-share amounts. Percentages are displayed as percentages; debt/equity is a
multiple. Yahoo debtToEquity is divided by 100 to normalize its percentage representation.

Quote currency and financial currency can differ. Annual statement currency metadata
is checked for affected symbols: Infosys reports USD annual statements; HCL has INR
annual statements even though its Yahoo profile has USD financial fields. Amounts show
their own currency; unverified mixed-currency valuation multiples are withheld. There is
no invented FX conversion. Published/derived ratios can differ due to definitions and dates.

Annual derivations use matching fiscal end dates. Calculated ROCE uses EBIT divided by
average annual capital employed (assets minus current liabilities). ROIC uses effective-tax
NOPAT divided by year-end debt + equity - cash, explicitly identifying that definition.
CAGR excludes TTM and requires positive endpoints with the actual fiscal-year span.
Retrieval time is not the financial disclosure date; every parameter includes its period.

CSV exports include formatted values, raw numeric values, units, ranges, source, basis,
period and explanations. Full JSON and individual statement exports are also available.

## Refreshing

```powershell
python scripts/refresh_fundamentals.py
python scripts/refresh_fundamentals.py --only-stale
python scripts/refresh_fundamentals.py --tickers TCS.NS,INFY.NS,HCLTECH.NS
python scripts/valuation_coverage.py
```

The existing scheduled daily workflow includes the stale-snapshot refresh. Public page
layout changes or provider failures can reduce coverage; warnings and N/A values are shown.
An entirely failed refresh preserves the previous saved snapshot and its timestamp.

## Valuation coverage

The company page and all-stock exporter share `calculate_fundamental_valuation`.
The exporter records a method, point estimate, scenario range, input provenance,
assumptions and confidence tier for each configured company. Its default parameters
are fixed research assumptions, including company-reported beta; they are not live
estimates of market discount rates. Company-page overrides can therefore produce
different scenarios from a report generated with the defaults.

Industrial FCFE DCF requires positive same-period operating cash flow less capital
expenditure plus reported net borrowing, a positive share count and verified currency.
Financial companies can use residual income from book value and ROE. Compatible
same-industry peers are preferred; a sector fallback is labelled a low-confidence
scenario. An unavailable method retains its precise reason instead of invented inputs.

The saved report is `data/evaluation/valuation_coverage.json`, with a matching CSV.
Stale snapshots fail the report's freshness check. A confidence tier describes input
and method coverage; scenario ranges are not statistical confidence intervals.
The separately gated 252-session return model still requires matured point-in-time
labels before it can support validated BUY/SELL recommendations.

Sources: [Screener company financials](https://www.screener.in/company/TCS/consolidated/),
[CFA Institute ratio interpretation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/financial-analysis-techniques),
[Fidelity industry context](https://www.fidelity.com/learning-center/trading-investing/fundamentals),
[ACCA ROCE methodology](https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f7/technical-articles/performance-appraisal.html).
The numerical screening defaults above are chosen by this application, not endorsed
cutoffs attributed to these sources.
