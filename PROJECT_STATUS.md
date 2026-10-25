# Completion report ? 24 October 2026

All ten requested development areas have been implemented. This report distinguishes implemented features from measured model quality.

## Completed work

1. Refreshed all 57 raw/processed datasets through 23 October 2026; trained four horizons per ticker: Day (1 session), Week (5), Month (22), and Quarter (88, approximately four months). Added a verified Yahoo provider alias for LTIM/LTM. Tiny feed-rounding noise preserves existing data; material corrections trigger retraining.
2. Evaluated all 228 models against persistence using purged chronological fitting, validation and test periods. Historical prediction origins, target dates and actual prices are exported.
3. Shared the forecast/news fusion formula between inference and evaluation.
4. Replaced placeholder performance with measured metrics, persistence comparisons and empirical validation intervals. Measured test coverage is displayed rather than assuming nominal coverage.
5. Implemented cached company fundamentals, selected-peer comparisons, same-industry positive-P/E medians and CSV export. Live TCS/Infosys/HCLTech snapshots were checked.
6. Connected Market Summary, recommendation and risk modules, with their technical/news heuristic basis identified.
7. Added deterministic provider-independent tests, Streamlit interaction coverage, a Windows/Python 3.13 CI workflow, pinned requirements and an installed dependency lockfile. Updated setup/run instructions.
8. Consolidated training entrypoints. Immutable bundles contain models, scaler, exact processed-data snapshot, feature order, hashes, version, training parameters, dates, evaluations and backtests. Active bundles publish atomically; missing/corrupt artifacts are not silently reused.
9. Added provider fallback/failure checks, public publisher RSS, historical news cutoff enforcement, archived observations, sentiment/retrieval evaluation tooling and a registered weekday refresh task.
10. Added technical-chart date selection, volume, ATR, volatility and period returns; the forecast interface provides direct multi-session forecasts, held-out performance metrics and export. Model training is handled by backend CLI/pipeline workflows, not a frontend Training Manager page.

## Verification

- Latest full local suite: **56 tests passed**, including fundamentals, currency regressions, dashboard screening and multipage interactions. The artifact-reuse guard also has focused coverage for missing model files.
- Runtime integrity: **57 datasets**, all four model/scaler/snapshot contracts checked, **zero failures**.
- Inference smoke checks: **10 passed** across TCS/NIFTY direct horizons and the legacy recursive backend path. The user-facing forecast interface now exposes direct forecasts only.
- Streamlit headless startup and health endpoint: **200 OK**; home and analyzer interactions tested.
- Core daily workflow: **all five original stages returned exit code 0**; see [daily status](logs/daily_status.json). The new sixth fundamentals stage was separately verified for all 56 companies, including reuse of fresh snapshots.
- `pip check`: no broken requirements. Python syntax and changed-file whitespace checks passed.
- GitHub Actions workflow is prepared locally; no remote CI run or repository push was performed.

## Forecast results

Unweighted means across 57 tickers per horizon. Stock-price MAE is compared within each ticker, not averaged across differently priced stocks.

| Period (sessions) | Mean directional accuracy | Mean MAPE | Models beating persistence MAE | Measured interval coverage |
|---|---:|---:|---:|---:|
| Day (1) | 50.34% | 1.26% | 25/57 | 76.33% |
| Week (5) | 49.86% | 2.83% | 30/57 | 75.71% |
| Month (22) | 48.41% | 6.57% | 22/57 | 67.83% |
| Quarter (88; ~4 months) | 46.53% | 11.09% | 21/57 | 37.55% |

These results do not establish a trading edge. The interval target is 90%; actual coverage is often substantially lower, especially at 88 sessions. Validation also selects early-stopping weights, so these are empirical ranges, not guaranteed conformal intervals. Historical predictions use a fixed fitted model with rolling observed input windows; repeated walk-forward refitting and transaction-cost backtests remain separate research studies.

## News validation

The default keyword scorer was evaluated on **2,264 independently expert-labeled financial sentences** from the original [FinancialPhraseBank source](https://huggingface.co/datasets/takala/financial_phrasebank). The source revision, archive checksum, attribution and CC-BY-NC-SA-3.0 license are recorded in [benchmark provenance](data/benchmarks/financial_phrasebank_provenance.json). No scorer fitting was performed on this benchmark.

- Accuracy: **69.61%**, versus **61.44%** for the majority class.
- Balanced accuracy: **49.39%**; macro F1: **0.507**.
- Negative-example recall: **15.84%**. This is a material weakness.
- Retrieval smoke Recall@1: 1.0 on two explicitly synthetic queries; this is not a real-world retrieval benchmark.
- News confidence/impact scores remain heuristics, not calibrated probabilities.
- Latest verified news refresh found articles for **9/57** configured symbols. The RSS feed is finite; empty results are preserved as unavailable evidence. Broader coverage can use NEWS_API_KEY.
- Current price backtests do not mix in today's news. Historical fusion requires archived articles available at each prediction origin.

## Schedule and operation

Windows task **Equimaster-Daily-Refresh** is registered and Ready. It runs weekdays at **18:30 Asia/Kolkata**, with the selected Python interpreter and project working directory, while the registering Windows user is logged in. The next scheduled weekday is **26 October 2026**. The manual end-to-end run was verified; the future scheduled trigger has not yet fired.

Run the dashboard with `python -m streamlit run app.py`. See [README](README.md) for setup, optional neural news models, evaluation commands, scheduling and limitations.

## Remaining research, beyond feature implementation

- Improve and independently validate models that do not beat persistence.
- Improve negative-news detection and calibrate sentiment confidence.
- Collect timestamped, independently labeled Indian-market news for regional sentiment/retrieval and historical fusion validation.
- Evaluate interval coverage on additional untouched periods and refitted walk-forward strategies.

No production profitability or calibrated-confidence claim is made. Existing legacy models and prior version bundles remain available as backups; no source files or Git history were deleted.

## Fundamentals expansion ? 24 October 2026

The Fundamentals tab now supports 96 parameters. All 56 configured companies were
refreshed from public Screener and Yahoo sources, with 54?89 populated parameters
per company (TCS: 88). All 56 returned quarterly results, annual profit/loss, balance
sheet, cash flow, historical ratios and shareholding sections. Ranges are visible,
sector-aware application heuristics, with compatible peer valuation bands where possible.

Monetary fields carry their own currency, including verified USD statements for Infosys
and INR annual statements for HCL. Ambiguous cross-currency valuation multiples are
withheld. Missing bank disclosures and pledged shares are not invented.

See [Fundamentals guide](FUNDAMENTALS_GUIDE.md) for the metrics, sources, formulas,
range interpretation, exports and refresh commands. Snapshots refresh with the existing
daily workflow after 24 hours. Source warnings and per-company status are recorded in
`logs/fundamentals_refresh.json`.

## Dashboard redesign - 24 October 2026

Implemented a Trendlyne-inspired research layout within Streamlit, preserving
Equimaster's dark background and cyan/green palette:
shared navigation, six dated index cards, benchmark and coverage charts, sector
moves, gainers/losers, recent news and a functional screener.
Company pages include a quote header, real financial/valuation/momentum cards and
six research sections that render on selection. Technical charts, forecasts,
fundamentals, statements, exports and model operations remain connected.

Browser checks verified screener search, native navigation, fundamentals,
technical charts and a generated TCS forecast. No browser JavaScript errors were
reported. The offline suite passed 56 tests. See [UI design notes](UI_DESIGN.md)
for implementation and screenshot commands.

The dark-theme contrast correction covers native controls, selected chips,
captions, company facts, dataframe headers and assessment cells, plus every
Plotly view. The native configuration and custom surfaces now use compatible
foreground/background pairs; captions retain full opacity.
