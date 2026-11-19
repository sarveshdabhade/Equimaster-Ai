# Equimaster-Ai

Streamlit stock research dashboard with LSTM price forecasts, technical charts,
company fundamentals, peer comparisons, and evidence-linked news sentiment.

## Setup

Tested on Windows with Python 3.13. Install in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m streamlit run app.py
```

A fresh clone needs market data and model artifacts. Restore the supplied local
runtime archive, or run `python daily_pipeline.py --workers 2 --epochs 60` first.
Provider or news shortages are recorded explicitly; the dashboard can show a
missing-model state while other research sections remain available. Use
[PROJECT_STATUS.md](PROJECT_STATUS.md) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)
for the current release evidence and remaining blockers.

`requirements.txt` pins direct runtime dependencies. `requirements-lock.txt` records
the installed core/test dependency closure for this Windows/Python version.
`requirements-dev.txt` includes pytest. Optional FinBERT dependencies are separate
in `requirements-news.txt`; the core app uses keyword sentiment and TF-IDF retrieval.

## Refresh, train, evaluate

```powershell
$env:NUMBA_DISABLE_JIT = "1"
python daily_pipeline.py --workers 1 --epochs 10
# Retrain one ticker and its currently active horizons:
python scripts/quick_run.py --ticker TCS.NS --epochs 10 --horizon 5
# Force a complete model refresh:
python scripts/train_manager.py --workers 3 --epochs 10
python scripts/evaluate_watchlist.py --horizons 1,5,22,63
python scripts/evaluate_cost_sensitivity.py
python scripts/evaluate_news.py
python -m pytest tests -q
```

GitHub Actions also runs the offline suite on Windows/Python 3.13 using the lockfile.

The daily pipeline refreshes raw data and computes indicators before training.
It force-fetches up to 20 live, verified news articles for each ticker that needs
training. A ticker with fewer than 20 is skipped and recorded as blocked; other
eligible tickers can proceed. The standalone trainer enforces the same requirement
for its ticker. News is a training prerequisite, not an LSTM input feature. News
refresh results and stage status are written under `logs/`. A failed market refresh
does not replace active models. Provider aliases preserve the internal `LTIM.NS`
name while requesting `LTM.NS` history.

All training entrypoints delegate to `src/training.py`; bulk jobs use the canonical
`scripts/train_manager.py`. Compatibility script names remain available. The
standalone data sequencer can export tensors for experiments; the canonical trainer
creates its own purged, chronological splits directly from processed CSVs.

## Model and evaluation contract

Each ticker has immutable bundles under `models/versions/<ticker>/<version>/`.
An atomic `models/active_<ticker>.json` pointer selects matching models, scaler,
feature order, training dates, data/model hashes, an immutable processed-data snapshot,
metrics, and backtest files.
Existing `.h5`/`.keras` files are retained for legacy fallback, without inventing
training dates or performance figures. Tiny provider rounding changes are ignored
when refreshing an otherwise identical history, preventing redundant retraining.

The scaler uses fitting rows only. Validation/calibration and test targets follow
chronologically; overlapping multi-session targets are purged at boundaries.
LSTM models learn corrections to the last observed close. Stored evaluations report
MAE, RMSE, MAPE and directional accuracy against a persistence baseline. Baseline MAE
can be lower than LSTM MAE.
The forecast task uses the last available close as its after-close origin and predicts
the close at the Day (1 session), Week (5), Month (22), or Quarter (63) horizon.
Quarter is approximately three calendar months of trading sessions. LLM/RAG evidence used for a
forecast must be available by that origin. See [forecast task contract](RAG_BLUEPRINT.md#forecast-task-contract).

Backtests are rolling predictions with a fixed fitted model and observed prior
prices, not a repeatedly refitted walk-forward study. `backtest_h*.csv` records
prediction origin and target dates. Direct-model intervals use validation residual
quantiles; the same validation period also selects early-stopping weights. These are
empirical intervals, not guaranteed conformal coverage. Test coverage is measured
in stored evaluation artifacts. The forecast interface uses direct multi-session forecasts;
stored directional accuracy is a historical holdout hit rate, not a probability or guarantee.
`scripts/evaluate_cost_sensitivity.py` applies a fixed non-overlapping long-only rule
to the saved test forecasts under 0, 10, 25 and 50 basis-point round-trip cost
scenarios. It is a holdout sensitivity report, not a refitted walk-forward or
broker-specific profitability estimate. See [validation protocol](VALIDATION_PROTOCOL.md).

Evaluation reports distinguish interval coverage at the terminal target from
coverage averaged over the entire predicted path. `interval_test_coverage` uses
the same terminal dates as MAE/MAPE; `interval_path_test_coverage` retains the
saved average over all lead steps. Historical reports created before 17 November
2026 used the path average under the former coverage field.

The LSTM forecast section shows the baseline alongside a final research estimate
that blends the LSTM return with retrieved-news sentiment and fresh cached fundamentals.
This fixed, volatility-scaled heuristic has not been validated as more accurate than
the LSTM or persistence; the LSTM-only holdout metrics do not measure the fused estimate.
The separately trained LSTM/news fusion experiment remains shadow-only.

## Dashboard features

- Dark research workspace with shared navigation, compact index quotes, market
  breadth, sector moves, gainers/losers, a working stock screener and recent news.
  Company pages use a compact quote header and load research
  sections on selection. Mobile layouts include scrolling index and research strips.
  See [UI design and browser checks](UI_DESIGN.md).
- The six market-index cards and NIFTY chart poll the latest Yahoo Finance quote
  every 30 seconds during regular market hours and show the provider timestamp.
  One-minute history is used when quote metadata is unavailable; provider delay can vary.
- Stock analyzer: price overview, technical charts, LSTM forecasts, news, and fundamentals.
- Technical chart: date range, daily/weekly/monthly candles, correctly aggregated
  volume, ATR, annualized volatility, and period return.
- Forecast: direct multi-session forecasts, persistent results across reruns and CSV export.
- Fundamentals: 98 supported parameters across valuation, profitability, growth,
  debt/liquidity, balance sheet, cash flow, efficiency, dividends and ownership.
  Public Screener tables provide quarterly/annual results, balance sheets, cash flows,
  historical ratios and shareholding history; Yahoo supplies additional fields.
  Each value shows source, reporting basis, period and currency. All 58 configured
  companies are cached; actual field availability varies and missing data remains N/A.
  Ratio assessments display their screening ranges. Suitable selected peers produce
  valuation bands at 80?120% of the peer median; these are application heuristics,
  not universal good/bad rules. Industrial-company ratings are disabled for banks,
  insurers and lenders. CSV/JSON and individual financial-statement exports are available.
  Yahoo statement currencies are checked when quote and financial-profile currencies
  differ; USD amounts are never labeled INR, and ambiguous valuation multiples are withheld.
  SWOT & Valuation adds evidence-linked strengths, weaknesses, opportunities and threats,
  and a 60% DCF / 40% same-industry peer estimate when both methods qualify. Industrial
  DCF requires same-period operating cash flow, capex, and reported net borrowing to derive
  FCFE; peer coverage requires at least two comparable companies. Additional peer symbols
  can be entered when the configured universe has no industry matches. The raw SWOT balance
  averages signals within evidence families and is an uncalibrated screening summary;
  fixed research defaults and provider-reported beta
  do not make it a validated rating. Buy requires full valuation,
  at least 15% upside and a validated ML estimate of at least 5% 12-month excess return
  versus NIFTY 50. Sell-to-exit follows the configured negative-return or full-value rules.
  A dated monthly fundamentals archive is being collected; current snapshots are never
  backdated. The return model remains pending until enough 252-session outcomes mature
  and purged walk-forward checks pass. Refreshes and the daily pipeline collect snapshots
  and train when eligible; `python scripts/archive_current_fundamentals.py` seeds the
  archive from existing current caches without claiming historical observations.
  Refresh all companies with `python scripts/refresh_fundamentals.py`; the daily workflow
  refreshes snapshots older than 24 hours. No private Screener sections are accessed.
- Model training and evaluation run through the backend pipeline and CLI; use
  `python daily_pipeline.py` for the scheduled workflow or
  `python scripts/train_manager.py` for model training.

## News and historical validation

The live news fetcher queries Yahoo Finance, Google News RSS, NewsAPI (when
`NEWS_API_KEY` is configured), and direct publisher RSS feeds from Economic
Times, CNBC, Business Standard, LiveMint, and Hindu BusinessLine concurrently.
Google News runs broad and focused company/symbol searches in parallel.
Google News results are accepted only when the feed attributes them to a
publisher whose domain is on the trusted-source allowlist, including specialist
coverage from India Shipping News and ET Infra, plus MarketWatch, CNBC TV18,
Zee Business, The Hindu, and both Economic Times domains. Searches include
ticker and company-name aliases for every symbol in the configured NIFTY list.
If no recent verified evidence is available, sentiment is shown as unavailable
rather than presenting an unsupported neutral score. Publisher feeds and Google
RSS and per-ticker results are cached for up to four minutes;
**Refresh feed** bypasses these caches and queries providers again. The panel
shows up to 20 verified articles published in the last 30 days, matched to the
company/ticker and deduplicated. Coverage is
provider-dependent: no publisher or API guarantees a current story for every
company. A successful refresh with no matching items means no eligible article
was returned; it does not mean the company has no news. Provider failures are
logged without exposing API keys.

Fetch a fresh 30-day news batch for every configured stock and save the article
snapshots to the news cache/archive:

```powershell
python scripts/refresh_all_news.py
```

To poll the repository's configured NIFTY symbols continuously (default interval
15 minutes), keep this process running:

```powershell
python scripts/news_ingest.py
```

The `NIFTY_50_TICKERS` list contains the 50 constituents from the NSE Indices
constituent CSV checked on 26 October 2026. Recheck that source after index
rebalances. Override the symbols or interval with `NEWS_TICKERS` and
`INTERVAL_SECONDS`. Optional NewsAPI coverage can be enabled by setting
`NEWS_API_KEY` in the local process environment; do not commit the key. Provider
publication cadence and outages mean this is near-real-time polling, not a
guarantee of instant or complete news.

Company matching uses configured ticker and cached company-name aliases; short
symbol matches require financial context to reduce ambiguous false matches.
Publication times are displayed in India time.

The normalizer retains missing publication times instead of replacing them with
retrieval time. Historical retrieval requires both publication and observation
timestamps, and rejects articles published or first observed after the prediction
origin. Historical sentiment requires explicit archived articles; it cannot silently
use the current live cache. Feed timestamps are normalized to UTC; URL tracking
parameters and same-day near-duplicate headlines are deduplicated while source and
URL provenance is retained.
Fresh observations are archived in `data/news_archive/` for future labeled studies.

An OpenAI-compatible LLM adapter is available in `src/llm/provider.py` but is
disabled by default and is not yet connected to forecast inference. To configure a
hosted or local endpoint, set `EQUIMASTER_LLM_PROVIDER=openai_compatible`,
`EQUIMASTER_LLM_BASE_URL`, and `EQUIMASTER_LLM_MODEL`; set
`EQUIMASTER_LLM_API_KEY` for hosted providers. Hosted endpoints must use HTTPS.
Article text sent to a hosted model is shared with that provider. See
[LLM provider boundary](RAG_BLUEPRINT.md#llm-provider-boundary).

Prepare a locally archived, timestamped batch for independent Indian-market
news annotation with `python scripts/export_news_annotation_batch.py --days 30`.
The exported labels are blank by design; sentiment labels require independent
human review before the batch can support accuracy claims.

To enable an already-cached FinBERT model:

```powershell
python -m pip install -r requirements-news.txt
$env:EQUIMASTER_FINBERT = "1"
# Allow model downloads only when desired:
$env:EQUIMASTER_ALLOW_MODEL_DOWNLOADS = "1"
```

Embeddings are optional; install compatible `sentence-transformers` and `faiss-cpu`
for your platform before enabling `EQUIMASTER_EMBEDDINGS=1`. They are not required
for TF-IDF retrieval. No model download occurs by default.

The bundled `tests/fixtures/news_evaluation.json` is an explicitly **synthetic smoke
corpus**. Its accuracy, per-class precision/recall/F1, confidence error and retrieval
Recall@K verify behavior; they do not establish real-world financial-news accuracy.
A real sentence-level benchmark is also supported:

```powershell
python scripts/download_news_benchmark.py
python scripts/evaluate_news.py --dataset data/benchmarks/financial_phrasebank.json --output data/evaluation/news_financial_phrasebank.json
```

The downloader reads the original authors' [FinancialPhraseBank dataset](https://huggingface.co/datasets/takala/financial_phrasebank),
records its revision, checksum, attribution and CC-BY-NC-SA-3.0 license, and evaluates
the 2,264 unanimously labeled sentences. This measures the configured scorer without
fitting on the benchmark. The evaluator also reports a majority-class baseline and
balanced accuracy. If enabling a pretrained FinBERT model, check its training data
before interpreting FinancialPhraseBank as an unseen test set.

Use `python scripts/evaluate_news.py --dataset <independently-labeled-corpus.json>`
for additional regional or timestamped studies. Heuristic news confidence and impact scores are not calibrated
probabilities. A timestamped, independently labeled Indian-market corpus remains necessary
before making production accuracy claims or validating historical fusion benefits.

## Scheduled operation on Windows

```powershell
.\scripts\install_schedule.ps1
Get-ScheduledTask -TaskName Equimaster-Daily-Refresh
```

The task runs weekdays at **18:30 Asia/Kolkata**, while the registering user is logged
in. It uses the selected interpreter and project working directory, skips overlapping
runs, starts missed runs when available, and has a four-hour execution limit.
To remove it: `Unregister-ScheduledTask -TaskName Equimaster-Daily-Refresh`.
The installer and `run_pipeline.bat` use `.venv\Scripts\python.exe` by default.
The installer uses two training workers; override with `-Workers 1` if needed.
A shared operating-system lock prevents overlap between separate schedules and
manual pipeline runs. Stage results are saved incrementally to
`logs/daily_status.json`; bounded stage timeouts terminate child process trees.
The installer also attempts to disable the legacy `Equimaster daily update` task.
If Windows denies that operation, rerun the installer once as administrator.

Release verification commands:

```powershell
python scripts/audit_universe.py
python scripts/valuation_coverage.py
python scripts/validate_runtime.py
# Also require every bundle to match the latest processed data:
python scripts/validate_runtime.py --require-current
python -m pytest tests -q
# Optional: install requirements-ui.txt, start the app on port 8502, then:
python scripts/validate_browser.py
# After reviewing and committing the source, package current runtime artifacts:
python scripts/package_release.py --output D:/Project-Backups/Equimaster-Ai-release.zip
```

## Artifacts and repository hygiene

Keep raw/processed data and matching scaler/model bundles for reproducibility.
Training arrays, Python caches, news caches, generated images and logs can be
regenerated. Runtime data/models/logs are ignored by Git. Files already tracked
before those ignore rules remain tracked until separately reviewed and untracked.
No cleanup command should remove `.git/`, source code or trained bundles blindly.
