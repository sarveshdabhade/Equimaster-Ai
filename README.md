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
python scripts/evaluate_watchlist.py --horizons 1,5,22,88
python scripts/evaluate_news.py
python -m pytest tests -q
```

GitHub Actions also runs the offline suite on Windows/Python 3.13 using the lockfile.

The daily pipeline refreshes raw data, computes indicators, trains changed datasets,
evaluates active bundles, refreshes news, and updates stale company fundamentals. Logs and stage status are written under
`logs/`. A failed market refresh does not replace active models. Provider aliases
preserve the internal `LTIM.NS` name while requesting `LTM.NS` history.

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
LSTM models learn corrections to the last observed close. Each horizon reports MAE,
RMSE, MAPE and directional accuracy against a persistence baseline. Baseline MAE can
be lower than LSTM MAE; the dashboard displays this rather than assuming superiority.
The forecast task uses the last available close as its after-close origin and predicts
the close at the Day (1 session), Week (5), Month (22), or Quarter (88) horizon.
The Quarter label retains the existing 88-session model, which spans approximately
four months. LLM/RAG evidence used for a
forecast must be available by that origin. See [forecast task contract](RAG_BLUEPRINT.md#forecast-task-contract).

Backtests are rolling predictions with a fixed fitted model and observed prior
prices, not a repeatedly refitted walk-forward study. `backtest_h*.csv` records
prediction origin and target dates. Direct-model intervals use validation residual
quantiles; the same validation period also selects early-stopping weights. These are
empirical intervals, not guaranteed conformal coverage. Test coverage is measured
and displayed. The forecast interface uses direct multi-session forecasts only;
reported directional accuracy is a historical holdout hit rate, not a probability
or guarantee.

The displayed forecast/news fusion and evaluation share one signal formula:
`0.6 * tanh(predicted_return * 10) + 0.4 * sentiment`. The separate Market Summary
uses technical momentum and news; it is labeled as a heuristic research summary.

## Dashboard features

- Dark research workspace with shared navigation, compact index quotes, market
  breadth, sector moves, gainers/losers, a working stock screener and recent news.
  Company pages use a compact quote header and load research
  sections on selection. Mobile layouts include scrolling index and research strips.
  See [UI design and browser checks](UI_DESIGN.md).
- Stock analyzer: price overview, technical charts, forecasts, news, fundamentals,
  and a connected Market Summary with recommendation/risk heuristics.
- Technical chart: date range, daily/weekly/monthly candles, correctly aggregated
  volume, ATR, annualized volatility, and period return.
- Forecast: direct multi-session forecasts, persistent results across reruns, CSV export,
  real held-out metrics, baseline comparison and historical-prediction export.
- Fundamentals: 96 supported parameters across valuation, profitability, growth,
  debt/liquidity, balance sheet, cash flow, efficiency, dividends and ownership.
  Public Screener tables provide quarterly/annual results, balance sheets, cash flows,
  historical ratios and shareholding history; Yahoo supplies additional fields.
  Each value shows source, reporting basis, period and currency. All 56 configured
  companies are cached; actual field availability varies and missing data remains N/A.
  Ratio assessments display their screening ranges. Suitable selected peers produce
  valuation bands at 80?120% of the peer median; these are application heuristics,
  not universal good/bad rules. Industrial-company ratings are disabled for banks,
  insurers and lenders. CSV/JSON and individual financial-statement exports are available.
  Yahoo statement currencies are checked when quote and financial-profile currencies
  differ; USD amounts are never labeled INR, and ambiguous valuation multiples are withheld.
  Refresh all companies with `python scripts/refresh_fundamentals.py`; the daily workflow
  refreshes snapshots older than 24 hours. No private Screener sections are accessed.
- Model training and evaluation run through the backend pipeline and CLI; use
  `python daily_pipeline.py` for the scheduled workflow or
  `python scripts/train_manager.py` for model training.

## News and historical validation

The live news fetcher merges available results from Yahoo Finance, Google News
RSS, NewsAPI (when `NEWS_API_KEY` is configured), and direct publisher RSS feeds
from Economic Times, CNBC, Business Standard, LiveMint, and Hindu BusinessLine.
Google News results are accepted only when the feed attributes them to a
publisher whose domain is on the trusted-source allowlist, including specialist
coverage from India Shipping News and ET Infra, plus MarketWatch, CNBC TV18,
Zee Business, The Hindu, and both Economic Times domains. Searches include
ticker and company-name aliases for every symbol in the configured NIFTY list.
If no recent verified evidence is available, sentiment is shown as unavailable
rather than presenting an unsupported neutral score. Publisher feeds and Google
RSS searches are cached for up to five minutes per running process;
**Refresh feed** bypasses the per-ticker cache and queries providers again while
reusing RSS data fetched in the last five minutes. News is filtered to recent
publication dates and company/ticker matches, then deduplicated. Coverage is
provider-dependent: no publisher or API guarantees a current story for every
company. A successful refresh with no matching items means no eligible article
was returned; it does not mean the company has no news. Provider failures are
logged without exposing API keys.

To poll the repository's configured NIFTY symbols continuously (default interval
15 minutes), keep this process running:

```powershell
python scripts/news_ingest.py
```

The current `NIFTY_50_TICKERS` configuration contains 49 symbols, so verify and
add the current 50th index constituent before describing this poller as complete
NIFTY 50 coverage. Override its symbols or interval with `NEWS_TICKERS` and
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
.\scripts\install_schedule.ps1 -PythonPath (Get-Command python).Source
Get-ScheduledTask -TaskName Equimaster-Daily-Refresh
```

The task runs weekdays at **18:30 Asia/Kolkata**, while the registering user is logged
in. It uses the selected interpreter and project working directory, skips overlapping
runs, starts missed runs when available, and has a four-hour execution limit.
To remove it: `Unregister-ScheduledTask -TaskName Equimaster-Daily-Refresh`.
`run_pipeline.bat` uses the Python interpreter on PATH, so activate your environment
first. Scheduled tasks retain their explicitly selected interpreter.

## Artifacts and repository hygiene

Keep raw/processed data and matching scaler/model bundles for reproducibility.
Training arrays, Python caches, news caches, generated images and logs can be
regenerated. Runtime data/models/logs are ignored by Git. Files already tracked
before those ignore rules remain tracked until separately reviewed and untracked.
No cleanup command should remove `.git/`, source code or trained bundles blindly.
