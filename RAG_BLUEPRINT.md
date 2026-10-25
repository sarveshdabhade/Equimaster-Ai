# Equimaster RAG Blueprint

## Product scope

- MVP: cached ticker news, retrieval, FinBERT or fallback sentiment, evidence in Streamlit.
- Beta: recency-aware queries, source weighting, evaluation datasets, and scheduled ingestion.
- Production: persistent embeddings, provider failover, monitoring, model versioning, and authenticated API services.

## Runtime architecture

`market data -> preprocessing -> LSTM forecast`

`news provider -> normalization -> cache -> chunking -> retrieval -> sentiment -> market context`

`LSTM forecast + technical momentum + news sentiment -> dashboard decision signal`

The current integration keeps sentiment outside the existing LSTM tensor so previously trained models remain compatible. A future learned fusion model must regenerate sequences with sentiment features and retrain.

Each news chunk retains its ticker, article/chunk IDs, source, original and
canonical URLs, publication and observation timestamps, and duplicate-source
provenance. IDs are stable across tracking-parameter URL variants, and embedding
metadata can be refreshed without re-encoding unchanged chunk text.
Historical sentiment loads archived snapshots only, rejects missing timestamps,
and applies publication and observation cutoffs before retrieval; it never falls
back to a live provider for an `as_of` request.

## Forecast task contract

- Universe: each ticker with a valid processed daily-price dataset.
- Forecast origin: after the close of the latest available trading-session row.
- Target: the `Close` price at the end of the requested future trading-session horizon.
- Display periods: Day (1 session), Week (5), Month (22), and Quarter (88).
  Quarter retains the existing 88-session model and therefore spans approximately
  four months. Values are trading-session counts, not calendar-day durations.
- Primary benchmark: persistence, which predicts the last observed close for the
  target horizon.
- Evaluation: chronological held-out predictions, reported per ticker and horizon
  using MAE, RMSE, MAPE and directional accuracy. Compare every learned/fused
  system against persistence and the existing LSTM; do not pool raw price errors
  across tickers with different price scales.
- Information cutoff: any future LLM/RAG feature for an origin may use only
  evidence available no later than that origin. Article publication/observation
  timestamps and market feature timestamps must be retained for leakage checks.

This contract defines what the next integration steps must preserve. It does not
claim that the current heuristic signal improves LSTM accuracy.

`src/fusion_dataset.py` validates schema-versioned fusion evaluation rows. Each
row binds ticker, UTC forecast origin, shared trading-session horizon, target
date, current/LSTM-predicted/realized closes, heuristic news score, LLM stance
and cited chunk IDs, evidence publication/observation times, and LLM/LSTM model
IDs. It rejects post-origin evidence and duplicate ticker/origin/horizon rows.
`build_fusion_dataset` joins LSTM backtest rows to LLM feature rows by the exact
normalized ticker and origin, then validates the result.
`llm_feature_row_from_sentiment` converts a successful sentiment response and
its cited timestamp provenance into the feature-row format used by the join.
The target session date must come from the aligned market-price data; this
validator does not infer exchange sessions from calendar days and does not train
a fusion model.

Before fitting any fusion weights, `src/fusion_split.py::chronological_purged_split`
uses global forecast-origin boundaries (default 60% train, 20% validation, 20%
test) and purges train/validation rows whose target close would not yet have
been observed at the next split boundary. The same date boundary is applied
across tickers and horizons to avoid cross-group time leakage.

`src/fusion_model.py::fit_fusion_model` fits an initial standardized Ridge
regressor on train rows only. Features are LSTM log-return, heuristic news
sentiment, mapped LLM stance, and log horizon; the target is realized log-return.
Validation and test rows are scored separately and never used to fit the scaler
or regressor. This first learned fusion model is experimental and must beat the
same-split persistence/LSTM baselines before any production use.

`src/fusion_artifacts.py` stores fitted fusion models in immutable version
bundles with estimator hashes, feature/schema versions, model IDs, split
boundaries, dataset hashes, and validation/test metrics. It does not publish an
active-model pointer; load only artifacts created and controlled locally because
joblib model files are executable pickle data.
`src.fusion_model.py::predict_fused_forecast` validates each request's evidence
cutoff and its LLM/LSTM model IDs against training provenance before returning
an explicitly experimental prediction. It is not connected to the production
forecast path.

`src/fusion_gate.py::assess_fusion_staged_trial` is a conservative, non-mutating
evaluation gate. It requires coverage of every configured horizon, validation
non-inferiority to LSTM, test macro MAE improvement over both LSTM and
persistence, and no per-ticker/horizon test regression in MAE or directional
accuracy versus LSTM. Passing only permits consideration of a staged shadow
trial; it never activates the fusion model. This initial diagnostic is not a
significance test, production approval, or evidence of generalizable accuracy.

The LSTM forecast view supports opt-in shadow display when
`EQUIMASTER_FUSION_SHADOW_VERSION` names an immutable local fusion artifact.
It requests archived, cutoff-safe LLM evidence at the latest session close and
shows the candidate separately from the existing LSTM forecast. Missing or
incompatible artifacts/evidence are reported as shadow status and do not
replace the LSTM result, combined signal, chart, or CSV. No latest-version
selection or production activation is performed.

`src/fusion_evaluation.py::evaluate_fusion_baselines` compares persistence,
LSTM-only, news-combined, LLM-stance-only, and LSTM+LLM fixed-rule results per
ticker and horizon, with macro-averages rather than pooled raw price errors.
It uses the existing 0.6/0.4 combined-signal rule; LLM stance maps bullish to
`+1`, bearish to `-1`, and mixed/neutral to `0`. These are untrained diagnostic
baselines over supplied rows, not a chronological holdout or evidence of
improvement. Any learned weights need a later purged chronological evaluation.

## LLM provider boundary

The optional provider adapter lives in `src/llm/provider.py`. It supports
OpenAI-compatible `/chat/completions` endpoints behind a small `LLMProvider`
interface. `EQUIMASTER_LLM_PROVIDER` defaults to `disabled`; no LLM network
requests or model downloads occur unless explicitly configured. Hosted endpoints
must use HTTPS; loopback HTTP is permitted for local inference servers.

Configuration is provided through environment variables:

- `EQUIMASTER_LLM_PROVIDER`: `disabled` or `openai_compatible`
- `EQUIMASTER_LLM_BASE_URL`: API root, such as `https://api.example/v1`
- `EQUIMASTER_LLM_MODEL`: provider model identifier
- `EQUIMASTER_LLM_API_KEY`: optional for local endpoints; keep hosted keys in
  process secrets, never in source or logs
- `EQUIMASTER_LLM_TIMEOUT_SECONDS`: request timeout, default 30

Sending retrieved article text to a hosted model shares that text with the
configured provider. The News & Sentiment view sends chunks only when a provider
is configured; provider errors are surfaced and do not replace the heuristic
sentiment score with a synthetic LLM result. Retry and redaction policies remain
for later integration steps.

The opt-in `synthesize_retrieved_evidence` operation in `src/llm/evidence.py`
accepts only timestamped retrieved chunks at or before an explicit `as_of`
cutoff. It returns a structured stance, summary, and claims, each with citations
restricted to supplied chunk IDs. Schema and citation membership are validated;
citations provide traceability, not a guarantee that a model's wording is
semantically entailed by the cited text. It is opt-in from the News & Sentiment
view and does not change the deterministic sentiment score or LSTM forecast signal.

`evaluate_llm_synthesis_dataset` in `src/llm/evaluation.py` requires an explicitly
injected provider and independently labeled cases containing expected stances
and relevant evidence IDs. It reports stance accuracy/macro-F1 and citation
precision/recall/F1. Use independent annotations; citation overlap is not a
semantic-grounding or forecast-accuracy metric.

## Contracts

News records contain `ticker`, `source`, `title`, `summary`, `url`, `published_at`, `text`, `content`, `sentiment`, `confidence`, `relevance`, and `source_weight` where applicable.

Market context contains technical signal, price trend, sentiment state, combined score, label, and summary.

## Success metrics

- Forecast: MAE, MAPE, directional hit rate by horizon.
- Sentiment: labeled accuracy, positive/negative precision and recall, score calibration.
- Retrieval: article-level Precision@K, Recall@K, MRR and nDCG@K on a labeled
  set; canonical URLs collapse tracking variants and duplicate chunks.
- LLM synthesis: stance accuracy/macro-F1 and citation precision/recall/F1 on an
  independently labeled set; evaluate before using LLM output in forecasting.
- Operations: cache hit rate, provider failure rate, and median UI response time.

## Accuracy evaluation API

Use `src.evaluate_predictions.evaluate_predictions` with aligned arrays:

- `current_close`: price known at prediction time
- `predicted_close`: LSTM output for the selected horizon
- `actual_close`: observed future close
- `sentiment_scores`: RAG scores available before each prediction time

The result includes `mae`, `rmse`, `mape`, `lstm_directional_accuracy`,
`combined_directional_accuracy`, and `sentiment_return_correlation`.

Do not use the current live news cache as historical evidence. A valid backtest
needs timestamped articles and must filter each prediction to articles published
before its prediction timestamp.
