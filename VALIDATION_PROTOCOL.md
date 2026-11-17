# Forecast and news validation protocol

Use this protocol for each new model or forecast-contract change. Freeze the
universe, data cutoff, feature contract, and decision rule before opening the test
results. Record the Git commit, pipeline version, data hashes, and artifact versions.

## Forecast evaluation

1. Use the configured stock universe and NIFTY index model as they existed at the
   evaluation cutoff. Report missing or stale bundles explicitly.
2. Keep dates in chronological order. Fit scalers on training rows only. Use a
   later validation segment for early stopping, a separate calibration segment for
   interval residuals, and a final test segment that is not used to choose models,
   thresholds, or features.
3. Purge at least `horizon - 1` target rows at each split boundary so a multi-session
   label cannot cross from one partition into another.
4. Evaluate Day (1), Week (5), Month (22), and Quarter (63) separately. Compare each
   with persistence on the same ticker, origin dates, and target dates. Report MAE,
   RMSE, MAPE, directional accuracy, sample counts, and per-ticker wins; do not
   average raw-price MAE across stocks with different prices.
5. For interval estimates, report empirical coverage and width by horizon against
   the 90% nominal target. Do not call validation-residual ranges calibrated or
   conformal intervals. Use terminal-target coverage on the same rows as terminal
   forecast error; report the all-lead path average separately. Historical reports
   before 17 November 2026 used the path average under `interval_test_coverage`.
6. For refitted walk-forward results, refit only on observations available before
   each origin. Keep validation and calibration data inside that history, purge
   overlapping targets, and score the next untouched block. Fix a monthly refit
   cadence and use a five-year trailing window unless a later study preregisters a
   different schedule. Aggregate results across origins only after all origins are
   complete.

## Cost sensitivity

The current `scripts/evaluate_cost_sensitivity.py` study uses saved frozen-model
test predictions, selects non-overlapping positions per ticker and horizon, and
enters long only when the forecast return exceeds the round-trip cost scenario.
It reports 0, 10, 25, and 50 basis-point scenarios against a zero-return cash
baseline. The scenarios are sensitivity inputs, not estimates of any specific
broker's charges. A later portfolio study must add actual taxes, brokerage,
slippage, liquidity limits, position sizing, and concurrent capital allocation.

Do not optimize the entry rule on the final test. Choose any threshold using
training/validation data, freeze it, and then report the test once. Include gross
and net results, trade counts, participation, drawdown, turnover, and uncertainty
across tickers and origins.

## News and forecast fusion

For each forecast origin, include an article only if both publication and first
observation timestamps are present and no later than that origin. Apply the same
cutoff to fundamentals using their disclosure/effective periods; retrieval time
alone is not a historical disclosure date. Keep news labels independent from the
model developers and split by time and issuer to limit duplicate-story leakage.

Report class balance, majority baseline, accuracy, macro F1, balanced accuracy,
negative-class precision/recall, and calibration error. Synthetic fixtures are
behavior checks only. FinancialPhraseBank is an English sentence benchmark, not an
Indian-market return benchmark. Do not tune on it and then describe it as an
unseen test.

The exported Indian-market annotation batch has blank labels by design. Historical
fusion remains unvalidated until independent labels and point-in-time price targets
are available. Keep any fusion output shadow-only until it beats the LSTM and
persistence baselines on a frozen, untouched temporal test.

## Current evidence

- Frozen LSTM test summaries: `data/evaluation/current-universe-20261117.json`.
- Cost sensitivity: `data/evaluation/cost-sensitivity-20261117.json`.
- Runtime model checks: `data/evaluation/runtime_validation.json`.
- Universe data audit: `data/evaluation/data_quality.json`.
- All-stock valuation inputs and scenarios: `data/evaluation/valuation_coverage.json`.
- Browser acceptance: `data/evaluation/browser_validation.json`.
- News annotation template: `data/benchmarks/indian_market_news_annotations_20261117.csv`.
- Current sentiment benchmark and provenance: `data/evaluation/news-financialphrasebank-20261117.json` and `data/benchmarks/financial_phrasebank_provenance.json`.

The retained October reports describe earlier checkpoints. Release evidence is
also copied into `reports/release-20261117/` so it remains reviewable without the
local runtime dataset archive.

These files describe evidence and coverage limits; they do not certify a trading edge.
