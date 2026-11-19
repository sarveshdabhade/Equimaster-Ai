# Equimaster-Ai release review - 19 November 2026

The research dashboard's application and artifact checks pass. All 59 configured
stock/index models cover Day, Week, Month and Quarter (1, 5, 22 and 63 sessions).
All 58 configured companies have fresh fundamentals and supported valuation scenarios.
Forecasting performance and historical fusion remain research work; passing software
checks does not establish predictive advantage or a trading edge.

## Verified release evidence

- 168 tests pass in the clean Windows/Python 3.13 lockfile environment; pip check passes.
- The isolated source checkout is also tested without the workspace's runtime data caches.
- 59/59 raw and processed datasets pass quality checks; provider reference ends 2026-11-19.
- 59/59 active bundles match their current processed datasets, scalers and model hashes.
- All 236 ticker/horizon inference checks return finite positive forecasts and ordered intervals.
- All 59 bundles record at least 20 fresh verified news articles at fitting; the gate was retained.
- 58/58 company valuation scenarios pass arithmetic, currency and freshness checks.
- 10/10 browser checks pass, including exports, forecast generation, mobile and unavailable states.
- The registered Task Scheduler pipeline completed with result 0. The ordinary weekday trigger remains 18:30 Asia/Kolkata, while this user is logged in.
- Remote CI passed for validated source baseline `234ac3ad`: https://github.com/sarveshdabhade/Equimaster-Ai/actions/runs/44559131716.
  The runtime archive's RELEASE_MANIFEST.json additionally records successful CI for its exact packaged source commit.

Evidence copies are in [reports/release-20261119](reports/release-20261119/). The runtime
archive contains source, current bundles, datasets, fundamentals, news archives,
evaluation artifacts and per-file checksums. It excludes environments, provider
cookie databases and secrets. Historical models/data remain on disk; 288 generated
paths were removed from Git tracking without deleting their files.

## Frozen chronological test results

Means are unweighted across the 59 active bundles. Persistence is evaluated on the
same ticker and terminal dates. Interval coverage is compared with a nominal 90%.

| Horizon | Models | Direction accuracy | LSTM MAPE | Persistence MAPE | MAE wins | Terminal interval coverage | Path coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day (1) | 59 | 27.25% | 1.299% | 1.296% | 13/59 | 77.80% | 77.80% |
| Week (5) | 59 | 35.18% | 2.972% | 2.960% | 19/59 | 76.09% | 77.21% |
| Month (22) | 59 | 38.34% | 6.653% | 6.565% | 16/59 | 59.94% | 68.52% |
| Quarter (63) | 59 | 39.08% | 9.800% | 9.129% | 18/59 | 37.79% | 52.66% |

Terminal coverage uses the target date scored by MAE/MAPE. Path coverage averages
all lead steps and is retained separately. Reports before 17 November used the path
average under the former coverage field. Intervals use validation residuals; that
segment also selects stopping weights and shrinkage, so these are not independently
calibrated or guaranteed conformal intervals. These are frozen fitted-model tests,
not monthly refitted walk-forward results or proof of an untouched final study.
The universe is the currently configured stock set, not a point-in-time constituent
database. Historical comparisons retain survivorship and corporate-action limits.

| Horizon | Net return per trade at illustrative 10 bp cost | Trades |
|---|---:|---:|
| 1 sessions | -0.066% | 916 |
| 5 sessions | +0.114% | 624 |
| 22 sessions | +0.248% | 314 |
| 63 sessions | +0.233% | 99 |

Cost inputs are illustrative round-trip sensitivities, not broker-specific charges.
The study uses non-overlapping long-only positions per ticker/horizon, equal weight
per trade and zero-return cash; it omits portfolio sizing, taxes, liquidity, financing,
market impact and uncertainty estimates. It does not establish profitability.

## News and fundamentals evidence

The keyword scorer was evaluated without fitting on 2,264 expert-labelled English
FinancialPhraseBank sentences: accuracy 76.19%, majority
baseline 61.44%, macro F1
0.626, and negative recall 27.06%.
This is not an Indian-market return benchmark; confidence remains uncalibrated.
The current annotation export contains 4,527 unlabelled
article rows across 58 companies. Independent human labels are still required.

Valuation reports share the company-page calculation, with default research
discount-rate assumptions, reported company beta, verified monetary units and
same-period FCFE inputs. Confidence labels describe method/input coverage; the
ranges are scenarios, not statistical confidence intervals. Page overrides can
produce different scenarios. The 252-session fundamentals return model remains
gated by matured point-in-time labels (minimum 30 matured snapshot months and 300
samples); no historical fundamentals were backfilled using today's observations.

## Remaining administrator and research work

1. Windows denied disabling the legacy `Equimaster daily update` task. Run
   `scripts/install_schedule.ps1` once as administrator to disable that duplicate.
   A shared operating-system lock prevents overlapping pipeline runs meanwhile.
2. Run preregistered refitted walk-forward studies on additional untouched origins
   with persistence baselines and realistic documented trading costs.
3. Separate model selection from interval calibration and evaluate additional
   untouched periods before claiming nominal coverage.
4. Obtain independent Indian-market news labels and point-in-time disclosure
   histories; validate sentiment, confidence and LSTM/news/fundamentals fusion.
   The dashboard blend remains a research heuristic; learned fusion stays shadow-only.

## Reproducible operation

Install requirements-lock.txt in a fresh Python 3.13 environment, restore the local
runtime archive or run the refresh pipeline, and launch `streamlit run app.py`.
The tested lockfile uses TensorFlow 2.20.0, Keras 3.13.2, pandas 2.3.3 and Streamlit
1.53.1. The existing training environment has newer TensorFlow/Streamlit packages;
the locked environment's inference and browser checks verify the saved weights
can be served under the documented installation.
