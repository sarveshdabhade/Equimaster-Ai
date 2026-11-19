from datetime import date, timedelta

import numpy as np

from src.fusion_model import fit_fusion_model
from src.fusion_split import chronological_purged_split


def _records():
    records = []
    for day in range(40):
        origin_date = date(2026, 1, 1) + timedelta(days=day)
        origin = f"{origin_date.isoformat()}T16:00:00+00:00"
        for ticker, scale in (("TEST.NS", 1.0), ("OTHER.NS", 5.0)):
            for horizon in (1, 5):
                current = scale * (100.0 + day)
                actual_return = 0.01 + 0.003 * np.sin(day / 3)
                lstm_return = actual_return + 0.002 * np.cos(day / 4)
                stance = "bullish" if actual_return > 0 else "bearish"
                records.append({
                    "schema_version": 1,
                    "ticker": ticker,
                    "as_of": origin,
                    "horizon": horizon,
                    "target_date": (origin_date + timedelta(days=horizon)).isoformat(),
                    "current_close": current,
                    "lstm_predicted_close": current * (1 + lstm_return),
                    "actual_close": current * (1 + actual_return),
                    "news_sentiment_score": 0.2,
                    "llm_stance": stance,
                    "llm_evidence_ids": [f"{ticker}-{day}"],
                    "llm_model_id": "llm-test-v1",
                    "lstm_model_id": "lstm-test-v1",
                    "evidence": [{
                        "evidence_id": f"{ticker}-{day}",
                        "published_at": "2025-12-30T00:00:00+00:00",
                        "observed_at": "2025-12-30T01:00:00+00:00",
                    }],
                })
    return records


def test_fusion_model_fits_train_only_and_reports_validation_and_test():
    split = chronological_purged_split(_records())
    result = fit_fusion_model(split)

    assert result["training_sample_count"] == len(split["train"])
    assert result["validation"]["sample_count"] == len(split["validation"])
    assert result["test"]["sample_count"] == len(split["test"])
    assert result["feature_names"] == [
        "lstm_log_return", "news_sentiment_score", "llm_stance_score", "log_horizon"
    ]
    assert len(result["model"].named_steps["ridge"].coef_) == 4
    assert result["test"]["per_ticker_horizon"]
    assert np.isfinite(result["test"]["macro_average"]["fused_mae"])


def test_fusion_model_parameters_do_not_depend_on_validation_or_test_targets():
    split = chronological_purged_split(_records())
    first = fit_fusion_model(split)
    changed = {
        name: [dict(row) for row in rows]
        for name, rows in split.items()
        if name in {"train", "validation", "test"}
    }
    for partition in ("validation", "test"):
        for row in changed[partition]:
            row["actual_close"] *= 1.25
    second = fit_fusion_model(changed)

    np.testing.assert_allclose(
        first["model"].named_steps["ridge"].coef_,
        second["model"].named_steps["ridge"].coef_,
    )
    assert first["test"]["macro_average"]["fused_mae"] != second["test"]["macro_average"]["fused_mae"]
