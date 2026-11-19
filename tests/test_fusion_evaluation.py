from src.fusion_evaluation import evaluate_fusion_baselines


def _sample(as_of, target_date, predicted, actual, llm_stance, news_score):
    return {
        "schema_version": 1,
        "ticker": "TEST.NS",
        "as_of": as_of,
        "horizon": 1,
        "target_date": target_date,
        "current_close": 100.0,
        "lstm_predicted_close": predicted,
        "actual_close": actual,
        "news_sentiment_score": news_score,
        "llm_stance": llm_stance,
        "llm_evidence_ids": ["chunk-1"],
        "llm_model_id": "llm-v1",
        "lstm_model_id": "lstm-v1",
        "evidence": [{
            "evidence_id": "chunk-1",
            "published_at": "2025-12-31T12:00:00+00:00",
            "observed_at": "2025-12-31T13:00:00+00:00",
        }],
    }


def test_fixed_fusion_baselines_are_reported_chronologically_per_ticker_horizon():
    records = [
        _sample("2026-01-02T00:00:00+00:00", "2026-01-05", 98., 90., "bearish", -1.),
        _sample("2026-01-01T00:00:00+00:00", "2026-01-02", 102., 110., "bearish", 1.),
    ]

    result = evaluate_fusion_baselines(records)

    group = result["per_ticker_horizon"][0]
    assert result["evaluation_scope"] == "chronological_fixed_rule_baselines"
    assert group["ticker"] == "TEST.NS"
    assert group["horizon"] == 1
    assert group["as_of_start"] == "2026-01-01T00:00:00+00:00"
    assert group["sample_count"] == 2
    assert group["persistence_mae"] == 10.0
    assert group["lstm_mae"] == 8.0
    assert group["lstm_directional_accuracy"] == 1.0
    assert group["news_combined_directional_accuracy"] == 1.0
    assert group["llm_stance_directional_accuracy"] == 0.5
    assert group["llm_combined_directional_accuracy"] == 0.5
    assert result["macro_by_horizon"][1]["llm_combined_directional_accuracy"] == 0.5
    assert "not out-of-sample proof" in result["limitation"]
