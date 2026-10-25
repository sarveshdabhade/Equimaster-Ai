import pytest

from src.fusion_dataset import (
    build_fusion_dataset,
    llm_feature_row_from_sentiment,
    validate_fusion_dataset,
    validate_fusion_record,
)


def _record(**overrides):
    record = {
        "schema_version": 1,
        "ticker": "test.ns",
        "as_of": "2026-01-01T11:00:00+05:30",
        "horizon": 5,
        "target_date": "2026-01-09",
        "current_close": 100.0,
        "lstm_predicted_close": 102.0,
        "actual_close": 101.0,
        "news_sentiment_score": 0.25,
        "llm_stance": "bullish",
        "llm_evidence_ids": ["chunk-1"],
        "llm_model_id": "llm-v1",
        "lstm_model_id": "lstm-v3",
        "evidence": [{
            "evidence_id": "chunk-1",
            "published_at": "2026-01-01T04:00:00+00:00",
            "observed_at": "2026-01-01T04:30:00+00:00",
        }],
    }
    record.update(overrides)
    return record


def test_fusion_record_normalizes_provenance_and_forecast_fields():
    row = validate_fusion_record(_record())

    assert row["ticker"] == "TEST.NS"
    assert row["as_of"] == "2026-01-01T05:30:00+00:00"
    assert row["target_date"] == "2026-01-09"
    assert row["horizon"] == 5
    assert row["evidence"][0]["observed_at"] == "2026-01-01T04:30:00+00:00"
    assert row["llm_evidence_ids"] == ["chunk-1"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon", 7, "horizon must be one of"),
        ("target_date", "2026-01-01", "target_date must be later"),
        ("news_sentiment_score", 1.1, "range -1..1"),
        ("lstm_model_id", "", "required for reproducibility"),
        ("llm_evidence_ids", ["unknown"], "reference supplied evidence"),
    ],
)
def test_fusion_record_rejects_invalid_contract_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_fusion_record(_record(**{field: value}))


@pytest.mark.parametrize("timestamp_field", ["published_at", "observed_at"])
def test_fusion_record_rejects_evidence_after_forecast_origin(timestamp_field):
    evidence = [{
        "evidence_id": "chunk-1",
        "published_at": "2026-01-01T04:00:00+00:00",
        "observed_at": "2026-01-01T04:30:00+00:00",
    }]
    evidence[0][timestamp_field] = "2026-01-01T06:00:00+00:00"
    with pytest.raises(ValueError, match="not available at the forecast origin"):
        validate_fusion_record(_record(evidence=evidence))


def test_fusion_dataset_rejects_duplicate_samples_and_sorts_by_origin():
    later = _record(as_of="2026-01-02T00:00:00+00:00", target_date="2026-01-12")
    earlier = _record(as_of="2026-01-01T12:00:00+00:00")
    ordered = validate_fusion_dataset([later, earlier])
    assert [row["as_of"] for row in ordered] == [
        "2026-01-01T12:00:00+00:00",
        "2026-01-02T00:00:00+00:00",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_fusion_dataset([earlier, earlier])


def test_fusion_dataset_builder_joins_forecasts_to_same_origin_features():
    origin = "2026-01-01T16:00:00+00:00"
    features = [{
        "ticker": "TEST.NS",
        "as_of": origin,
        "news_sentiment_score": 0.25,
        "llm_stance": "bullish",
        "llm_evidence_ids": ["chunk-1"],
        "llm_model_id": "provider-model-v2",
        "evidence": [{
            "evidence_id": "chunk-1",
            "published_at": "2026-01-01T10:00:00+00:00",
            "observed_at": "2026-01-01T11:00:00+00:00",
        }],
    }]
    forecasts = [
        {
            "ticker": "TEST.NS", "as_of": origin, "horizon": horizon,
            "target_date": target_date, "current_close": 100.,
            "predicted_close": predicted, "actual_close": actual,
            "lstm_model_id": "lstm-bundle-v3",
        }
        for horizon, target_date, predicted, actual in (
            (1, "2026-01-02", 101., 100.5),
            (5, "2026-01-08", 103., 102.),
        )
    ]

    rows = build_fusion_dataset(forecasts, features)

    assert [row["horizon"] for row in rows] == [1, 5]
    assert all(row["llm_stance"] == "bullish" for row in rows)
    assert all(row["lstm_model_id"] == "lstm-bundle-v3" for row in rows)
    assert all(row["llm_model_id"] == "provider-model-v2" for row in rows)


def test_fusion_dataset_builder_rejects_missing_origin_features():
    with pytest.raises(ValueError, match="Missing point-in-time LLM features"):
        build_fusion_dataset(
            [{
                "ticker": "TEST.NS", "as_of": "2026-01-01T16:00:00+00:00",
                "horizon": 1, "target_date": "2026-01-02", "current_close": 100.,
                "predicted_close": 101., "actual_close": 100.5, "lstm_model_id": "v1",
            }],
            [],
        )


def test_sentiment_result_converts_to_auditable_llm_feature_row():
    result = {
        "overall_score": 0.3,
        "llm_synthesis_status": "generated",
        "llm_synthesis": {
            "llm_model_id": "provider-model-v2",
            "stance": "bullish",
            "summary_evidence_ids": ["chunk-1"],
            "claims": [{"evidence_ids": ["chunk-2"]}],
            "citation_sources": {
                "chunk-1": {
                    "published_at": "2026-01-01T10:00:00+00:00",
                    "observed_at": "2026-01-01T11:00:00+00:00",
                },
                "chunk-2": {
                    "published_at": "2026-01-01T09:00:00+00:00",
                    "observed_at": "2026-01-01T10:00:00+00:00",
                },
            },
        },
    }

    row = llm_feature_row_from_sentiment(
        "test.ns", "2026-01-01T16:00:00+00:00", result
    )

    assert row["ticker"] == "TEST.NS"
    assert row["llm_model_id"] == "provider-model-v2"
    assert row["llm_evidence_ids"] == ["chunk-1", "chunk-2"]
    assert len(row["evidence"]) == 2
