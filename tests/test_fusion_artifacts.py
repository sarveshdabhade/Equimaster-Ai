from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.fusion_artifacts import load_fusion_artifact, save_fusion_artifact
from src.fusion_model import FEATURE_NAMES, predict_fused_forecast
from src.fusion_split import chronological_purged_split


def _records():
    rows = []
    for day in range(20):
        origin = date(2026, 1, 1) + timedelta(days=day)
        rows.append({
            "schema_version": 1,
            "ticker": "TEST.NS",
            "as_of": f"{origin.isoformat()}T16:00:00+00:00",
            "horizon": 1,
            "target_date": (origin + timedelta(days=1)).isoformat(),
            "current_close": 100.,
            "lstm_predicted_close": 101.,
            "actual_close": 102.,
            "news_sentiment_score": 0.2,
            "llm_stance": "bullish",
            "llm_evidence_ids": ["chunk-1"],
            "llm_model_id": "llm-v1",
            "lstm_model_id": "lstm-v1",
            "evidence": [{
                "evidence_id": "chunk-1",
                "published_at": "2025-12-30T00:00:00+00:00",
                "observed_at": "2025-12-30T01:00:00+00:00",
            }],
        })
    return rows


def _model_result(train_count):
    features = np.arange(train_count * len(FEATURE_NAMES), dtype=float).reshape(
        train_count, len(FEATURE_NAMES)
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features, np.ones(train_count))
    return {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "alpha": 1.0,
        "training_sample_count": train_count,
        "validation": {"sample_count": 2, "macro_average": {"fused_mae": 1.0}},
        "test": {"sample_count": 2, "macro_average": {"fused_mae": 1.1}},
    }


def test_fusion_artifact_round_trip_records_provenance_and_hash(tmp_path):
    split = chronological_purged_split(_records())
    model_result = _model_result(len(split["train"]))
    bundle = save_fusion_artifact(model_result, split, artifact_root=tmp_path)

    loaded = load_fusion_artifact(bundle.name, artifact_root=tmp_path)

    assert loaded["metadata"]["split"]["split_method"] == "global_chronological_target_date_purged"
    assert loaded["metadata"]["sample_counts"]["train"] == len(split["train"])
    assert loaded["metadata"]["model_ids_by_split"]["train"]["TEST.NS"] == {
        "llm": ["llm-v1"], "lstm": ["lstm-v1"]
    }
    features = np.zeros((1, len(FEATURE_NAMES)))
    np.testing.assert_allclose(
        loaded["model"].predict(features), model_result["model"].predict(features)
    )


def test_fusion_artifact_rejects_corruption_and_unsafe_version(tmp_path):
    split = chronological_purged_split(_records())
    bundle = save_fusion_artifact(
        _model_result(len(split["train"])), split, artifact_root=tmp_path
    )
    (bundle / "fusion_model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_fusion_artifact(bundle.name, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="single directory name"):
        load_fusion_artifact("..", artifact_root=tmp_path)


def test_fusion_inference_checks_cutoff_and_training_model_provenance(tmp_path):
    split = chronological_purged_split(_records())
    bundle = save_fusion_artifact(
        _model_result(len(split["train"])), split, artifact_root=tmp_path
    )
    artifact = load_fusion_artifact(bundle.name, artifact_root=tmp_path)
    request = {
        "ticker": "TEST.NS",
        "as_of": "2026-01-21T16:00:00+00:00",
        "horizon": 1,
        "current_close": 100.0,
        "lstm_predicted_close": 101.0,
        "news_sentiment_score": 0.2,
        "llm_stance": "bullish",
        "llm_evidence_ids": ["chunk-1"],
        "llm_model_id": "llm-v1",
        "lstm_model_id": "lstm-v1",
        "evidence": [{
            "evidence_id": "chunk-1",
            "published_at": "2026-01-21T10:00:00+00:00",
            "observed_at": "2026-01-21T11:00:00+00:00",
        }],
    }

    result = predict_fused_forecast(artifact, request)

    assert result["status"] == "experimental_not_activated"
    assert result["fused_predicted_close"] > 0
    assert result["fusion_model_version"] == bundle.name
    late_evidence = {**request, "evidence": [{
        **request["evidence"][0], "observed_at": "2026-01-22T00:00:00+00:00"
    }]}
    with pytest.raises(ValueError, match="after the requested forecast origin"):
        predict_fused_forecast(artifact, late_evidence)
    unknown_model = {**request, "llm_model_id": "unseen-model"}
    with pytest.raises(ValueError, match="not present in fusion training"):
        predict_fused_forecast(artifact, unknown_model)
