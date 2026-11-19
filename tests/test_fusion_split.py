from datetime import date, timedelta

from src.fusion_split import chronological_purged_split


def _row(origin_day, horizon, ticker="TEST.NS"):
    origin = date(2026, 1, 1) + timedelta(days=origin_day)
    target = origin + timedelta(days=horizon)
    return {
        "schema_version": 1,
        "ticker": ticker,
        "as_of": f"{origin.isoformat()}T16:00:00+00:00",
        "horizon": horizon,
        "target_date": target.isoformat(),
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
    }


def test_chronological_purged_split_uses_global_origins_and_purges_overlap():
    records = [
        _row(day, 1) for day in range(20)
    ] + [
        _row(day, 5, "OTHER.NS") for day in range(20)
    ]

    result = chronological_purged_split(records)
    validation_start = result["metadata"]["validation_start"][:10]
    test_start = result["metadata"]["test_start"][:10]

    assert result["train"] and result["validation"] and result["test"]
    assert all(row["as_of"] < result["metadata"]["validation_start"] for row in result["train"])
    assert all(row["target_date"] <= validation_start for row in result["train"])
    assert all(
        result["metadata"]["validation_start"] <= row["as_of"] < result["metadata"]["test_start"]
        for row in result["validation"]
    )
    assert all(row["target_date"] <= test_start for row in result["validation"])
    assert all(row["as_of"] >= result["metadata"]["test_start"] for row in result["test"])
    assert result["metadata"]["purged_rows"]["train"] > 0
    assert result["metadata"]["purged_rows"]["validation"] > 0


def test_chronological_purged_split_rejects_insufficient_origins():
    import pytest

    with pytest.raises(ValueError, match="three distinct forecast origins"):
        chronological_purged_split([_row(0, 1), _row(1, 1)])
