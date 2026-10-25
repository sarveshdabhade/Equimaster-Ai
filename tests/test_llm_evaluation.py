import json

import pytest

from src.llm.evaluation import evaluate_llm_synthesis_dataset


class LabeledOutputProvider:
    def generate(self, messages, response_format=None):
        payload = json.loads(messages[1]["content"])
        if payload["ticker"] == "TEST1":
            output = {
                "stance": "bullish",
                "summary": "The evidence supports growth.",
                "summary_evidence_ids": ["a", "b"],
                "claims": [],
            }
        else:
            output = {
                "stance": "neutral",
                "summary": "The evidence is inconclusive.",
                "summary_evidence_ids": ["c"],
                "claims": [],
            }
        return json.dumps(output)


def _evidence(ticker, chunk_id):
    return {
        "ticker": ticker,
        "chunk_id": chunk_id,
        "title": f"{ticker} report",
        "published_at": "2026-01-01T00:00:00+00:00",
        "observed_at": "2026-01-01T01:00:00+00:00",
        "chunk_text": "Company earnings and revenue update.",
    }


def test_llm_evaluation_reports_stance_and_citation_metrics():
    payload = {
        "kind": "independently_labeled",
        "provenance": "manual review",
        "cases": [
            {
                "ticker": "TEST1",
                "as_of": "2026-01-02T00:00:00+00:00",
                "evidence": [_evidence("TEST1", "a"), _evidence("TEST1", "b")],
                "expected_stance": "bullish",
                "relevant_evidence_ids": ["a"],
            },
            {
                "ticker": "TEST2",
                "as_of": "2026-01-02T00:00:00+00:00",
                "evidence": [_evidence("TEST2", "c"), _evidence("TEST2", "d")],
                "expected_stance": "bearish",
                "relevant_evidence_ids": ["c", "d"],
            },
        ],
    }

    result = evaluate_llm_synthesis_dataset(payload, LabeledOutputProvider())

    assert result["sample_count"] == 2
    assert result["stance_accuracy"] == 0.5
    assert result["citation_precision"] == pytest.approx(2 / 3)
    assert result["citation_recall"] == pytest.approx(2 / 3)
    assert result["citation_f1"] == pytest.approx(2 / 3)
    assert result["per_case"][0]["stance_correct"]
    assert not result["per_case"][1]["stance_correct"]
    assert "does not prove semantic entailment" in result["limitation"]


def test_llm_evaluation_rejects_invalid_gold_ids_before_calling_provider():
    class NoCallProvider:
        def generate(self, *args, **kwargs):
            pytest.fail("invalid labels must be rejected before model invocation")

    payload = {
        "cases": [{
            "ticker": "TEST",
            "as_of": "2026-01-02T00:00:00+00:00",
            "evidence": [_evidence("TEST", "known")],
            "expected_stance": "neutral",
            "relevant_evidence_ids": ["missing"],
        }]
    }
    with pytest.raises(ValueError, match="unknown or invalid"):
        evaluate_llm_synthesis_dataset(payload, NoCallProvider())
