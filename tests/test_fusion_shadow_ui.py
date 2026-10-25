import pandas as pd

from tabs import lstm_model


def test_shadow_forecast_is_opt_in():
    assert lstm_model._run_fusion_shadow(
        pd.DataFrame(), {}, {"legacy": False, "metadata": {"version": "lstm-v1"}},
        "TEST.NS", 1, "",
    ) is None


def test_shadow_forecast_uses_cutoff_evidence_and_does_not_activate(monkeypatch):
    evidence_id = "chunk-1"
    as_of = "2024-01-02T15:30:00+05:30"
    sentiment = {
        "overall_score": 0.25,
        "llm_synthesis_status": "generated",
        "llm_synthesis": {
            "stance": "bullish",
            "summary_evidence_ids": [evidence_id],
            "claims": [],
            "llm_model_id": "llm-v1",
            "citation_sources": {
                evidence_id: {
                    "published_at": as_of,
                    "observed_at": as_of,
                    "title": "Results",
                    "source": "Test",
                },
            },
        },
    }
    captured = {}
    monkeypatch.setattr(lstm_model, "analyze_ticker_sentiment", lambda *a, **k: sentiment)
    monkeypatch.setattr(lstm_model, "load_fusion_artifact", lambda version: {"version": version})

    def predict(artifact, request):
        captured["artifact"] = artifact
        captured["request"] = request
        return {
            "status": "experimental_not_activated",
            "fused_predicted_close": 105.0,
            "fusion_model_version": "fusion-v1",
            "llm_model_id": "llm-v1",
            "lstm_model_id": "lstm-v1",
        }

    monkeypatch.setattr(lstm_model, "predict_fused_forecast", predict)
    frame = pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2024-01-02")])
    result = lstm_model._run_fusion_shadow(
        frame,
        {"prices": [103.0]},
        {"legacy": False, "metadata": {"version": "lstm-v1"}},
        "TEST.NS",
        1,
        "fusion-v1",
    )

    assert result["status"] == "shadow_candidate"
    assert result["inference_status"] == "experimental_not_activated"
    assert captured["artifact"] == {"version": "fusion-v1"}
    request = captured["request"]
    assert request["as_of"] == as_of
    assert request["llm_evidence_ids"] == [evidence_id]
    assert request["lstm_predicted_close"] == 103.0
    assert request["evidence"][0]["evidence_id"] == evidence_id
