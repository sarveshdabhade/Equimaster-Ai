from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from src.prediction_contract import combined_signal
from src.evaluate_predictions import signal_metrics
from src.news.retriever import NewsRetriever
from src.news.sentiment import analyze_ticker_sentiment
from src.news.fetcher import normalize_article, fetch_ticker_articles
from src.fundamentals import valuation_summary
from src.artifacts import atomic_json, resolve_artifact


def test_combined_signal_matches_evaluation():
    returns = [.01, -.02, .05]
    sentiment = [-.1, .2, -.9]
    actual_returns = [.02, -.01, -.03]
    expected = np.mean([np.sign(combined_signal(r, s)) == np.sign(a) for r, s, a in zip(returns, sentiment, actual_returns)])
    result = signal_metrics([100]*3, [100*(1+r) for r in returns], [100*(1+r) for r in actual_returns], sentiment)
    assert result["combined_directional_accuracy"] == expected


def test_point_in_time_news_excludes_future_and_late_observations():
    articles = []
    for name, published, observed in [("known", "2026-01-01", "2026-01-01"),
                                      ("future", "2026-02-01", None),
                                      ("late", "2026-01-01", "2026-02-01"),
                                      ("undated", None, None)]:
        articles.append({"title": name, "summary": "factory expansion", "published_at": published,
                         "observed_at": observed, "url": f"https://example.test/{name}"})
    hits = NewsRetriever(10).retrieve(articles, "factory", recency_days=None,
                                     as_of=datetime(2026,1,2,tzinfo=timezone.utc))
    assert [hit["title"] for hit in hits] == ["known"]


def test_historical_retrieval_rejects_missing_observation_time():
    article = {"title": "unknown observation", "summary": "factory expansion",
               "published_at": "2026-01-01", "url": "https://example.test/news"}
    hits = NewsRetriever(10).retrieve(
        [article], "factory", recency_days=None,
        as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert hits == []


def test_tfidf_retrieval_preserves_chunk_provenance():
    article = {
        "ticker": "TEST", "title": "TEST reports earnings growth",
        "summary": "Revenue and profit increased.", "source": "Reuters",
        "published_at": "2026-01-01T00:00:00+00:00",
        "observed_at": "2026-01-01T01:00:00+00:00",
        "url": "https://www.reuters.com/story?utm_source=newsletter",
        "duplicate_sources": ["Reuters", "Bloomberg"],
        "duplicate_urls": ["https://www.reuters.com/story?utm_source=newsletter"],
    }
    hits = NewsRetriever(5).retrieve([article], "earnings revenue", recency_days=None)

    assert len(hits) == 1
    assert hits[0]["ticker"] == "TEST"
    assert hits[0]["observed_at"] == article["observed_at"]
    assert hits[0]["canonical_url"] == "https://reuters.com/story"
    assert hits[0]["duplicate_sources"] == article["duplicate_sources"]


def test_ingestion_refreshes_metadata_without_reembedding_existing_chunks(monkeypatch, tmp_path):
    import hashlib
    from src.news.chunker import chunk_article
    from src.news.ingest import index_articles_for_ticker

    article = {
        "ticker": "TEST", "title": "TEST earnings update",
        "summary": "Revenue increased.", "source": "Reuters",
        "published_at": "2026-01-01T00:00:00+00:00",
        "observed_at": "2026-01-02T00:00:00+00:00",
        "url": "https://reuters.com/story",
    }
    chunk = chunk_article(article)[0]
    chunk_text = chunk["chunk_text"]

    class ExistingIndex:
        instance = None

        def __init__(self, persist_dir=None):
            self.ids = [self.compute_id(chunk_text)]
            self.documents = [chunk_text]
            self.metadata = [{"observed_at": "2026-01-01T00:00:00+00:00"}]
            self.saved = 0
            ExistingIndex.instance = self

        @staticmethod
        def compute_id(text):
            return hashlib.sha1(text.encode("utf-8")).hexdigest()

        def load_index(self):
            return True

        def save_index(self):
            self.saved += 1

        def add_documents(self, texts, metadata):
            pytest.fail("unchanged chunks should not be re-embedded")

        def build_index(self, texts, metadata):
            pytest.fail("unchanged chunks should not trigger a full rebuild")

    monkeypatch.setattr("src.news.ingest.EmbeddingRetriever", ExistingIndex)
    index_articles_for_ticker("TEST", [article], tmp_path)
    assert ExistingIndex.instance.metadata[0]["observed_at"] == article["observed_at"]
    assert ExistingIndex.instance.saved == 1


def test_historical_sentiment_requires_archived_evidence():
    with pytest.raises(ValueError, match="archived"):
        analyze_ticker_sentiment("TCS.NS", as_of=datetime.now(timezone.utc))


def test_historical_sentiment_loads_archive_and_never_fetches_live_news(monkeypatch):
    import src.news.sentiment as sentiment

    article = {
        "ticker": "TESTTICKER", "title": "TESTTICKER earnings growth",
        "summary": "Profit and revenue increased.", "content": "Strong earnings growth.",
        "source": "Reuters", "url": "https://www.reuters.com/markets/test",
        "published_at": "2026-01-01T10:00:00+00:00",
        "observed_at": "2026-01-01T11:00:00+00:00",
    }
    monkeypatch.setattr(sentiment, "load_archived_articles", lambda ticker, as_of: [article])
    monkeypatch.setattr(
        sentiment, "fetch_ticker_articles",
        lambda *args, **kwargs: pytest.fail("historical analysis must not fetch live news"),
    )

    result = sentiment.analyze_ticker_sentiment(
        "TESTTICKER", as_of=datetime(2026, 1, 2, tzinfo=timezone.utc), recency_days=None
    )

    assert result["evidence"]
    assert result["evidence"][0]["published_at"] == article["published_at"]


def test_news_sentiment_opt_in_integrates_grounded_llm_synthesis(monkeypatch):
    import json
    import src.news.sentiment as sentiment
    from src.llm.provider import LLMProviderError

    article = {
        "ticker": "TESTTICKER", "title": "TESTTICKER earnings growth",
        "summary": "Profit and revenue increased.", "content": "Strong earnings growth.",
        "source": "Reuters", "url": "https://www.reuters.com/markets/test",
        "published_at": "2026-01-01T10:00:00+00:00",
        "observed_at": "2026-01-01T11:00:00+00:00",
    }

    monkeypatch.setattr(sentiment, "fetch_ticker_articles", lambda *args, **kwargs: [article])
    monkeypatch.setattr(sentiment, "create_llm_provider", lambda: None)
    disabled = sentiment.analyze_ticker_sentiment(
        "TESTTICKER", recency_days=None, include_llm=True
    )
    assert disabled["llm_synthesis_status"] == "disabled"
    assert disabled["llm_synthesis"] is None

    class FakeProvider:
        def generate(self, messages, response_format=None):
            payload = json.loads(messages[1]["content"])
            evidence_id = payload["evidence"][0]["evidence_id"]
            return json.dumps({
                "stance": "bullish",
                "summary": "The source reports earnings growth.",
                "summary_evidence_ids": [evidence_id],
                "claims": [{"text": "Profit increased.", "evidence_ids": [evidence_id]}],
            })

    monkeypatch.setattr(sentiment, "create_llm_provider", lambda: FakeProvider())
    result = sentiment.analyze_ticker_sentiment(
        "TESTTICKER", recency_days=None, include_llm=True
    )

    assert result["llm_synthesis_status"] == "generated"
    assert result["llm_synthesis"]["stance"] == "bullish"
    citation_id = result["llm_synthesis"]["summary_evidence_ids"][0]
    assert result["llm_synthesis"]["citation_sources"][citation_id]["source"] == "Reuters"
    assert isinstance(result["overall_score"], float)

    class FailingProvider:
        def generate(self, messages, response_format=None):
            raise LLMProviderError("provider unavailable")

    monkeypatch.setattr(sentiment, "create_llm_provider", lambda: FailingProvider())
    failed = sentiment.analyze_ticker_sentiment(
        "TESTTICKER", recency_days=None, include_llm=True
    )
    assert failed["llm_synthesis_status"] == "error"
    assert failed["llm_synthesis_error"] == "provider unavailable"
    assert isinstance(failed["overall_score"], float)


def test_finbert_neutral_is_not_overwritten_by_keyword_score():
    article = normalize_article({"title":"TEST profit growth", "source":"Reuters",
                                 "url":"https://reuters.com/test", "published_at":"2026-01-01",
                                 "observed_at":"2026-01-01T00:00:00+00:00"}, "TEST")
    with patch("src.news.sentiment._finbert_score", return_value=0.0):
        result = analyze_ticker_sentiment("TEST", articles=[article], as_of=datetime(2026,1,2,tzinfo=timezone.utc), recency_days=None)
    assert result["overall_score"] == 0
    assert result["articles"][0]["backend"] == "finbert"


def test_provider_failure_preserves_verified_cache(monkeypatch, tmp_path):
    import src.news.fetcher as fetcher
    published_at = datetime.now(timezone.utc).isoformat()
    article = normalize_article({"title":"TEST profit growth", "source":"Reuters",
                                 "url":"https://reuters.com/test", "published_at":published_at}, "TEST")
    path = fetcher._cache_path_for_ticker("TEST")
    path.write_text(json.dumps([article]))
    original = path.read_bytes()
    for provider in ("_newsapi_articles", "_yahoo_articles", "_rss_articles", "_google_news_articles"):
        monkeypatch.setattr(fetcher, provider, lambda *args: [])
    assert fetch_ticker_articles("TEST", force_refresh=True)
    assert path.read_bytes() == original


def test_ticker_relevance_rejects_other_company_and_spoofed_source():
    from src.news.fetcher import is_finance_relevant, _is_trusted_article
    assert not is_finance_relevant({"text":"Infosys reports strong earnings growth"}, "TCS.NS")
    assert is_finance_relevant({"text":"Axis Bank reports strong earnings growth"}, "AXISBANK.NS")
    assert not _is_trusted_article({"source":"Reuters", "url":"https://untrusted.test/news"})


def test_peer_median_uses_same_industry_only():
    frame = pd.DataFrame([{"Ticker":"A", "Industry":"IT", "P/E":10},
                          {"Ticker":"B", "Industry":"IT", "P/E":20},
                          {"Ticker":"C", "Industry":"IT", "P/E":20},
                          {"Ticker":"D", "Industry":"Bank", "P/E":1}])
    assert "50.0% below" in valuation_summary(frame, "A")


def test_bundle_resolver_keeps_model_and_scaler_together(tmp_path, monkeypatch):
    import src.artifacts as artifacts
    monkeypatch.setattr(artifacts, "model_directory", lambda: tmp_path)
    bundle = tmp_path / "versions" / "TEST" / "v1"
    atomic_json(bundle / "metadata.json", {"horizons":{"1":{"model":"h1.keras"}}})
    atomic_json(tmp_path / "active_TEST.json", {"bundle":"versions/TEST/v1"})
    result = resolve_artifact("TEST", 1)
    assert result["model"].parent == result["scaler"].parent
    atomic_json(tmp_path / "active_TEST.json", {"bundle":"../outside"})
    with pytest.raises(ValueError, match="inside"):
        resolve_artifact("TEST", 1)


def test_residual_model_starts_at_persistence():
    from src.train_model import build_model
    model = build_model((60, 3), 5, close_index=1)
    data = np.ones((2, 60, 3), dtype=np.float32)
    data[:, -1, 1] = [0.2, 0.7]
    result = np.asarray(model(data, training=False))
    np.testing.assert_allclose(result, np.tile([[.2],[.7]], (1,5)), atol=1e-6)


def test_news_evaluation_fixture_is_explicitly_synthetic():
    from src.news.evaluation import evaluate_news_dataset
    payload = json.loads((Path(__file__).parent / "fixtures" / "news_evaluation.json").read_text())
    result = evaluate_news_dataset(payload)
    assert result["dataset_kind"] == "synthetic_smoke"
    assert result["retrieval_recall_at_k"] == 1
    assert 0 <= result["accuracy"] <= 1


def test_news_evaluation_reports_article_level_ranking_metrics(monkeypatch):
    from src.news.evaluation import evaluate_news_dataset

    hits = [
        {"url": "https://example.test/irrelevant", "article_id": "irrelevant"},
        {
            "url": "https://www.reuters.com/story?utm_source=feed",
            "article_id": "relevant",
        },
        {
            "url": "https://reuters.com/story",
            "article_id": "relevant",
        },
    ]
    monkeypatch.setattr(
        "src.news.evaluation.NewsRetriever.retrieve",
        lambda *args, **kwargs: hits,
    )
    payload = {
        "sentiment": [{"text": "A board meeting was held.", "label": "neutral"}],
        "retrieval": [{
            "query": "earnings", "as_of": "2026-01-02T00:00:00+00:00",
            "articles": [], "relevant_urls": ["https://reuters.com/story?utm_medium=email"],
            "k": 3,
        }],
    }

    result = evaluate_news_dataset(payload)
    metrics = result["retrieval_per_query"][0]
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == 1
    assert metrics["mrr"] == pytest.approx(1 / 2)
    assert metrics["ndcg_at_k"] == pytest.approx(1 / np.log2(3))
    assert result["retrieval_query_count"] == 1


def test_feed_noise_is_ignored_but_material_corrections_are_not():
    from src.update_data import equivalent_history
    frame = pd.DataFrame({"Close":[2000.,2100.], "Volume":[100000.,120000.]},
                         index=pd.date_range("2026-01-01",periods=2))
    noisy = frame.copy()
    noisy["Close"] += .00025
    assert equivalent_history(frame, noisy)
    corrected = frame.copy()
    corrected.loc[corrected.index[-1],"Close"] += 1
    assert not equivalent_history(frame, corrected)
    assert not equivalent_history(frame, frame.iloc[:1])


def test_unchanged_bundle_is_reused_without_fitting(tmp_path, monkeypatch):
    from src import training, artifacts
    frame = pd.DataFrame({"Close":np.arange(300,dtype=float)+100},
                         index=pd.date_range("2025-01-01",periods=300))
    processed = tmp_path / "TEST_processed.csv"
    frame.to_csv(processed)
    root = tmp_path / "models"
    bundle = root / "versions" / "TEST" / "v1"
    bundle.mkdir(parents=True)
    (bundle / "h1.keras").write_bytes(b"model fixture")
    (bundle / "scaler.pkl").write_bytes(b"scaler fixture")
    (bundle / "processed.csv").write_bytes(processed.read_bytes())
    metadata = {"version":"v1", "pipeline_version":training.PIPELINE_VERSION,
                "data_sha256":artifacts.sha256(processed),
                "scaler_sha256":artifacts.sha256(bundle / "scaler.pkl"),
                "horizons":{"1":{"model":"h1.keras", "model_sha256":artifacts.sha256(bundle / "h1.keras")}}}
    artifacts.atomic_json(bundle / "metadata.json", metadata)
    artifacts.atomic_json(root / "active_TEST.json", {"bundle":"versions/TEST/v1"})
    monkeypatch.setattr(training, "model_directory", lambda: root)
    monkeypatch.setattr(training, "resolve_app_path", lambda *args: str(processed))
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: pytest.fail("Unexpected retraining"))
    assert training.train_ticker("TEST", horizons=(1,), only_missing=True)["version"] == "v1"
    (bundle / "h1.keras").unlink()
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("retraining required")))
    with pytest.raises(RuntimeError, match="retraining required"):
        training.train_ticker(
            "TEST",
            horizons=(1,),
            only_missing=True,
            news_validation={"article_count": 20, "fetched_at": "2026-01-01T00:00:00+00:00"},
        )


def test_train_manager_uses_current_universe_and_checks_bundle_integrity(tmp_path):
    import hashlib
    from config import ALL_DOWNLOAD_TICKERS, HORIZONS, PIPELINE_VERSION
    from scripts.train_manager import active_bundle_matches_data, configured_model_tickers

    assert configured_model_tickers() == list(dict.fromkeys([*ALL_DOWNLOAD_TICKERS, "NSEI"]))

    data_dir = tmp_path / "processed"
    model_root = tmp_path / "models"
    bundle = model_root / "versions" / "TEST.NS" / "v1"
    data_dir.mkdir()
    bundle.mkdir(parents=True)
    source = data_dir / "TEST.NS_processed.csv"
    source.write_bytes(b"processed snapshot")
    (bundle / "processed.csv").write_bytes(source.read_bytes())
    (bundle / "scaler.pkl").write_bytes(b"scaler")
    entries = {}
    for horizon in HORIZONS:
        filename = f"h{horizon}.weights.h5"
        model = bundle / filename
        model.write_bytes(f"weights-{horizon}".encode())
        entries[str(horizon)] = {
            "model": filename,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scaler_sha256": hashlib.sha256((bundle / "scaler.pkl").read_bytes()).hexdigest(),
        "horizons": entries,
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (model_root / "active_TEST.NS.json").write_text(
        json.dumps({"bundle": "versions/TEST.NS/v1"}), encoding="utf-8"
    )

    assert active_bundle_matches_data("TEST.NS", HORIZONS, data_dir, model_root)
    (bundle / "h22.weights.h5").write_bytes(b"corrupted weights")
    assert not active_bundle_matches_data("TEST.NS", HORIZONS, data_dir, model_root)


def test_training_blocks_a_ticker_without_twenty_fresh_news_items(tmp_path, monkeypatch):
    from src import training

    processed = tmp_path / "TEST_processed.csv"
    pd.DataFrame(
        {"Close": np.arange(300, dtype=float) + 100},
        index=pd.date_range("2025-01-01", periods=300),
    ).to_csv(processed)
    monkeypatch.setattr(training, "resolve_app_path", lambda *args: str(processed))
    monkeypatch.setattr(training, "model_directory", lambda: tmp_path / "models")
    monkeypatch.setattr(training, "fetch_ticker_articles", lambda *args, **kwargs: [{}] * 19)

    with pytest.raises(RuntimeError, match="requires 20 fresh verified news articles"):
        training.train_ticker("TEST", horizons=(1,))


def test_training_manager_trains_eligible_stocks_and_blocks_short_news(monkeypatch):
    from scripts import train_manager

    started_training = []
    monkeypatch.setattr(train_manager, "write_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_manager,
        "refresh_news_for_tickers",
        lambda tickers, required_articles: [
            {"ticker": "FIRST.NS", "articles": required_articles, "ok": True, "fetched_at": "now"},
            {"ticker": "SECOND.NS", "articles": required_articles - 1, "ok": False, "fetched_at": "now"},
        ],
    )
    monkeypatch.setattr(
        train_manager,
        "process_and_train_ticker",
        lambda *args, **kwargs: (started_training.append(args[0]) or {"ticker": args[0], "ok": True}),
    )

    with pytest.raises(RuntimeError, match="Training incomplete"):
        train_manager.main(tickers=["FIRST.NS", "SECOND.NS"])

    assert started_training == ["FIRST.NS"]


def test_panel_training_blocks_before_fit_if_any_ticker_lacks_twenty_news(monkeypatch):
    from src import panel_training

    monkeypatch.setattr(
        panel_training,
        "_load_panel",
        lambda tickers: ({"FIRST.NS": object(), "SECOND.NS": object()}, {}),
    )
    monkeypatch.setattr(
        panel_training, "fetch_ticker_articles", lambda *args, **kwargs: [{}] * 19
    )
    monkeypatch.setattr(
        panel_training,
        "_date_boundaries",
        lambda frames: pytest.fail("Training must stop before model fitting"),
    )

    with pytest.raises(RuntimeError, match="Panel training blocked before fitting any stock"):
        panel_training.train_panel_model()


def test_invalid_raw_history_does_not_replace_existing_processed_data(tmp_path, monkeypatch):
    from src import preprocessor
    source = tmp_path / "TEST.csv"
    source.write_text("Date,Close\n2026-01-01,100\n")
    output = tmp_path / "TEST_processed.csv"
    output.write_bytes(b"existing valid processed data")
    monkeypatch.setattr(preprocessor.glob, "glob", lambda *args: [str(source)])
    monkeypatch.setattr(preprocessor, "PROCESSED_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Missing required price columns"):
        preprocessor.process_all_files()
    assert output.read_bytes() == b"existing valid processed data"


def test_recursive_forecast_loads_one_step_weights(monkeypatch):
    import src.forecast as forecast

    frame = pd.DataFrame({
        "Open": np.arange(100, 160, dtype=float),
        "High": np.arange(101, 161, dtype=float),
        "Low": np.arange(99, 159, dtype=float),
        "Close": np.arange(100, 160, dtype=float),
        "Volume": np.full(60, 1000.0),
    })
    built = []

    class Scaler:
        data_range_ = np.array([1.0, 1.0, 1.0, 100.0, 1.0])
        data_min_ = np.array([0.0, 0.0, 0.0, 100.0, 0.0])

        def transform(self, values):
            return np.asarray(values, dtype=float)

    class Model:
        def load_weights(self, _path):
            return None

        def __call__(self, _values, training=False):
            return np.array([[0.1]], dtype=float)

    artifact = {
        "metadata": {"features": list(frame.columns), "window_size": 60,
                     "architecture": "residual_lstm_last_close_calibrated_weights",
                     "forecast_contract": {"target_column": "Close"}},
        "model": "weights.h5", "scaler": "scaler.pkl", "evaluation": {},
    }
    monkeypatch.setattr(forecast, "resolve_artifact", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(forecast.joblib, "load", lambda _path: Scaler())
    monkeypatch.setattr(
        "src.train_model.build_model",
        lambda shape, horizon, **kwargs: (built.append(horizon) or Model()),
    )

    result = forecast.predict_forecast(frame, "TEST.NS", 5, recursive=True)

    assert built == [1]
    assert len(result["prices"]) == 5
    assert np.isfinite(result["prices"]).all()


def test_news_annotation_batch_deduplicates_and_requires_both_timestamps(tmp_path):
    from datetime import timedelta
    from scripts.export_news_annotation_batch import build_annotation_rows

    archive = tmp_path / "news_archive" / "TEST_NS"
    archive.mkdir(parents=True)
    published = "2026-10-25T10:00:00+00:00"
    observed = "2026-10-25T11:00:00+00:00"
    article = {
        "title": "Company reports quarterly results", "summary": "Revenue increased.",
        "source": "Example News", "published_at": published, "observed_at": observed,
        "url": "https://example.com/story?utm_source=feed",
    }
    (archive / "one.json").write_text(json.dumps([article, article, {
        **article, "url": "https://example.com/untimed", "observed_at": None,
    }]), encoding="utf-8")
    until = datetime(2026, 10, 26, tzinfo=timezone.utc)

    rows = build_annotation_rows(tmp_path / "news_archive", ["TEST.NS"], until - timedelta(days=30), until)

    assert len(rows) == 1
    assert rows[0]["sentiment_label"] == ""
    assert rows[0]["annotator_id"] == ""
    assert rows[0]["published_at"] == published
