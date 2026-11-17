import json
from datetime import datetime, timedelta, timezone
import pytest

import src.news.fetcher as news_fetcher
ORIGINAL_RSS_ARTICLES = news_fetcher._rss_articles
ORIGINAL_GOOGLE_NEWS_ARTICLES = news_fetcher._google_news_articles
from src.news.fetcher import (
    canonicalize_article_url,
    deduplicate_articles,
    fetch_ticker_articles,
    is_finance_relevant,
    load_archived_articles,
    normalize_article,
)
from src.news.sentiment import analyze_ticker_sentiment
from src.news.sentiment import _keyword_score
from src.market_context import build_market_context
from src.prediction_contract import PredictionRecord, classify_signal, directional_accuracy
from src.evaluate_predictions import evaluate_predictions


@pytest.fixture(autouse=True)
def trusted_news_fixture(monkeypatch):
    def fake_newsapi_articles(ticker, max_articles):
        headlines = (
            "earnings outlook improves",
            "factory expansion boosts demand",
            "analyst upgrade lifts shares",
        )
        return [
            normalize_article({
                "title": f"{ticker} {headlines[index % len(headlines)]}",
                "summary": "Analysts review revenue and profit growth.",
                "content": "Investors assess the latest earnings outlook.",
                "source": "Reuters",
                "published_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "url": f"https://www.reuters.com/markets/{ticker.lower()}-{index}",
            }, ticker)
            for index in range(max_articles)
        ]

    monkeypatch.setattr(news_fetcher, "_newsapi_articles", fake_newsapi_articles)
    monkeypatch.setattr(news_fetcher, "_yahoo_articles", lambda ticker, max_articles: [])
    monkeypatch.setattr(news_fetcher, "_rss_articles", lambda ticker, max_articles: [])
    monkeypatch.setattr(news_fetcher, "_google_news_articles", lambda ticker, max_articles: [])


def test_news_fetcher_returns_normalized_articles():
    articles = fetch_ticker_articles("TESTTICKER", max_articles=3)
    assert len(articles) == 3
    assert all(article["title"] and article["source"] for article in articles)
    assert all(article["ticker"] == "TESTTICKER" and article["text"] for article in articles)


def test_news_fetcher_supports_force_refresh_for_latest_articles():
    articles = fetch_ticker_articles("TESTTICKER", max_articles=2, force_refresh=True)
    assert len(articles) == 2
    assert all(article["title"] and article["source"] for article in articles)
    assert all(article["published_at"] for article in articles)
    assert all(article["observed_at"] for article in articles)


def test_news_fetcher_merges_every_live_source(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    articles_by_source = {
        "newsapi": normalize_article({
            "title": "UNLISTEDCO earnings outlook improves",
            "source": "Reuters", "published_at": now,
            "url": "https://www.reuters.com/markets/unlistedco-newsapi",
        }, "UNLISTEDCO.NS"),
        "yahoo": normalize_article({
            "title": "UNLISTEDCO shares rise after results",
            "source": "Yahoo Finance", "published_at": now,
            "url": "https://finance.yahoo.com/news/unlistedco-yahoo",
        }, "UNLISTEDCO.NS"),
        "rss": normalize_article({
            "title": "UNLISTEDCO reports revenue growth",
            "source": "Economic Times", "published_at": now,
            "url": "https://economictimes.indiatimes.com/markets/unlistedco-rss",
        }, "UNLISTEDCO.NS"),
    }
    calls = []
    monkeypatch.setattr(
        news_fetcher, "_newsapi_articles",
        lambda ticker, limit: calls.append("newsapi") or [articles_by_source["newsapi"]],
    )
    monkeypatch.setattr(
        news_fetcher, "_yahoo_articles",
        lambda ticker, limit: calls.append("yahoo") or [articles_by_source["yahoo"]],
    )
    monkeypatch.setattr(
        news_fetcher, "_rss_articles",
        lambda ticker, limit: calls.append("rss") or [articles_by_source["rss"]],
    )
    monkeypatch.setattr(news_fetcher, "NEWS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(news_fetcher, "NEWS_ARCHIVE_DIR", tmp_path / "archive")

    articles = fetch_ticker_articles("UNLISTEDCO.NS", force_refresh=True)

    assert calls == ["newsapi", "yahoo", "rss"]
    assert {article["source"] for article in articles} == {
        "Reuters", "Yahoo Finance", "Economic Times",
    }


def test_rss_fetcher_reads_all_configured_publisher_feeds(monkeypatch):
    import xml.etree.ElementTree as ET
    from email.utils import format_datetime

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(news_fetcher, "_RSS_FEED_CACHE", {})
    monkeypatch.setattr(news_fetcher, "_rss_articles", ORIGINAL_RSS_ARTICLES)
    calls = []

    class Response:
        content = b""

        def __init__(self, source):
            root = ET.Element("rss")
            channel = ET.SubElement(root, "channel")
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = f"UNLISTEDCO earnings update from {source}"
            ET.SubElement(item, "description").text = "Revenue and profit growth."
            ET.SubElement(item, "pubDate").text = format_datetime(now)
            ET.SubElement(item, "link").text = {
                "Economic Times": "https://economictimes.indiatimes.com/story",
                "CNBC": "https://www.cnbc.com/story",
                "Business Standard": "https://www.business-standard.com/story",
                "LiveMint": "https://www.livemint.com/story",
                "Hindu BusinessLine": "https://www.thehindubusinessline.com/story",
            }[source]
            self.content = ET.tostring(root)

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        source = next(name for name, feed_url in news_fetcher._RSS_FEEDS.items() if feed_url == url)
        calls.append(source)
        return Response(source)

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)

    articles = news_fetcher._rss_articles("UNLISTEDCO.NS", 10)

    assert set(calls) == set(news_fetcher._RSS_FEEDS)
    assert {article["source"] for article in articles} == set(news_fetcher._RSS_FEEDS)


def test_google_news_rss_requires_allowlisted_publisher_attribution(monkeypatch):
    import xml.etree.ElementTree as ET
    from email.utils import format_datetime

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(news_fetcher, "_GOOGLE_NEWS_CACHE", {})
    monkeypatch.setattr(news_fetcher, "_google_news_articles", ORIGINAL_GOOGLE_NEWS_ARTICLES)

    class Response:
        def __init__(self):
            root = ET.Element("rss")
            channel = ET.SubElement(root, "channel")
            for title, source_name, source_url in (
                ("UNLISTEDCO reports earnings growth", "Business Standard", "https://www.business-standard.com"),
                ("UNLISTEDCO expands port operations", "India Shipping News", "https://indiashippingnews.com"),
                ("UNLISTEDCO shares gain after results", "The Economic Times", "https://m.economictimes.com"),
                ("UNLISTEDCO rumored takeover", "Unknown Blog", "https://unknown.example"),
            ):
                item = ET.SubElement(channel, "item")
                ET.SubElement(item, "title").text = title
                ET.SubElement(item, "description").text = "Revenue and profit update."
                ET.SubElement(item, "pubDate").text = format_datetime(now)
                ET.SubElement(item, "source", {"url": source_url}).text = source_name
                ET.SubElement(item, "link").text = f"https://news.google.com/rss/articles/{source_name.replace(' ', '-')}"
            self.content = ET.tostring(root)

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_fetcher.requests, "get", lambda *args, **kwargs: Response())

    articles = news_fetcher._google_news_articles("UNLISTEDCO.NS", 10)

    assert {article["source"] for article in articles} == {
        "Business Standard", "India Shipping News", "The Economic Times",
    }
    shipping_article = next(article for article in articles if article["source"] == "India Shipping News")
    assert shipping_article["publisher_url"] == "https://indiashippingnews.com"
    assert shipping_article["url"].startswith("https://news.google.com/")


def test_google_news_query_uses_ticker_and_company_aliases():
    query = news_fetcher._google_news_query("ADANIPORTS.NS")

    assert '"ADANIPORTS stock"' in query
    assert '"Adani Ports"' in query
    assert '"Adani Ports and Special Economic Zone"' in query
    assert f'when:{news_fetcher.NEWS_RECENCY_DAYS}d' in query


def test_all_configured_nifty_stocks_have_searchable_company_aliases():
    from config import NIFTY_50_TICKERS

    assert len(NIFTY_50_TICKERS) == 50
    assert len(set(NIFTY_50_TICKERS)) == 50
    assert all(news_fetcher._company_name_aliases(ticker) for ticker in NIFTY_50_TICKERS)
    assert '"Royal Enfield"' in news_fetcher._google_news_query("EICHERMOT.NS")


def test_news_fetcher_preserves_missing_publication_time_and_archives_observation(monkeypatch, tmp_path):
    article = normalize_article({
        "title": "TESTTICKER company update", "summary": "Company market update",
        "source": "Reuters", "url": "https://www.reuters.com/markets/test",
    }, "TESTTICKER")
    monkeypatch.setattr(news_fetcher, "_newsapi_articles", lambda *args: [article])
    monkeypatch.setattr(news_fetcher, "NEWS_ARCHIVE_DIR", tmp_path / "archive")
    articles = fetch_ticker_articles("TESTTICKER", force_refresh=True)

    assert articles[0]["published_at"] is None
    assert articles[0]["observed_at"]
    archived = list((tmp_path / "archive" / "TESTTICKER").glob("*.json"))
    assert len(archived) == 1
    archived_article = json.loads(archived[0].read_text())[0]
    assert archived_article["published_at"] is None
    assert archived_article["observed_at"] == articles[0]["observed_at"]


def test_archived_news_loader_enforces_publication_and_observation_cutoffs():
    archive = news_fetcher.NEWS_ARCHIVE_DIR / "TESTTICKER"
    archive.mkdir(parents=True)
    records = [
        {
            "title": "TESTTICKER earnings growth",
            "summary": "Profit increased.",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/test-allowed",
            "published_at": "2026-01-01T10:00:00+00:00",
            "observed_at": "2026-01-01T11:00:00+00:00",
        },
        {
            "title": "TESTTICKER cutoff-time earnings",
            "summary": "Profit increased at the cutoff.",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/test-cutoff",
            "published_at": "2026-01-02T00:00:00+00:00",
            "observed_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "title": "TESTTICKER earnings future",
            "summary": "Profit expected to increase.",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/test-future",
            "published_at": "2026-01-03T10:00:00+00:00",
            "observed_at": "2026-01-01T11:00:00+00:00",
        },
        {
            "title": "TESTTICKER profit update",
            "summary": "Revenue increased.",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/test-late",
            "published_at": "2026-01-01T10:00:00+00:00",
            "observed_at": "2026-01-02T01:00:00+00:00",
        },
        {
            "title": "TESTTICKER earnings without observation time",
            "summary": "Profit increased.",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/test-missing",
            "published_at": "2026-01-01T10:00:00+00:00",
        },
    ]
    (archive / "20260102T000000000000Z.json").write_text(json.dumps(records), encoding="utf-8")

    articles = load_archived_articles(
        "TESTTICKER", datetime(2026, 1, 2, tzinfo=timezone.utc)
    )

    assert {article["url"] for article in articles} == {
        "https://www.reuters.com/markets/test-allowed",
        "https://www.reuters.com/markets/test-cutoff",
    }


def test_normalizer_standardizes_timestamps_without_inventing_missing_dates():
    article = normalize_article({
        "title": "TESTTICKER earnings", "published_at": "2026-10-24T10:00:00+05:30",
        "observed_at": "Sat, 24 Oct 2026 04:35:00 +0000",
    }, "TESTTICKER")
    assert article["published_at"] == "2026-10-24T04:30:00+00:00"
    assert article["observed_at"] == "2026-10-24T04:35:00+00:00"
    assert normalize_article({"title": "TESTTICKER earnings"}, "TESTTICKER")["published_at"] is None


def test_canonical_url_and_near_duplicate_merging_preserve_provenance():
    first_url = "https://www.reuters.com/story/?utm_source=newsletter&id=4#details"
    assert canonicalize_article_url(first_url) == "https://reuters.com/story?id=4"
    articles = [
        {"ticker": "TESTTICKER", "title": "Acme shares rise after earnings beat",
         "source": "Reuters", "url": first_url, "published_at": "2026-10-24"},
        {"ticker": "TESTTICKER", "title": "Acme shares rise after earnings beats",
         "source": "Bloomberg", "url": "https://bloomberg.com/story/1",
         "published_at": "2026-10-24"},
    ]
    result = deduplicate_articles(articles)
    assert len(result) == 1
    assert set(result[0]["duplicate_sources"]) == {"Reuters", "Bloomberg"}
    assert set(result[0]["duplicate_urls"]) == {first_url, "https://bloomberg.com/story/1"}


def test_article_chunks_keep_provenance_and_stable_ids():
    from src.news.chunker import chunk_article

    base = {
        "ticker": "TESTTICKER", "title": "TESTTICKER reports earnings",
        "source": "Reuters", "published_at": "2026-10-24T04:30:00+00:00",
        "observed_at": "2026-10-24T04:35:00+00:00",
        "duplicate_sources": ["Reuters", "Bloomberg"],
        "duplicate_urls": ["https://reuters.com/story/earnings"],
        "summary": "Revenue rose during the quarter.",
    }
    first = chunk_article({**base, "url": "https://www.reuters.com/story/earnings?utm_source=feed"})
    second = chunk_article({**base, "url": "https://reuters.com/story/earnings"})

    assert first[0]["chunk_id"] == second[0]["chunk_id"]
    assert first[0]["article_id"] == second[0]["article_id"]
    assert first[0]["ticker"] == "TESTTICKER"
    assert first[0]["source"] == "Reuters"
    assert first[0]["observed_at"] == base["observed_at"]
    assert first[0]["canonical_url"] == "https://reuters.com/story/earnings"
    assert first[0]["duplicate_sources"] == base["duplicate_sources"]


def test_finance_relevance_filter_rejects_unrelated_text():
    unrelated = normalize_article({"title": "Weekend recipe ideas", "summary": "A cooking story", "source": "Blog"}, "TESTTICKER")
    relevant = normalize_article({"title": "TESTTICKER earnings outlook", "summary": "Analysts review revenue", "source": "Finance"}, "TESTTICKER")
    assert unrelated is not None and not is_finance_relevant(unrelated, "TESTTICKER")
    assert relevant is not None and is_finance_relevant(relevant, "TESTTICKER")


def test_company_aliases_match_and_short_symbols_need_financial_context():
    from src.news.fetcher import is_finance_relevant

    assert is_finance_relevant(
        {"text": "Larsen & Toubro wins a major infrastructure contract"}, "LT.NS"
    )
    assert is_finance_relevant(
        {"text": "TCS reports strong quarterly earnings growth"}, "TCS.NS"
    )
    assert not is_finance_relevant(
        {"text": "The protests continued through the city"}, "ITC.NS"
    )
    assert not is_finance_relevant(
        {"text": "HDFC Bank reports earnings"}, "SBIN.NS"
    )


def test_untrusted_articles_are_rejected():
    article = normalize_article({
        "title": "TESTTICKER market update",
        "source": "Local fallback feed",
        "url": "",
    }, "TESTTICKER")
    assert article is not None
    assert not news_fetcher._is_trusted_article(article)


def test_sentiment_result_contains_retrieved_evidence():
    result = analyze_ticker_sentiment("TESTTICKER", max_articles=3)
    assert result["ticker"] == "TESTTICKER"
    assert result["status"] == "available"
    assert result["label"] in {"Bullish", "Bearish", "Neutral", "Mixed", "Risk-off"}
    assert -1.0 <= result["overall_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["evidence"]
    assert all("relevance" in item and "score" in item for item in result["evidence"])
    assert all("sentiment" in item and "confidence" in item for item in result["evidence"])
    assert all("relevance_level" in item and "impact_level" in item for item in result["evidence"])


def test_sentiment_without_eligible_evidence_is_not_available():
    result = analyze_ticker_sentiment("TESTTICKER", articles=[])

    assert result["status"] == "no_evidence"
    assert result["label"] == "Neutral"
    assert result["overall_score"] == 0.0
    assert result["evidence"] == []


def test_stale_published_articles_are_not_reused_from_fresh_file_cache():
    ticker = "TESTTICKER"
    published_at = datetime.now(timezone.utc) - timedelta(days=news_fetcher.NEWS_RECENCY_DAYS + 1)
    article = normalize_article({
        "title": "TESTTICKER earnings growth",
        "source": "Reuters",
        "url": "https://www.reuters.com/markets/test-stale",
        "published_at": published_at.isoformat(),
    }, ticker)
    cache_path = news_fetcher._cache_path_for_ticker(ticker)
    cache_path.write_text(json.dumps([article]), encoding="utf-8")

    assert news_fetcher._read_cache(ticker) == []


def test_forced_training_refresh_does_not_count_cached_news_as_live(monkeypatch, tmp_path):
    ticker = "TESTTICKER"
    monkeypatch.setattr(news_fetcher, "NEWS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(news_fetcher, "NEWS_ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(news_fetcher, "_newsapi_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(news_fetcher, "_yahoo_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(news_fetcher, "_rss_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(news_fetcher, "_google_news_articles", lambda *args, **kwargs: [])
    article = normalize_article({
        "title": "TESTTICKER earnings growth",
        "source": "Reuters",
        "url": "https://www.reuters.com/markets/test-cached",
        "published_at": datetime.now(timezone.utc).isoformat(),
    }, ticker)
    cache_path = news_fetcher._cache_path_for_ticker(ticker)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps([article]), encoding="utf-8")

    articles = fetch_ticker_articles(
        ticker, max_articles=20, force_refresh=True, require_live=True
    )

    assert articles == []


def test_keyword_sentiment_recognizes_common_word_forms_without_substring_matches():
    assert _keyword_score("The company expands operations and revenue grows") > 0
    assert _keyword_score("Shares fall after earnings misses") < 0
    assert _keyword_score("Shares jump as the stock rallies") > 0
    assert _keyword_score("The merger was called off") < 0
    assert _keyword_score("Glossary includes no market direction") == 0


def test_sentiment_ignores_google_news_html_and_publisher_attribution():
    article = {
        "title": "Max Life-HDFC Life Merger Called Off - NDTV Profit",
        "summary": (
            '<a href="https://news.google.com/story">Max Life-HDFC Life Merger Called Off</a> '
            '<font color="#6f6f6f">NDTV Profit</font>'
        ),
        "source": "NDTV Profit",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "url": "https://www.ndtvprofit.com/markets/test-merger",
    }

    result = analyze_ticker_sentiment("HDFCLIFE.NS", articles=[article])

    assert result["label"] in {"Bearish", "Risk-off"}
    assert result["articles"][0]["score"] < 0
    assert "NDTV Profit" not in result["articles"][0]["text"]
    assert "<a " not in result["articles"][0]["text"]


def test_training_news_refresh_requires_twenty_for_each_ticker(monkeypatch, tmp_path):
    import scripts.refresh_all_news as news_refresh

    monkeypatch.setattr(news_refresh, "ROOT", tmp_path)
    monkeypatch.setattr(
        news_refresh,
        "fetch_ticker_articles",
        lambda ticker, **kwargs: [{}] * (20 if ticker == "COMPLETE.NS" else 19),
    )

    results = news_refresh.refresh_news_for_tickers(
        ["COMPLETE.NS", "SHORT.NS"], required_articles=20
    )
    report = json.loads((tmp_path / "logs" / "news_refresh.json").read_text(encoding="utf-8"))

    assert len(results) == 2
    assert report["results"][0]["ok"] is True
    assert report["results"][1]["ok"] is False
    assert report["results"][1]["articles"] == 19


def test_named_query_and_market_context_contract():
    result = analyze_ticker_sentiment("TESTTICKER", max_articles=3, query_type="risk events")
    context = build_market_context("TESTTICKER", horizon=5)
    assert result["evidence"]
    assert context["ticker"] == "TESTTICKER"
    assert "sentiment" in context
    assert "combined_score" in context


def test_prediction_contract_has_shared_score_and_accuracy_rules():
    record = PredictionRecord(
        ticker="TESTTICKER",
        as_of="2026-08-20T00:00:00+00:00",
        horizon=5,
        current_close=100.0,
        predicted_close=105.0,
        predicted_return=0.05,
        lstm_signal=0.5,
        sentiment_score=0.2,
        sentiment_label="Bullish",
        sentiment_confidence=0.8,
        combined_score=0.38,
        combined_label=classify_signal(0.38),
        evidence_count=2,
    )
    assert record.to_dict()["ticker"] == "TESTTICKER"
    assert classify_signal(-0.2) == "Bearish"
    assert directional_accuracy(0.1, 0.04) == 1.0
    assert directional_accuracy(0.1, -0.04) == 0.0


def test_prediction_evaluation_reports_lstm_and_combined_accuracy():
    metrics = evaluate_predictions(
        current_close=[100.0, 100.0, 100.0],
        predicted_close=[105.0, 95.0, 101.0],
        actual_close=[104.0, 102.0, 99.0],
        sentiment_scores=[0.2, 0.8, -0.2],
    )
    assert metrics["sample_count"] == 3
    assert metrics["mae"] > 0
    assert 0.0 <= metrics["lstm_directional_accuracy"] <= 1.0
    assert 0.0 <= metrics["combined_directional_accuracy"] <= 1.0
