from src.news.fetcher import fetch_ticker_articles
from src.news.sentiment import analyze_ticker_sentiment


def test_fetch_ticker_articles_returns_list():
    articles = fetch_ticker_articles("RELIANCE", max_articles=3)
    assert isinstance(articles, list)
    assert len(articles) > 0
    for article in articles:
        assert "title" in article
        assert "source" in article


def test_analyze_ticker_sentiment_returns_structured_result():
    result = analyze_ticker_sentiment("RELIANCE", max_articles=3)
    assert result["ticker"] == "RELIANCE"
    assert result["label"] in {"Bullish", "Bearish", "Neutral"}
    assert -1.0 <= result["overall_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["summary"], str) and len(result["summary"]) > 0
    assert isinstance(result["evidence"], list)
    assert isinstance(result["articles"], list)
