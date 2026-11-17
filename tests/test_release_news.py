from datetime import datetime, timezone

from src.news import fetcher


def test_company_news_uses_business_names_and_keeps_short_alias_context():
    assert fetcher.is_finance_relevant({"title": "HCLTech announces earnings growth"}, "HCLTECH.NS")
    assert fetcher.is_finance_relevant({"title": "Sun Pharma reports quarterly profit"}, "SUNPHARMA.NS")
    assert fetcher.is_finance_relevant({"title": "LIC insurance premium increases"}, "LICI.NS")
    assert not fetcher.is_finance_relevant({"title": "Local lic collection opens"}, "LICI.NS")


def test_training_news_excludes_undated_and_duplicate_stories(monkeypatch):
    def story(slug, stamp):
        return fetcher.normalize_article({"title": f"Example earnings {slug}",
                "url": f"https://reuters.com/{slug}", "source": "Reuters",
                "published_at": stamp}, "EXAMPLE.NS")
    dated = story("known", datetime.now(timezone.utc).isoformat())
    unknown = story("unknown", None)
    monkeypatch.setattr(fetcher, "_newsapi_articles", lambda *a, **kw: [dated, unknown])
    monkeypatch.setattr(fetcher, "_yahoo_articles", lambda *a, **kw: [dated])
    monkeypatch.setattr(fetcher, "_rss_articles", lambda *a, **kw: [])
    monkeypatch.setattr(fetcher, "_google_news_articles", lambda *a, **kw: [])
    result = fetcher.fetch_ticker_articles("EXAMPLE.NS", max_articles=20, force_refresh=True, require_live=True)
    assert len(result) == 1
    assert result[0]["published_at"] is not None
