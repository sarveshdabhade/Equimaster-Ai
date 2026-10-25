"""Behavior checks for coverage filters and the redesigned stock workspace."""
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.dashboard_data import apply_screen

ROOT = Path(__file__).resolve().parents[1]


def test_screener_filters_positive_pe_and_excludes_financial_leverage_rules():
    rows = pd.DataFrame([
        {"Symbol": "A", "Company": "Alpha", "Sector": "IT", "P/E": 12., "ROE %": 20., "Debt/equity": .2, "Financial sector": False, "1M %": 2.},
        {"Symbol": "B", "Company": "Bank", "Sector": "Finance", "P/E": -5., "ROE %": 10., "Debt/equity": .1, "Financial sector": True, "1M %": -2.},
        {"Symbol": "C", "Company": "Gamma", "Sector": "IT", "P/E": 30., "ROE %": 5., "Debt/equity": 2., "Financial sector": False, "1M %": 3.},
    ])
    assert apply_screen(rows, preset="Positive P/E under 20")["Symbol"].tolist() == ["A"]
    assert apply_screen(rows, preset="Debt/equity under 0.5x")["Symbol"].tolist() == ["A"]
    assert apply_screen(rows, sector="IT", query="gamma")["Symbol"].tolist() == ["C"]
    assert len(apply_screen(rows, query="[")) == 0  # Literal search, not an invalid regex.


def test_company_route_has_no_watchlist_controls(monkeypatch, multipage_app):
    import utils
    import tabs.overview as overview
    import tabs.news_sentinel as news
    dates = pd.bdate_range("2025-01-01", periods=300)
    frame = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000}, index=dates)
    monkeypatch.setattr(utils, "load_data", lambda *args: frame)
    monkeypatch.setattr(utils, "fetch_stock_snapshot", lambda *args: None)
    monkeypatch.setattr(overview, "render_overview_tab", lambda *args: None)
    calls = []
    monkeypatch.setattr(news, "render_news_tab", lambda *args: calls.append(args))
    app = multipage_app.switch_page("pages/stock_analyzer.py")
    app.query_params["ticker"] = "TCS.NS"
    app.run()
    assert not app.exception
    assert app.session_state["selected_ticker"] == "TCS.NS"
    assert calls == []  # Inactive research sections do not fetch/render.
    assert not any("watchlist" in button.key.lower() for button in app.button if button.key)
    app.radio(key="research_section").set_value("News Sentiment").run()
    assert len(calls) == 1


def test_unknown_ticker_cannot_become_a_data_path(multipage_app):
    app = multipage_app.switch_page("pages/stock_analyzer.py")
    app.query_params["ticker"] = "../../uncovered"
    app.run()
    assert not app.exception
    assert any("outside the current" in warning.value for warning in app.warning)
    assert any(field.key == "research_company_search" for field in app.text_input)
    assert any(field.key == "research_sector_filter" for field in app.selectbox)


def test_market_news_preview_opens_the_full_news_section(monkeypatch, multipage_app):
    import views.home_view as home

    articles = [
        {
            "title": f"Market update {index}",
            "url": f"https://example.com/news/{index}",
            "source": "Example News",
            "published_at": "2026-01-01T12:00:00+00:00",
        }
        for index in range(20)
    ]
    requested_limits = []

    def fake_cached_news(limit):
        requested_limits.append(limit)
        return articles[:limit]

    monkeypatch.setattr(home, "cached_news", fake_cached_news)
    app = multipage_app.run()

    assert not app.exception
    assert requested_limits[-1] == 5
    assert sum('class="news-row"' in item.value for item in app.markdown) == 5

    app.button(key="open_latest_news").click().run()
    assert not app.exception
    assert app.session_state["home_section"] == "Latest news"
    assert requested_limits[-1] == 20
    assert sum('class="news-row"' in item.value for item in app.markdown) == 20
