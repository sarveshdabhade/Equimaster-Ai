"""Streamlit interaction checks with deterministic provider responses."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
INDIA_TZ = ZoneInfo("Asia/Kolkata")


def test_market_session_status_uses_indian_trading_hours():
    from views.home_view import market_session_status

    assert market_session_status(datetime(2026, 10, 26, 9, 15, tzinfo=INDIA_TZ)) == ("OPEN", "market-open")
    assert market_session_status(datetime(2026, 10, 26, 15, 29, tzinfo=INDIA_TZ)) == ("OPEN", "market-open")
    assert market_session_status(datetime(2026, 10, 26, 15, 30, tzinfo=INDIA_TZ)) == ("CLOSED", "market-closed")
    assert market_session_status(datetime(2026, 10, 25, 11, 0, tzinfo=INDIA_TZ)) == ("CLOSED", "market-closed")


def test_home_page_renders_without_providers(monkeypatch):
    import views.home_view as home
    monkeypatch.setattr(home, "fetch_index_snapshot", lambda: {})
    monkeypatch.setattr(home, "render_indices_ticker", lambda *a: None)
    monkeypatch.setattr(home, "render_sensex_ticker", lambda: None)
    monkeypatch.setattr(home, "render_market_movers", lambda *args, **kwargs: None)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any(s.label == "Search a stock" for s in app.text_input)


def test_analyzer_tabs_and_fundamentals_interaction(monkeypatch, multipage_app):
    import utils
    import tabs.lstm_model as forecast
    import tabs.news_sentinel as news
    import tabs.market_overview as market
    import tabs.fundamentals as fundamentals
    monkeypatch.setattr(fundamentals, "read_cached_details", lambda *args: None)
    dates = pd.bdate_range("2024-01-01", periods=400)
    price = np.linspace(100, 200, len(dates))
    frame = pd.DataFrame({"Open":price, "High":price+1, "Low":price-1, "Close":price,
                          "Volume":np.ones(len(dates))*1000}, index=dates)
    monkeypatch.setattr(utils, "load_data", lambda *args: frame)
    monkeypatch.setattr(utils, "fetch_stock_snapshot", lambda *args: None)
    monkeypatch.setattr(forecast, "resolve_artifact", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    sentiment = {
        "overall_score": 0., "confidence": 0., "label": "Neutral", "evidence": [], "drivers": [],
        "llm_synthesis_status": "generated",
        "llm_synthesis": {
            "stance": "neutral", "summary": "The report provides a neutral update.",
            "summary_evidence_ids": ["chunk-1"],
            "claims": [{"text": "The report describes an update.", "evidence_ids": ["chunk-1"]}],
            "citation_sources": {
                "chunk-1": {
                    "title": "Company update", "source": "Reuters",
                    "url": "https://www.reuters.com/markets/test",
                }
            },
        },
    }
    context = {"sentiment":sentiment, "combined_score":0., "technical_signal":0., "label":"Neutral"}
    monkeypatch.setattr(news, "build_market_context", lambda *a, **k: context)
    monkeypatch.setattr(news, "analyze_ticker_sentiment", lambda *a, **k: sentiment)
    monkeypatch.setattr(market, "build_market_context", lambda *a, **k: context)
    monkeypatch.setattr(fundamentals, "compare_peers", lambda *a, **k: (pd.DataFrame([
        {"Ticker":"TCS.NS", "Industry":"IT", "P/E":20.}]), []))
    app = multipage_app
    app.session_state["selected_ticker"] = "TCS.NS"
    app.switch_page("pages/stock_analyzer.py")
    app.run()
    assert not app.exception
    assert {"Overview", "Technical Chart", "LSTM Model", "News Sentiment", "Fundamentals", "Market Summary"}.issubset(set(app.radio(key="research_section").options))
    app.radio(key="research_section").set_value("Technical Chart").run()
    assert not app.exception
    assert any(slider.label == "Chart date range" for slider in app.slider)
    app.radio(key="research_section").set_value("Fundamentals").run()
    assert not app.exception
    refresh = next(b for b in app.button if b.label == "Refresh Company Data")
    refresh.click().run()
    assert not app.exception
    assert any(b.label == "Export Peer Comparison" for b in app.get("download_button"))
    app.radio(key="research_section").set_value("News Sentiment").run()
    assert not app.exception
    assert any("Sentiment is unavailable" in item.value for item in app.info)
    assert any(metric.label == "News sentiment" and metric.value == "Unavailable" for metric in app.metric)
    assert any(metric.label == "Sentiment score" and metric.value == "N/A" for metric in app.metric)
    assert any("LLM evidence synthesis" in item.value for item in app.markdown)
