"""Tests isolate caches and never depend on model downloads or provider availability."""
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["EQUIMASTER_FINBERT"] = "0"
os.environ["EQUIMASTER_EMBEDDINGS"] = "0"
import pytest


@pytest.fixture(autouse=True)
def isolate_news_cache(tmp_path, monkeypatch):
    import src.news.fetcher as fetcher
    monkeypatch.setattr(fetcher, "NEWS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetcher, "NEWS_ARCHIVE_DIR", tmp_path / "archive")


@pytest.fixture
def multipage_app(monkeypatch):
    """Exercise subpages under the real root entrypoint and page registry."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import views.home_view as home
    monkeypatch.setattr(home, "fetch_index_snapshot", lambda: {})
    monkeypatch.setattr(home, "render_market_movers", lambda *args, **kwargs: None)
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
