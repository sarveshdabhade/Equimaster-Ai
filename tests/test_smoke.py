def test_smoke():
    """Basic smoke test - ensure key modules can be imported."""
    # Test config import
    import sys
    import os
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    from config import NIFTY_50_TICKERS, INDEX_TICKERS
    assert len(NIFTY_50_TICKERS) > 0
    assert "^NSEI" in INDEX_TICKERS.values()


def test_train_model_build():
    """Test that the model can be built with expected shape."""
    import sys
    import os
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    from src.train_model import build_model
    model = build_model((60, 14), output_size=1)
    assert model.input_shape[1:] == (60, 14)


def test_project_paths_are_resolved_from_repo_root():
    """Project paths should resolve from the repository root, not from CWD."""
    import sys
    import os
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)

    from config import get_project_root, ensure_runtime_directories, resolve_app_path

    root = get_project_root()
    assert os.path.isdir(root)

    ensure_runtime_directories()
    models_dir = resolve_app_path("models")
    data_dir = resolve_app_path("data")
    assert os.path.isdir(models_dir)
    assert os.path.isdir(data_dir)


def test_live_index_snapshot_returns_distinct_values_per_index(monkeypatch):
    """Each market index should reflect its own live price, not a repeated value."""
    import sys
    import os
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)

    import utils
    values = {label: 1000.0 + i * 100 for i, label in enumerate(utils.INDEX_TICKERS)}
    monkeypatch.setattr(utils, "_fetch_single_index", lambda item: (item[0], {"last": values[item[0]], "delta": 1., "pct": .1}))
    utils.fetch_index_snapshot.clear()
    snapshot = utils.fetch_index_snapshot()
    assert snapshot

    last_values = [entry["last"] for entry in snapshot.values()]
    assert len(set(round(v, 4) for v in last_values)) == len(last_values)
