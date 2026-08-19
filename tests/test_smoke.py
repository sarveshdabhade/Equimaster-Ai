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
