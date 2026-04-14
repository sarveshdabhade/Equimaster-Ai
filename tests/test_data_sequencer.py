import numpy as np
from src.data_sequencer import create_sequences, create_multi_step_sequences


def test_create_sequences():
    # 50 rows, 2 features
    data = np.arange(100).reshape(50, 2)
    X, y = create_sequences(data, window_size=5, close_col_idx=1)
    assert X.shape[1:] == (5, 2)
    assert y.ndim == 1
    assert len(X) == len(y)


def test_create_multi_step_sequences():
    # 100 rows, 2 features
    data = np.arange(200).reshape(100, 2)
    X, Y = create_multi_step_sequences(data, window_size=4, horizon=3, close_col_idx=1)
    assert X.shape[1:] == (4, 2)
    assert Y.shape[1] == 3
    assert len(X) == len(Y)
