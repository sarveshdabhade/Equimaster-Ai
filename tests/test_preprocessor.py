import pandas as pd

from src.preprocessor import can_reuse_active_snapshot


def _history():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [11.0, 12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5, 13.5],
            "Volume": [100, 110, 120, 130],
            "RSI": [40.0, 41.0, 42.0, 43.0],
        },
        index=index,
    )


def test_no_new_completed_session_reuses_snapshot_despite_oldest_row_rolloff():
    snapshot = _history()
    candidate = snapshot.iloc[1:].copy()
    candidate.loc[candidate.index[-1], "Close"] += 0.000001
    candidate.loc[candidate.index[-1], "RSI"] += 0.1

    assert can_reuse_active_snapshot(snapshot, candidate)


def test_snapshot_is_not_reused_when_latest_session_advances():
    snapshot = _history()
    candidate = pd.concat(
        [snapshot, pd.DataFrame({"Open": [14.0], "High": [15.0], "Low": [13.0],
                                 "Close": [14.5], "Volume": [140], "RSI": [44.0]},
                                index=[pd.Timestamp("2026-01-05")])]
    )

    assert not can_reuse_active_snapshot(snapshot, candidate)


def test_snapshot_is_not_reused_after_material_price_revision_or_internal_gap():
    snapshot = _history()
    revised = snapshot.copy()
    revised.loc[revised.index[-1], "Close"] += 1.0
    with_gap = snapshot.drop(index=snapshot.index[1])

    assert not can_reuse_active_snapshot(snapshot, revised)
    assert not can_reuse_active_snapshot(snapshot, with_gap)
