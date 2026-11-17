from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.market_data import merge_history_without_window_loss, remove_incomplete_current_daily_bar


def test_open_session_daily_bar_is_removed():
    index = pd.to_datetime(["2026-10-23", "2026-10-26"])
    frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    now = datetime(2026, 10, 26, 13, 45, tzinfo=ZoneInfo("Asia/Kolkata"))

    result = remove_incomplete_current_daily_bar(frame, now)

    assert result.index.tolist() == [pd.Timestamp("2026-10-23")]
    assert frame.index.tolist() == index.tolist()


def test_daily_bar_is_kept_after_finalization_cutoff():
    index = pd.to_datetime(["2026-10-23", "2026-10-26"])
    frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    now = datetime(2026, 10, 26, 16, 5, tzinfo=ZoneInfo("Asia/Kolkata"))

    result = remove_incomplete_current_daily_bar(frame, now)

    assert result.index.tolist() == index.tolist()


def test_current_only_incomplete_bar_returns_empty_frame():
    frame = pd.DataFrame({"Close": [101.0]}, index=pd.to_datetime(["2026-10-26"]))
    now = datetime(2026, 10, 26, 13, 45, tzinfo=ZoneInfo("Asia/Kolkata"))

    result = remove_incomplete_current_daily_bar(frame, now)

    assert result.empty
    assert list(result.columns) == ["Close"]


def test_rolling_provider_window_keeps_older_history_and_uses_current_overlap():
    previous = pd.DataFrame(
        {"Close": [10.0, 11.0, 12.0]}, index=pd.to_datetime(["2021-10-25", "2021-10-26", "2021-10-27"])
    )
    current = pd.DataFrame(
        {"Close": [11.5, 13.0]}, index=pd.to_datetime(["2021-10-27", "2021-10-28"])
    )

    result = merge_history_without_window_loss(previous, current)

    assert result.index.tolist() == pd.to_datetime(
        ["2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28"]
    ).tolist()
    assert result["Close"].tolist() == [10.0, 11.0, 11.5, 13.0]
