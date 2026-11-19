"""Point-in-time safeguards for daily market bars."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
DAILY_BAR_FINALIZATION_TIME = time(16, 0)


def remove_incomplete_current_daily_bar(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop today's daily bar before the cash session has safely finalized.

    Yahoo may return a partial daily OHLCV row while NSE is open. Such a row
    must not be treated as the completed close for training or evaluation.
    ``now`` is injectable so callers can test this rule deterministically.
    """
    if frame is None or frame.empty:
        return frame

    local_now = datetime.now(MARKET_TIMEZONE) if now is None else now
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=MARKET_TIMEZONE)
    else:
        local_now = local_now.astimezone(MARKET_TIMEZONE)

    latest = pd.Timestamp(frame.index[-1])
    if latest.tzinfo is not None:
        latest = latest.tz_convert(MARKET_TIMEZONE)
    if (latest.date() == local_now.date()
            and local_now.time().replace(tzinfo=None) < DAILY_BAR_FINALIZATION_TIME):
        return frame.iloc[:-1].copy()
    return frame


def merge_history_without_window_loss(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Merge a refreshed rolling provider window without dropping older history.

    The current provider rows win on overlapping dates. Older dates that have
    fallen outside a rolling request such as ``period="5y"`` are retained.
    """
    if current is None or current.empty:
        return current
    if previous is None or previous.empty:
        return current.copy()
    if previous.index.has_duplicates or current.index.has_duplicates:
        raise ValueError("Cannot merge market history with duplicate dates")

    older = previous.loc[previous.index < current.index.min()]
    merged = pd.concat([older, current]).sort_index()
    if merged.index.has_duplicates:
        merged = merged.loc[~merged.index.duplicated(keep="last")]
    return merged
