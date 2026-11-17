"""Start the point-in-time archive from the current local snapshots, without backdating them."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_DOWNLOAD_TICKERS
from src.fundamentals import read_cached_details
from src.fundamental_history import archive_snapshot


if __name__ == "__main__":
    observed_at = datetime.now(timezone.utc)
    archived = []
    missing = []
    for ticker in ALL_DOWNLOAD_TICKERS:
        details = read_cached_details(ticker)
        if details and archive_snapshot(details, observed_at=observed_at):
            archived.append(ticker)
        else:
            missing.append(ticker)
    print(f"Archived {len(archived)} current snapshots at {observed_at.isoformat()}.")
    if missing:
        print(f"No current cached snapshot: {', '.join(missing)}")
