"""Record data quality and freshness for every configured stock and index."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config import ALL_DOWNLOAD_TICKERS
from src.artifacts import atomic_json
from src.market_data import MARKET_TIMEZONE, DAILY_BAR_FINALIZATION_TIME


def audit_frame(frame):
    errors = []
    if frame.empty:
        return ["Dataset is empty"]
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        errors.append("Dates are duplicated or unsorted")
    if frame.index.isna().any():
        errors.append("Invalid date")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        errors.append("Non-finite numeric data")
    for name in ("Open", "High", "Low", "Close"):
        if name in frame and (frame[name] <= 0).any():
            errors.append(f"Non-positive {name}")
    if "Volume" in frame and (frame["Volume"] < 0).any():
        errors.append("Negative volume")
    if all(name in frame for name in ("Open", "High", "Low", "Close")):
        if ((frame.High < frame[["Open", "Close", "Low"]].max(axis=1)) |
                (frame.Low > frame[["Open", "Close", "High"]].min(axis=1))).any():
            errors.append("Inconsistent OHLC bounds")
    now = datetime.now(MARKET_TIMEZONE)
    latest = pd.Timestamp(frame.index[-1]).date()
    if latest > now.date() or (latest == now.date() and now.time().replace(tzinfo=None) < DAILY_BAR_FINALIZATION_TIME):
        errors.append("Future or incomplete current-session bar")
    return errors


def main():
    tickers = [*ALL_DOWNLOAD_TICKERS, "NSEI"]
    reference = pd.read_csv(ROOT / "data/raw/NSEI.csv", index_col=0, parse_dates=True).index
    reference_latest = reference.max()
    recent = reference[-20:]
    records = []
    for ticker in tickers:
        record = {"ticker": ticker, "errors": []}
        for kind, suffix in (("raw", ""), ("processed", "_processed")):
            try:
                frame = pd.read_csv(ROOT / "data" / kind / f"{ticker}{suffix}.csv", index_col=0, parse_dates=True)
                errors = audit_frame(frame)
                last = frame.index.max()
                if last != reference_latest:
                    errors.append("Latest date differs from provider NIFTY reference")
                missing = recent.difference(frame.index)
                if len(missing):
                    errors.append(f"Missing {len(missing)} recent reference sessions")
                record[kind] = {"rows": len(frame), "last_date": str(last.date()),
                                "missing_recent_sessions": [str(date.date()) for date in missing]}
                record["errors"].extend(f"{kind}: {error}" for error in errors)
            except Exception as exc:
                record["errors"].append(f"{kind}: {type(exc).__name__}: {exc}")
        records.append(record)
    age = (datetime.now(MARKET_TIMEZONE).date() - reference_latest.date()).days
    report = {"as_of": datetime.now(timezone.utc).isoformat(),
              "reference_latest": str(reference_latest.date()), "reference_age_calendar_days": age,
              "freshness_basis": "Provider NIFTY dates, not an independent NSE holiday calendar",
              "requested": len(tickers), "passed": sum(not item["errors"] for item in records),
              "failures": [item for item in records if item["errors"]], "datasets": records}
    if age > 7:
        report["reference_warning"] = "Provider reference is more than seven calendar days old"
    atomic_json(ROOT / "data/evaluation/data_quality.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "datasets"}, indent=2))
    return int(bool(report["failures"]) or age > 7)


if __name__ == "__main__":
    raise SystemExit(main())
