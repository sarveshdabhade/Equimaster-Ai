"""Normalize CSV date formatting while preserving historical dates."""
from pathlib import Path
import pandas as pd


def normalize_csv_dates(path: Path) -> None:
    df = pd.read_csv(path)
    candidates = [c for c in df.columns if c.lower() in ("date", "timestamp")]
    if not candidates:
        raise ValueError(f"{path}: missing Date or Timestamp column")
    column = candidates[0]
    dates = pd.to_datetime(df[column], errors="raise")
    if dates.isna().any() or dates.dt.normalize().duplicated().any():
        raise ValueError(f"{path}: missing or duplicate dates; restore valid source data")
    df[column] = dates.dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "data" / "processed"
    failures = []
    for path in directory.glob("*_processed.csv"):
        try:
            normalize_csv_dates(path)
            print(f"Normalized {path}")
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
