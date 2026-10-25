"""Compatibility command to train NIFTY with the canonical pipeline."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from src.training import train_ticker
    from config import HORIZON_LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1,
                        help=f"Forecast horizon in sessions: {', '.join(f'{days}={label}' for days, label in HORIZON_LABELS.items())}.")
    args = parser.parse_args()
    train_ticker("NSEI", (args.horizon,), args.epochs)
