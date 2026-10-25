"""Train a ticker using the shared versioned pipeline."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training import train_ticker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in sessions: 1=Day, 5=Week, 22=Month, 88=Quarter.")
    args = parser.parse_args()
    train_ticker(args.ticker, (args.horizon,), args.epochs)


if __name__ == "__main__":
    main()
