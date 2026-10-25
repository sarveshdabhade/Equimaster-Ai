"""Manual prediction smoke test; no import-time work."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import pandas as pd
    from src.forecast import predict_forecast
    ticker = os.getenv("TICKER", "TCS.NS")
    frame = pd.read_csv(ROOT / "data" / "processed" / f"{ticker}_processed.csv", index_col=0, parse_dates=True)
    print(predict_forecast(frame, ticker)["prices"])
