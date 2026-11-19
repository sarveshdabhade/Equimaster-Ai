"""Train the point-in-time 12-month fundamentals return model when ready."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fundamental_ml import train_fundamental_model


if __name__ == "__main__":
    result = train_fundamental_model()
    print({key: result.get(key) for key in (
        "ready", "validation_passed", "samples", "matured_months", "reason",
    )}, flush=True)
