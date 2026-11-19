"""Compatibility command: recreate matching versioned models and scalers together."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.train_manager import main

if __name__ == "__main__":
    main(only_missing=True)
