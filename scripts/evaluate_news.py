"""Run sentiment/retrieval evaluation against a labeled JSON corpus."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.news.evaluation import evaluate_news_dataset
from src.artifacts import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "tests" / "fixtures" / "news_evaluation.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "evaluation" / "news.json"))
    args = parser.parse_args()
    result = evaluate_news_dataset(json.loads(Path(args.dataset).read_text(encoding="utf-8")))
    atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
