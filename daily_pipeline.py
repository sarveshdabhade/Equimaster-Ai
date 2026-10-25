"""Scheduled market refresh, training, evaluation and news ingestion."""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import sys
from src.artifacts import atomic_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Retrain even if data hashes match")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    failures, results = [], []
    commands = [("market", ["src/update_data.py", "--skip-downstream"]),
                ("preprocess", ["src/preprocessor.py"])]
    if not args.skip_training:
        training = ["scripts/train_manager.py", "--workers", str(args.workers), "--epochs", str(args.epochs)]
        if not args.force:
            training.append("--only-missing")
        commands += [("training", training), ("evaluation", ["scripts/evaluate_watchlist.py"])]
    commands += [("news", ["scripts/refresh_all_news.py"]),
                 ("fundamentals", ["scripts/refresh_fundamentals.py", "--only-stale"])]
    with (ROOT / "logs" / "daily_pipeline.log").open("a", encoding="utf-8") as log:
        for name, command in commands:
            # Preserve the previous active bundle if refresh/preprocessing fails.
            if name in ("training", "evaluation") and any(x in failures for x in ("market", "preprocess", "training")):
                results.append({"step": name, "skipped": True})
                continue
            log.write(f"\n{datetime.now(timezone.utc).isoformat()} Starting {name}\n")
            log.flush()
            result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=log)
            results.append({"step": name, "returncode": result.returncode})
            if result.returncode:
                failures.append(name)
            print(f"{name}: {'failed' if result.returncode else 'complete'}", flush=True)
    atomic_json(ROOT / "logs" / "daily_status.json", {"as_of": datetime.now(timezone.utc).isoformat(),
                "results": results, "failures": failures})
    if failures:
        raise SystemExit(f"Failed pipeline stages: {failures}. See logs/daily_pipeline.log")


if __name__ == "__main__":
    main()
