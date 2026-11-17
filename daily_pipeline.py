"""Scheduled market refresh, training, evaluation and news ingestion."""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import signal
from src.artifacts import atomic_json
from src.pipeline_lock import pipeline_lock

ROOT = Path(__file__).resolve().parent


def run_stage(command, log, timeout):
    """Stop the complete child process tree if a stage times out or is cancelled."""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT,
                               stdout=log, stderr=log, **options)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        process.wait()
        raise


def run_pipeline(args):
    """Persist each completed stage, including failures and interrupted runs."""
    failures, results = [], []
    started_at = datetime.now(timezone.utc).isoformat()
    status_path = ROOT / "logs" / "daily_status.json"

    def save(state, current_step=None):
        atomic_json(status_path, {"as_of": datetime.now(timezone.utc).isoformat(),
                    "started_at": started_at, "state": state, "pid": os.getpid(),
                    "python": sys.executable, "current_step": current_step,
                    "results": results, "failures": failures})

    commands = [("market", ["src/update_data.py", "--skip-downstream"]),
                ("preprocess", ["src/preprocessor.py"])]
    if not args.skip_training:
        training = ["scripts/train_manager.py", "--workers", str(args.workers), "--epochs", str(args.epochs)]
        if not args.force:
            training.append("--only-missing")
        commands += [("training", training), ("evaluation", ["scripts/evaluate_watchlist.py"])]
    else:
        commands += [("news", ["scripts/refresh_all_news.py", "--allow-incomplete"])]
    commands += [("fundamentals", ["scripts/refresh_fundamentals.py", "--only-stale"]),
                 ("fundamental_ml", ["scripts/train_fundamental_model.py"])]
    save("running")
    try:
        with (ROOT / "logs" / "daily_pipeline.log").open("a", encoding="utf-8") as log:
            for name, command in commands:
                if name in ("training", "evaluation") and any(x in failures for x in ("market", "preprocess")):
                    results.append({"step": name, "skipped": True, "reason": "Market refresh or preprocessing failed"})
                    save("running", name)
                    continue
                save("running", name)
                log.write(f"\n{datetime.now(timezone.utc).isoformat()} Starting {name}\n")
                log.flush()
                try:
                    returncode = run_stage(command, log, 10800 if name == "training" else 1800)
                    step_result = {"step": name, "returncode": returncode}
                except subprocess.TimeoutExpired:
                    step_result = {"step": name, "returncode": 124, "error": "Stage exceeded its time limit"}
                if name in ("news", "training"):
                    try:
                        news_path = ROOT / "logs" / ("training_news.json" if name == "training" else "news_refresh.json")
                        news = json.loads(news_path.read_text(encoding="utf-8"))
                        incomplete = [item.get("ticker") for item in news.get("results", []) if not item.get("ok")]
                        step_result.update(news_incomplete_count=len(incomplete), news_incomplete_tickers=incomplete)
                    except (OSError, ValueError, TypeError):
                        pass
                results.append(step_result)
                if step_result["returncode"]:
                    failures.append(name)
                save("running")
                print(f"{name}: {'failed' if step_result['returncode'] else 'complete'}", flush=True)
    except BaseException:
        save("interrupted")
        raise
    save("failed" if failures else "complete")
    if failures:
        raise SystemExit(f"Failed pipeline stages: {failures}. See logs/daily_pipeline.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Retrain even if data hashes match")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    with pipeline_lock(ROOT / "logs" / "daily_pipeline.lock"):
        run_pipeline(args)


if __name__ == "__main__":
    main()
