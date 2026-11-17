"""Scheduled-run safety: concurrency, partial failure and cancellation evidence."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import daily_pipeline
from src.pipeline_lock import pipeline_lock


def test_lock_rejects_another_process_and_releases_after_exit(tmp_path):
    path = tmp_path / "pipeline.lock"
    script = "from pathlib import Path; from src.pipeline_lock import pipeline_lock; import sys\nwith pipeline_lock(Path(sys.argv[1])): print('acquired')"
    with pipeline_lock(path):
        result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "already running" in result.stderr
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "acquired" in result.stdout


def test_failed_market_keeps_training_skipped_but_checks_fundamentals(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_pipeline, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    calls = []

    def run(command, log, timeout):
        calls.append(command[0])
        status = json.loads((tmp_path / "logs/daily_status.json").read_text())
        assert status["state"] == "running"
        assert status["python"] == sys.executable
        return 1 if command[0] == "src/update_data.py" else 0

    monkeypatch.setattr(daily_pipeline, "run_stage", run)
    with pytest.raises(SystemExit, match="market"):
        daily_pipeline.run_pipeline(SimpleNamespace(skip_training=False, workers=1, epochs=60, force=False))
    status = json.loads((tmp_path / "logs/daily_status.json").read_text())
    assert status["state"] == "failed"
    assert status["failures"] == ["market"]
    assert "scripts/train_manager.py" not in calls
    assert "scripts/evaluate_watchlist.py" not in calls
    assert "scripts/refresh_fundamentals.py" in calls


def test_cancelled_run_keeps_completed_stage_results(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_pipeline, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()

    def run(command, log, timeout):
        if command[0] == "src/preprocessor.py":
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(daily_pipeline, "run_stage", run)
    with pytest.raises(KeyboardInterrupt):
        daily_pipeline.run_pipeline(SimpleNamespace(skip_training=True))
    status = json.loads((tmp_path / "logs/daily_status.json").read_text())
    assert status["state"] == "interrupted"
    assert status["results"] == [{"step": "market", "returncode": 0}]
