#!/usr/bin/env python3
"""Run the sequencer and then train models for all available datasets.
This script is safe to launch in background from the Streamlit UI.
It writes a small log file `logs/run_sequencer_train.log` with stdout/stderr.
"""
import os
import sys
import subprocess
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

os.makedirs('logs', exist_ok=True)
log_path = os.path.join('logs', 'run_sequencer_train.log')

with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n=== run_sequencer_train started at {datetime.utcnow().isoformat()} UTC ===\n")
    try:
        seq_cmd = [sys.executable, os.path.join('src', 'data_sequencer.py')]
        f.write(f"Running: {' '.join(seq_cmd)}\n")
        subprocess.run(seq_cmd, check=True, stdout=f, stderr=f)

        train_cmd = [sys.executable, os.path.join('src', 'train_model.py')]
        f.write(f"Running: {' '.join(train_cmd)}\n")
        subprocess.run(train_cmd, check=True, stdout=f, stderr=f)

        f.write(f"Completed at {datetime.utcnow().isoformat()} UTC\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
        raise
