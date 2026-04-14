#!/usr/bin/env python3
"""Convenience script to run end-to-end quick_run for NSEI (NIFTY 50).

This calls `scripts/quick_run.py --ticker NSEI` which:
 - preprocesses data/NSEI.csv -> data/processed/NSEI_processed.csv
 - sequences -> data/train_data/*.npy and scaler
 - trains model -> models/lstm_NSEI.h5
 - saves prediction plot to images/

Usage:
  python scripts/train_nifty.py --epochs 3 --horizon 1
"""
import os
import sys
import subprocess
import argparse

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--horizon', type=int, default=1)
args = p.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable
runner = os.path.join('scripts', 'quick_run.py')
cmd = [PY, runner, '--ticker', 'NSEI', '--epochs', str(args.epochs), '--horizon', str(args.horizon)]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=False)
