#!/usr/bin/env python3
"""Train manager: sequence and train per-ticker with progress files and concurrency.

Writes per-ticker progress to `logs/progress_{ticker}.json` that the UI can poll.
"""
import os
import sys
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.data_sequencer import create_sequences, create_multi_step_sequences
from src.train_model import build_model

LOG_DIR = os.path.join(ROOT, 'logs')
TRAIN_DIR = os.path.join(ROOT, 'data', 'train_data')
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def write_progress(ticker, data):
    path = os.path.join(LOG_DIR, f'progress_{ticker}.json')
    tmp = path + '.tmp'
    # use timezone-aware UTC timestamp
    data['last_updated'] = datetime.now(timezone.utc).isoformat()
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def resolve_model_path(ticker, horizon=None):
    suffix = f'_h{horizon}' if horizon and horizon > 1 else ''
    candidates = []
    for ext in ('.keras', '.h5'):
        candidates.append(os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}{ext}'))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def process_and_train_ticker(ticker, horizons, window_size, epochs, only_missing=False):
    # Called in worker process
    try:
        write_progress(ticker, {'status': 'starting', 'step': 'init', 'message': 'Starting'})

        proc_path = os.path.join(PROCESSED_DIR, f'{ticker}_processed.csv')
        if not os.path.exists(proc_path):
            write_progress(ticker, {'status': 'error', 'step': 'load', 'message': 'Processed CSV missing'})
            return {'ticker': ticker, 'ok': False}

        df = pd.read_csv(proc_path, index_col=0)
        if 'Close' not in df.columns:
            write_progress(ticker, {'status': 'error', 'step': 'load', 'message': 'Close column missing'})
            return {'ticker': ticker, 'ok': False}

        close_idx = df.columns.get_loc('Close')
        dataset = df.values

        # scale
        write_progress(ticker, {'status': 'sequencing', 'step': 'scaling', 'message': 'Fitting scaler'})
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(dataset)
        joblib.dump(scaler, os.path.join(TRAIN_DIR, f'scaler_{ticker}.pkl'))

        # single-step
        write_progress(ticker, {'status': 'sequencing', 'step': 'single_step', 'message': 'Creating single-step sequences'})
        X, y = create_sequences(scaled, window_size, close_idx)
        np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}.npy'), X[:int(len(X)*0.8)])
        np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}.npy'), y[:int(len(y)*0.8)])
        np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}.npy'), X[int(len(X)*0.8):])
        np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}.npy'), y[int(len(y)*0.8):])

        # multi-horizon
        for h in horizons:
            if h <= 1:
                continue
            write_progress(ticker, {'status': 'sequencing', 'step': f'h{h}', 'message': f'Creating sequences horizon {h}'})
            Xm, Ym = create_multi_step_sequences(scaled, window_size, h, close_idx)
            if Xm.size == 0 or Ym.size == 0:
                # not enough rows
                continue
            split = int(len(Xm) * 0.8)
            np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}_h{h}.npy'), Xm[:split])
            np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}_h{h}.npy'), Ym[:split])
            np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}_h{h}.npy'), Xm[split:])
            np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}_h{h}.npy'), Ym[split:])

        # training per-horizon
        # 1: single-step
        train_tasks = []
        # determine if we need to train (only_missing check)
        # single-step model path
        msingle = resolve_model_path(ticker)
        if not (only_missing and os.path.exists(msingle)):
            train_tasks.append((1, os.path.join(TRAIN_DIR, f'X_train_{ticker}.npy'), os.path.join(TRAIN_DIR, f'y_train_{ticker}.npy')))

        for h in horizons:
            if h <= 1:
                continue
            mpath = resolve_model_path(ticker, h)
            if only_missing and os.path.exists(mpath):
                continue
            xfile = os.path.join(TRAIN_DIR, f'X_train_{ticker}_h{h}.npy')
            yfile = os.path.join(TRAIN_DIR, f'y_train_{ticker}_h{h}.npy')
            if os.path.exists(xfile) and os.path.exists(yfile):
                train_tasks.append((h, xfile, yfile))

        for (h, xfile, yfile) in train_tasks:
            write_progress(ticker, {'status': 'training', 'step': f'h{h}', 'message': f'Training horizon {h}'})
            X_train = np.load(xfile)
            y_train = np.load(yfile)
            out_size = 1 if y_train.ndim == 1 else y_train.shape[1]
            model = build_model((X_train.shape[1], X_train.shape[2]), output_size=out_size)
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            suffix = f'_h{h}' if h > 1 else ''
            model_path = os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}.keras')
            model.save(model_path)
            write_progress(ticker, {'status': 'trained', 'step': f'h{h}', 'message': f'Trained horizon {h}'})

        write_progress(ticker, {'status': 'done', 'step': 'complete', 'message': 'All done'})
        return {'ticker': ticker, 'ok': True}
    except Exception as e:
        write_progress(ticker, {'status': 'error', 'step': 'exception', 'message': str(e)})
        return {'ticker': ticker, 'ok': False, 'error': str(e)}


def main(workers=2, horizons=[1,5,22,88], window_size=60, epochs=3, only_missing=False):
    # collect tickers
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_processed.csv')]
    tickers = [os.path.basename(f).replace('_processed.csv', '') for f in files]

    # initialize progress files
    for t in tickers:
        write_progress(t, {'status': 'queued', 'step': 'queued', 'message': 'Queued for processing'})

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_and_train_ticker, t, horizons, window_size, epochs, only_missing): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                write_progress(t, {'status': 'error', 'step': 'crash', 'message': str(e)})


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--only-missing', action='store_true')
    p.add_argument('--horizons', default='1,5,22,88')
    args = p.parse_args()
    horizons = [int(x) for x in args.horizons.split(',') if x]
    main(workers=args.workers, horizons=horizons, epochs=args.epochs, only_missing=args.only_missing)
