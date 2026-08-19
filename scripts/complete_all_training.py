#!/usr/bin/env python3
"""
Complete All Training Script
Ensures all tickers have scalers and all models (single-step + multi-horizon) are trained.
Handles missing data gracefully and reports detailed status.
"""
import os
import sys
import json
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from src.data_sequencer import create_sequences, create_multi_step_sequences
from src.train_model import build_model
from config import ALL_DOWNLOAD_TICKERS, HORIZONS

# Directories
LOG_DIR = os.path.join(ROOT, 'logs')
TRAIN_DIR = os.path.join(ROOT, 'data', 'train_data')
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Use a simple in-memory progress tracker to avoid file permission issues
progress_cache = {}

def log_progress(ticker, data):
    """Log progress to console and optionally to file (with fallback)."""
    status = data.get('status', 'unknown')
    step = data.get('step', 'N/A')
    message = data.get('message', '')
    progress_cache[ticker] = {'status': status, 'step': step, 'message': message}
    print(f"[{ticker}] {status.upper()} | Step: {step} | {message}")
    
    # Try to write progress file, but don't fail if permission denied
    try:
        path = os.path.join(LOG_DIR, f'progress_{ticker}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'status': status,
                'step': step,
                'message': message,
                'last_updated': pd.Timestamp.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        print(f"  [Warning] Could not write progress file for {ticker}: {e}")

def check_scaler_exists(ticker):
    """Check if scaler file exists for ticker."""
    scaler_path = os.path.join(TRAIN_DIR, f'scaler_{ticker}.pkl')
    return os.path.exists(scaler_path)

def check_model_exists(ticker, horizon=1):
    """Check if model file exists for ticker and horizon."""
    suffix = f'_h{horizon}' if horizon > 1 else ''
    model_path = os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}.keras')
    return os.path.exists(model_path)

def create_scaler_and_sequences(ticker, window_size=60, horizons=HORIZONS):
    """Create scaler and sequence files for a ticker."""
    log_progress(ticker, {'status': 'sequencing', 'step': 'init', 'message': 'Starting scaler creation'})
    
    proc_path = os.path.join(PROCESSED_DIR, f'{ticker}_processed.csv')
    
    if not os.path.exists(proc_path):
        log_progress(ticker, {'status': 'error', 'step': 'load', 'message': f'Processed CSV missing: {proc_path}'})
        return False
    
    try:
        df = pd.read_csv(proc_path, index_col=0)
        if 'Close' not in df.columns:
            log_progress(ticker, {'status': 'error', 'step': 'load', 'message': 'Close column missing'})
            return False
        
        if len(df) < window_size + max(horizons):
            log_progress(ticker, {'status': 'error', 'step': 'load', 
                                  'message': f'Not enough data: {len(df)} rows, need at least {window_size + max(horizons)}'})
            return False
        
        close_idx = df.columns.get_loc('Close')
        dataset = df.values
        
        # Create scaler
        log_progress(ticker, {'status': 'sequencing', 'step': 'scaling', 'message': 'Fitting scaler'})
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(dataset)
        scaler_path = os.path.join(TRAIN_DIR, f'scaler_{ticker}.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"  [OK] Scaler saved to {scaler_path}")
        
        # Create single-step sequences
        log_progress(ticker, {'status': 'sequencing', 'step': 'single_step', 'message': 'Creating single-step sequences'})
        X, y = create_sequences(scaled, window_size, close_idx)
        split = int(len(X) * 0.8)
        
        np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}.npy'), X[:split])
        np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}.npy'), y[:split])
        np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}.npy'), X[split:])
        np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}.npy'), y[split:])
        print(f"  [OK] Single-step sequences: train={split}, test={len(X)-split}")
        
        # Create multi-horizon sequences
        for h in horizons:
            if h <= 1:
                continue
            log_progress(ticker, {'status': 'sequencing', 'step': f'h{h}', 'message': f'Creating sequences horizon {h}'})
            Xm, Ym = create_multi_step_sequences(scaled, window_size, h, close_idx)
            if Xm.size == 0 or Ym.size == 0:
                print(f"  [SKIP] Not enough data for horizon {h}")
                continue
            split_m = int(len(Xm) * 0.8)
            np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}_h{h}.npy'), Xm[:split_m])
            np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}_h{h}.npy'), Ym[:split_m])
            np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}_h{h}.npy'), Xm[split_m:])
            np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}_h{h}.npy'), Ym[split_m:])
            print(f"  [OK] Horizon {h} sequences: train={split_m}, test={len(Xm)-split_m}")
        
        log_progress(ticker, {'status': 'sequenced', 'step': 'complete', 'message': 'Scaler and sequences created'})
        return True
        
    except Exception as e:
        log_progress(ticker, {'status': 'error', 'step': 'exception', 'message': str(e)})
        print(f"  [ERROR] {traceback.format_exc()}")
        return False

def train_ticker_models(ticker, epochs=10, horizons=HORIZONS, force=False):
    """Train all models for a ticker."""
    log_progress(ticker, {'status': 'training', 'step': 'init', 'message': 'Starting model training'})
    
    train_tasks = []
    
    # Check single-step model
    msingle = os.path.join(MODELS_DIR, f'lstm_{ticker}.keras')
    if force or not os.path.exists(msingle):
        xfile = os.path.join(TRAIN_DIR, f'X_train_{ticker}.npy')
        yfile = os.path.join(TRAIN_DIR, f'y_train_{ticker}.npy')
        if os.path.exists(xfile) and os.path.exists(yfile):
            train_tasks.append((1, xfile, yfile))
        else:
            print(f"  [SKIP] Single-step training files missing for {ticker}")
    else:
        print(f"  [SKIP] Single-step model already exists: {msingle}")
    
    # Check multi-horizon models
    for h in horizons:
        if h <= 1:
            continue
        mpath = os.path.join(MODELS_DIR, f'lstm_{ticker}_h{h}.keras')
        if force or not os.path.exists(mpath):
            xfile = os.path.join(TRAIN_DIR, f'X_train_{ticker}_h{h}.npy')
            yfile = os.path.join(TRAIN_DIR, f'y_train_{ticker}_h{h}.npy')
            if os.path.exists(xfile) and os.path.exists(yfile):
                train_tasks.append((h, xfile, yfile))
            else:
                print(f"  [SKIP] Horizon {h} training files missing for {ticker}")
        else:
            print(f"  [SKIP] Horizon {h} model already exists: {mpath}")
    
    if not train_tasks:
        log_progress(ticker, {'status': 'done', 'step': 'complete', 'message': 'All models already exist'})
        return True
    
    # Train models
    for (h, xfile, yfile) in train_tasks:
        try:
            log_progress(ticker, {'status': 'training', 'step': f'h{h}', 'message': f'Training horizon {h}'})
            X_train = np.load(xfile)
            y_train = np.load(yfile)
            out_size = 1 if y_train.ndim == 1 else y_train.shape[1]
            
            model = build_model((X_train.shape[1], X_train.shape[2]), output_size=out_size)
            
            print(f"  [TRAIN] Horizon {h}: X shape={X_train.shape}, y shape={y_train.shape}, epochs={epochs}")
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
            
            suffix = f'_h{h}' if h > 1 else ''
            model_path = os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}.keras')
            model.save(model_path)
            
            # Log final metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history.get('val_loss', [None])[-1]
            print(f"  [OK] Saved: {model_path} | Loss: {final_loss:.4f} | Val Loss: {final_val_loss:.4f if final_val_loss else 'N/A'}")
            
            log_progress(ticker, {'status': 'trained', 'step': f'h{h}', 
                                  'message': f'Trained horizon {h}, loss={final_loss:.4f}'})
            
        except Exception as e:
            log_progress(ticker, {'status': 'error', 'step': f'h{h}', 'message': str(e)})
            print(f"  [ERROR] Training failed for horizon {h}: {traceback.format_exc()}")
            return False
    
    log_progress(ticker, {'status': 'done', 'step': 'complete', 'message': 'All models trained'})
    return True

def get_all_tickers():
    """Get list of all tickers that have processed CSV files."""
    if not os.path.exists(PROCESSED_DIR):
        return []
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_processed.csv')]
    return [f.replace('_processed.csv', '') for f in files]

def check_missing_tickers():
    """Check which tickers from config are missing processed data."""
    processed_tickers = set(get_all_tickers())
    expected_tickers = set(ALL_DOWNLOAD_TICKERS)
    missing = expected_tickers - processed_tickers
    return missing

def main():
    print("=" * 70)
    print("COMPLETE ALL TRAINING - Ensuring all models are up to date")
    print("=" * 70)
    
    # Configuration
    epochs = int(input("Enter number of epochs (default 10): ") or "10")
    force_retrain = input("Force retrain existing models? (y/N): ").lower() == 'y'
    
    # Get all tickers
    all_tickers = get_all_tickers()
    print(f"\nFound {len(all_tickers)} tickers with processed data")
    
    # Check for missing tickers from config
    missing = check_missing_tickers()
    if missing:
        print(f"\n[WARNING] {len(missing)} tickers from config missing processed data:")
        for t in sorted(missing):
            print(f"  - {t}")
    
    # Statistics
    stats = {
        'total': len(all_tickers),
        'scaler_created': 0,
        'scaler_exists': 0,
        'scaler_failed': 0,
        'models_trained': 0,
        'models_exist': 0,
        'models_failed': 0,
    }
    
    print(f"\nProcessing {len(all_tickers)} tickers...")
    print("-" * 70)
    
    for i, ticker in enumerate(sorted(all_tickers), 1):
        print(f"\n[{i}/{len(all_tickers)}] Processing {ticker}...")
        
        # Step 1: Ensure scaler exists
        if not check_scaler_exists(ticker):
            print(f"  [INFO] Scaler missing for {ticker}, creating...")
            success = create_scaler_and_sequences(ticker)
            if success:
                stats['scaler_created'] += 1
            else:
                stats['scaler_failed'] += 1
                continue
        else:
            print(f"  [OK] Scaler exists for {ticker}")
            stats['scaler_exists'] += 1
        
        # Step 2: Train models
        success = train_ticker_models(ticker, epochs=epochs, force=force_retrain)
        if success:
            # Count how many models were actually trained vs already existed
            # This is approximate - we check at the end
            stats['models_trained'] += 1
        else:
            stats['models_failed'] += 1
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Summary")
    print("=" * 70)
    print(f"Total tickers: {stats['total']}")
    print(f"Scalers created: {stats['scaler_created']}")
    print(f"Scalers already existed: {stats['scaler_exists']}")
    print(f"Scalers failed: {stats['scaler_failed']}")
    print(f"Models trained/exist: {stats['models_trained']}")
    print(f"Models failed: {stats['models_failed']}")
    
    # Final verification - count actual model files
    model_count = len([f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])
    scaler_count = len([f for f in os.listdir(TRAIN_DIR) if f.startswith('scaler_') and f.endswith('.pkl')])
    print(f"\nActual files:")
    print(f"  - Scalers: {scaler_count}")
    print(f"  - Models: {model_count}")
    
    if stats['scaler_failed'] > 0 or stats['models_failed'] > 0:
        print("\n[WARNING] Some tickers failed. Check logs above for details.")
        return 1
    
    print("\n[SUCCESS] All training complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
