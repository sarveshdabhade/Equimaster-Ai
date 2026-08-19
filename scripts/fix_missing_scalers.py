#!/usr/bin/env python3
"""
Fix Missing Scalers - Non-interactive script to create scalers for all tickers.
"""
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.data_sequencer import create_sequences, create_multi_step_sequences
from config import HORIZONS

TRAIN_DIR = os.path.join(ROOT, 'data', 'train_data')
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')

os.makedirs(TRAIN_DIR, exist_ok=True)

def create_scaler_and_sequences(ticker, window_size=60, horizons=HORIZONS):
    """Create scaler and sequence files for a ticker."""
    print(f"\n[Processing {ticker}]...")
    
    proc_path = os.path.join(PROCESSED_DIR, f'{ticker}_processed.csv')
    
    if not os.path.exists(proc_path):
        print(f"  [SKIP] Processed CSV missing: {proc_path}")
        return False
    
    # Check if scaler already exists
    scaler_path = os.path.join(TRAIN_DIR, f'scaler_{ticker}.pkl')
    if os.path.exists(scaler_path):
        print(f"  [OK] Scaler already exists: {scaler_path}")
        return True
    
    try:
        df = pd.read_csv(proc_path, index_col=0)
        if 'Close' not in df.columns:
            print(f"  [ERROR] Close column missing")
            return False
        
        if len(df) < window_size + max(horizons):
            print(f"  [ERROR] Not enough data: {len(df)} rows, need at least {window_size + max(horizons)}")
            return False
        
        close_idx = df.columns.get_loc('Close')
        dataset = df.values
        
        # Create scaler
        print(f"  [INFO] Creating scaler...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(dataset)
        joblib.dump(scaler, scaler_path)
        print(f"  [OK] Scaler saved: {scaler_path}")
        
        # Create single-step sequences
        print(f"  [INFO] Creating single-step sequences...")
        X, y = create_sequences(scaled, window_size, close_idx)
        split = int(len(X) * 0.8)
        
        np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}.npy'), X[:split])
        np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}.npy'), y[:split])
        np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}.npy'), X[split:])
        np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}.npy'), y[split:])
        print(f"  [OK] Single-step: train={split}, test={len(X)-split}")
        
        # Create multi-horizon sequences
        for h in horizons:
            if h <= 1:
                continue
            Xm, Ym = create_multi_step_sequences(scaled, window_size, h, close_idx)
            if Xm.size == 0 or Ym.size == 0:
                print(f"  [SKIP] Horizon {h}: not enough data")
                continue
            split_m = int(len(Xm) * 0.8)
            np.save(os.path.join(TRAIN_DIR, f'X_train_{ticker}_h{h}.npy'), Xm[:split_m])
            np.save(os.path.join(TRAIN_DIR, f'y_train_{ticker}_h{h}.npy'), Ym[:split_m])
            np.save(os.path.join(TRAIN_DIR, f'X_test_{ticker}_h{h}.npy'), Xm[split_m:])
            np.save(os.path.join(TRAIN_DIR, f'y_test_{ticker}_h{h}.npy'), Ym[split_m:])
            print(f"  [OK] Horizon {h}: train={split_m}, test={len(Xm)-split_m}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        print(f"  {traceback.format_exc()}")
        return False

def get_all_tickers():
    """Get list of all tickers that have processed CSV files."""
    if not os.path.exists(PROCESSED_DIR):
        return []
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_processed.csv')]
    return [f.replace('_processed.csv', '') for f in files]

def main():
    print("=" * 70)
    print("FIX MISSING SCALERS - Creating scalers for all tickers")
    print("=" * 70)
    
    all_tickers = get_all_tickers()
    print(f"\nFound {len(all_tickers)} tickers with processed data")
    print(f"Target directory: {TRAIN_DIR}")
    
    # Count existing scalers
    existing_scalers = [f for f in os.listdir(TRAIN_DIR) if f.startswith('scaler_') and f.endswith('.pkl')]
    print(f"Existing scalers: {len(existing_scalers)}")
    
    stats = {'created': 0, 'exists': 0, 'failed': 0}
    
    for i, ticker in enumerate(sorted(all_tickers), 1):
        scaler_path = os.path.join(TRAIN_DIR, f'scaler_{ticker}.pkl')
        if os.path.exists(scaler_path):
            stats['exists'] += 1
            print(f"[{i}/{len(all_tickers)}] {ticker}: scaler exists")
        else:
            success = create_scaler_and_sequences(ticker)
            if success:
                stats['created'] += 1
            else:
                stats['failed'] += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tickers: {len(all_tickers)}")
    print(f"Scalers created: {stats['created']}")
    print(f"Scalers already existed: {stats['exists']}")
    print(f"Scalers failed: {stats['failed']}")
    
    # Final count
    final_scalers = len([f for f in os.listdir(TRAIN_DIR) if f.startswith('scaler_') and f.endswith('.pkl')])
    print(f"\nTotal scaler files now: {final_scalers}")
    
    if stats['failed'] > 0:
        print("\n[WARNING] Some scalers failed to create. Check errors above.")
        return 1
    
    print("\n[SUCCESS] All scalers are now available!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
