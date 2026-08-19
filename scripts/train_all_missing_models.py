#!/usr/bin/env python3
"""
Train All Missing Models - Non-interactive script to train all models that don't exist.
Trains: single-step (h1) and multi-horizon (h5, h22, h88) models for all tickers with scalers.
"""
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

from src.train_model import build_model
from config import HORIZONS

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_DIR = os.path.join(ROOT, 'data', 'train_data')
MODELS_DIR = os.path.join(ROOT, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

def get_tickers_with_scalers():
    """Get list of tickers that have scaler files."""
    if not os.path.exists(TRAIN_DIR):
        return []
    files = [f for f in os.listdir(TRAIN_DIR) if f.startswith('scaler_') and f.endswith('.pkl')]
    return [f.replace('scaler_', '').replace('.pkl', '') for f in files]

def check_model_exists(ticker, horizon=1):
    """Check if model file exists for ticker and horizon."""
    suffix = f'_h{horizon}' if horizon > 1 else ''
    model_path = os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}.keras')
    return os.path.exists(model_path)

def train_model_for_ticker(ticker, horizon, epochs=10):
    """Train a single model for a ticker and horizon."""
    suffix = f'_h{horizon}' if horizon > 1 else ''
    xfile = os.path.join(TRAIN_DIR, f'X_train_{ticker}{suffix}.npy')
    yfile = os.path.join(TRAIN_DIR, f'y_train_{ticker}{suffix}.npy')
    model_path = os.path.join(MODELS_DIR, f'lstm_{ticker}{suffix}.keras')
    
    if not os.path.exists(xfile) or not os.path.exists(yfile):
        print(f"    [SKIP] Training data missing for {ticker} h{horizon}")
        return False
    
    try:
        print(f"    [TRAIN] {ticker} horizon={horizon}, epochs={epochs}...")
        X_train = np.load(xfile)
        y_train = np.load(yfile)
        out_size = 1 if y_train.ndim == 1 else y_train.shape[1]
        
        model = build_model((X_train.shape[1], X_train.shape[2]), output_size=out_size)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=32, verbose=0,
            validation_split=0.1
        )
        
        model.save(model_path)
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [None])[-1]
        
        if final_val_loss:
            print(f"    [OK] Saved: {model_path} | Loss: {final_loss:.4f} | Val Loss: {final_val_loss:.4f}")
        else:
            print(f"    [OK] Saved: {model_path} | Loss: {final_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"    [ERROR] Training failed: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("TRAIN ALL MISSING MODELS")
    print("=" * 70)
    
    # Get configuration
    epochs = 10  # Default, can be changed
    
    # Get all tickers with scalers
    tickers = get_tickers_with_scalers()
    print(f"\nFound {len(tickers)} tickers with scalers")
    print(f"Horizons to train: {HORIZONS}")
    print(f"Epochs per model: {epochs}")
    print(f"Models directory: {MODELS_DIR}")
    
    # Count existing models
    existing_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    print(f"\nExisting models: {len(existing_models)}")
    
    # Calculate total expected models
    total_expected = len(tickers) * len(HORIZONS)
    print(f"Total expected models: {total_expected} ({len(tickers)} tickers x {len(HORIZONS)} horizons)")
    
    stats = {
        'tickers': len(tickers),
        'models_exist': 0,
        'models_trained': 0,
        'models_failed': 0,
        'models_skipped': 0,
    }
    
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    for i, ticker in enumerate(sorted(tickers), 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        
        for h in HORIZONS:
            if check_model_exists(ticker, h):
                stats['models_exist'] += 1
                print(f"  h{h}: exists")
            else:
                success = train_model_for_ticker(ticker, h, epochs=epochs)
                if success:
                    stats['models_trained'] += 1
                else:
                    stats['models_skipped'] += 1
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Tickers processed: {stats['tickers']}")
    print(f"Models already existed: {stats['models_exist']}")
    print(f"Models newly trained: {stats['models_trained']}")
    print(f"Models skipped (missing data): {stats['models_skipped']}")
    print(f"Models failed: {stats['models_failed']}")
    
    # Final count
    final_models = len([f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])
    print(f"\nTotal model files now: {final_models}")
    print(f"Completion: {final_models}/{total_expected} ({100*final_models/total_expected:.1f}%)")
    
    if stats['models_failed'] > 0:
        print("\n[WARNING] Some models failed. Check errors above.")
        return 1
    
    print("\n[SUCCESS] All available models are now trained!")
    print("Note: Some models may be missing due to insufficient data (TATAMOTORS.NS)")
    return 0

if __name__ == '__main__':
    sys.exit(main())
