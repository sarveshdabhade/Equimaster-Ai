import os
import glob
import joblib
import subprocess
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(ROOT, "data", "train_data")
MODELS_DIR = os.path.join(ROOT, "models")


def find_tickers():
    # find scalers for tickers
    scalers = glob.glob(os.path.join(TRAIN_DIR, "scaler_*.pkl"))
    tickers = [os.path.basename(s).replace("scaler_", "").replace('.pkl','') for s in scalers]
    return tickers


def resolve_model_path(ticker):
    candidates = [
        os.path.join(MODELS_DIR, f'lstm_{ticker}.keras'),
        os.path.join(MODELS_DIR, f'lstm_{ticker}.h5'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def model_input_features(model_path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        shape = model.input_shape
        # expecting shape (None, time_steps, features)
        if isinstance(shape, tuple) and len(shape) >= 3:
            return shape[2]
        return None
    except Exception:
        return None


def scaler_features(scaler_path):
    try:
        s = joblib.load(scaler_path)
        return getattr(s, 'n_features_in_', None)
    except Exception:
        return None


def retrain_ticker(ticker, epochs=3):
    cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'quick_run.py'), '--ticker', ticker, '--epochs', str(epochs)]
    print("Retraining:", ticker, "cmd:", ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Retrain failed for", ticker, e)
        return False


def main(epochs=3):
    tickers = find_tickers()
    print(f"Found {len(tickers)} tickers with scalers to verify.")
    to_retrain = []
    for t in tickers:
        scaler_p = os.path.join(TRAIN_DIR, f'scaler_{t}.pkl')
        model_p = resolve_model_path(t)
        sf = scaler_features(scaler_p)
        mf = model_input_features(model_p) if os.path.exists(model_p) else None
        if mf is None or sf is None or mf != sf:
            to_retrain.append((t, mf, sf))

    print(f"Tickers to retrain: {len(to_retrain)}")
    for t,mf,sf in to_retrain:
        print(f"-> {t}: model_features={mf} scaler_features={sf}")
        ok = retrain_ticker(t, epochs=epochs)
        if not ok:
            print("Failed to retrain", t)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    try:
        main(epochs=args.epochs)
    except Exception:
        traceback.print_exc()
