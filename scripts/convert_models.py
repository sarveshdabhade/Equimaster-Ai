"""Convert legacy Keras HDF5 models (.h5) to native Keras format (.keras).

Usage:
    python scripts/convert_models.py [--dry-run]

This will scan the `models/` directory for `.h5` files and write `.keras`
files alongside them. If a `.keras` already exists for a given model the
conversion is skipped.
"""
import os
import argparse
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')


def convert(h5_path, dry_run=False):
    base = os.path.basename(h5_path)
    name = os.path.splitext(base)[0]
    out_path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(out_path):
        print(f"Skipping (already converted): {out_path}")
        return
    print(f"Converting {h5_path} -> {out_path}")
    if dry_run:
        return
    try:
        model = tf.keras.models.load_model(h5_path)
        model.save(out_path)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to convert {h5_path}: {e}")


def main(dry_run=False):
    if not os.path.exists(MODELS_DIR):
        print("No models directory found.")
        return
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
    if not files:
        print("No .h5 files to convert.")
        return
    for f in files:
        convert(os.path.join(MODELS_DIR, f), dry_run=dry_run)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='List conversions without writing files')
    args = p.parse_args()
    main(dry_run=args.dry_run)
