import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import sys

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Config
TRAIN_DIR = "data/train_data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_model(input_shape, output_size=1):
    """
    Creates the LSTM Neural Network Architecture.
    """
    # DEBUG: print(f"[*] Building Model with input shape: {input_shape}...")
    
    model = Sequential()
    # Use an explicit Input layer to avoid passing `input_shape` to RNN layers
    model.add(Input(shape=input_shape))
    # Layer 1: LSTM (The Memory Layer)
    # return_sequences=True because we have another LSTM layer after this
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2)) # Prevents overfitting (forgetting random things)

    # Layer 2: LSTM (The Reasoning Layer)
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Layer 3: Dense (The Decision Layer)
    model.add(Dense(units=25))
    
    # Layer 4: Output (The Price or multi-step vector)
    model.add(Dense(units=output_size))

    # Compile the brain
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_all():
    # Find all training files
    # Look for files like "X_train_RELIANCE.NS.npy"
    files = glob.glob(f"{TRAIN_DIR}/X_train_*.npy")
    # DEBUG: print(f"Found {len(files)} datasets to train on...")

    for file_path in files:
        try:
            # Extract ticker from filename "data/train_data\X_train_TCS.NS.npy"
            ticker = file_path.split("X_train_")[-1].replace(".npy", "")

            # DEBUG: print(f"\nTraining for: {ticker}")

            # 1. Load THIS stock's data
            X_train = np.load(f"{TRAIN_DIR}/X_train_{ticker}.npy")
            y_train = np.load(f"{TRAIN_DIR}/y_train_{ticker}.npy")
            X_test = np.load(f"{TRAIN_DIR}/X_test_{ticker}.npy")
            y_test = np.load(f"{TRAIN_DIR}/y_test_{ticker}.npy")

            # 2. Build Fresh Model (output size depends on y_train shape)
            out_size = 1
            if len(y_train.shape) == 1:
                out_size = 1
            else:
                out_size = y_train.shape[1]
            model = build_model((X_train.shape[1], X_train.shape[2]), output_size=out_size)

            # 3. Train with validation split to detect overfitting
            history = model.fit(
                X_train, y_train,
                epochs=10, batch_size=32, verbose=0,
                validation_split=0.1,
            )

            # 4. Save Unique Model (use native Keras format)
            model_path = os.path.join(MODELS_DIR, f"lstm_{ticker}.keras")
            model.save(model_path)
            # DEBUG: print(f"Saved: {os.path.basename(model_path)}")

        except Exception:
            # DEBUG: print(f"Failed {ticker}: {e}")
            pass


if __name__ == "__main__":
    train_all()
