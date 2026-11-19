import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_STORAGE_PATH_ENV, resolve_app_path, ensure_runtime_directories

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Config
ensure_runtime_directories()
TRAIN_DIR = resolve_app_path("data", "train_data")
MODELS_DIR = os.getenv(MODEL_STORAGE_PATH_ENV, resolve_app_path("models"))
MODELS_DIR = os.path.abspath(MODELS_DIR)
os.makedirs(MODELS_DIR, exist_ok=True)

def build_model(input_shape, output_size=1, close_index=None, compile_model=True):
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
    model.add(Dense(units=output_size, kernel_initializer="zeros" if close_index is not None else "glorot_uniform"))

    if close_index is not None:
        # Learn a correction to the last close rather than relearning its level.
        inputs = tf.keras.Input(shape=input_shape)
        last_row = tf.keras.layers.Cropping1D((input_shape[0] - 1, 0))(inputs)
        last_row = tf.keras.layers.Flatten()(last_row)
        weights = np.zeros((input_shape[1], output_size), dtype="float32")
        weights[close_index, :] = 1.0
        baseline = tf.keras.layers.Dense(output_size, use_bias=False, trainable=False,
                                         kernel_initializer=tf.keras.initializers.Constant(weights))(last_row)
        model = tf.keras.Model(inputs, tf.keras.layers.Add()([baseline, model(inputs)]))

    # Compile the brain
    if compile_model:
        model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_all():
    from scripts.train_manager import main
    main(epochs=60, only_missing=False)


if __name__ == "__main__":
    train_all()
