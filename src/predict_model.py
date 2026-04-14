import os
import sys

# Fix Windows console encoding so emoji/Unicode print correctly
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# 1. Config
TRAIN_DIR = "data/train_data"
MODELS_DIR = "models"
IMG_DIR = "images"  # We will save the graph here
os.makedirs(IMG_DIR, exist_ok=True)

# Close is index 1 in scaled data (matches data_sequencer.py: Adj Close=0, Close=1, ...)
CLOSE_COL_IDX = 1


def predict_and_plot():
    print("Loading resources...")

    # A. Load the Model
    model = tf.keras.models.load_model(f"{MODELS_DIR}/lstm_reliance.h5")

    # B. Load the Scaler (Crucial to convert 0-1 back to Rupees)
    scaler = joblib.load(f"{TRAIN_DIR}/scaler_reliance.pkl")

    # C. Load Test Data
    X_test = np.load(f"{TRAIN_DIR}/X_test.npy")
    y_test = np.load(f"{TRAIN_DIR}/y_test.npy")

    print(f"Making predictions on {len(X_test)} days of data...")
    predictions = model.predict(X_test)

    # --- THE TRICKY PART: INVERSE SCALING ---
    # The scaler expects the same number of columns as in training (e.g. 10).
    # Our prediction is just 1 column (Close). We create a dummy array and
    # put our values in the Close column, then inverse_transform.

    n_features = scaler.n_features_in_

    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_actual = np.zeros((len(y_test), n_features))

    dummy_pred[:, CLOSE_COL_IDX] = predictions.flatten()
    dummy_actual[:, CLOSE_COL_IDX] = y_test.flatten()

    inv_pred = scaler.inverse_transform(dummy_pred)[:, CLOSE_COL_IDX]
    inv_actual = scaler.inverse_transform(dummy_actual)[:, CLOSE_COL_IDX]

    # --- PLOTTING ---
    print("Generating Chart...")
    plt.figure(figsize=(14, 6))
    plt.plot(inv_actual, color="blue", label="Actual Price")
    plt.plot(inv_pred, color="red", label="AI Prediction")
    plt.title("Reliance Stock Price Prediction (LSTM Test Results)")
    plt.xlabel("Time (Days in Test Set)")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_path = f"{IMG_DIR}/prediction_result.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

    # Close figure so script exits when run from terminal (plt.show() would block)
    plt.close()


if __name__ == "__main__":
    predict_and_plot()
