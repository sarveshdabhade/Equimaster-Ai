import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

TICKER = os.environ.get('TICKER', 'TCS.NS')
PROCESSED = f"data/processed/{TICKER}_processed.csv"
SCALER = f"data/train_data/scaler_{TICKER}.pkl"
MODEL = f"models/lstm_{TICKER}.h5"

print('Testing prediction for', TICKER)
if not os.path.exists(PROCESSED):
    raise SystemExit('Processed CSV not found: ' + PROCESSED)
if not os.path.exists(SCALER):
    raise SystemExit('Scaler not found: ' + SCALER)
if not os.path.exists(MODEL):
    raise SystemExit('Model not found: ' + MODEL)

# Load
print('Loading resources...')
df = pd.read_csv(PROCESSED, index_col=0, parse_dates=True)
print('Processed rows:', len(df))
scaler = joblib.load(SCALER)
model = tf.keras.models.load_model(MODEL)

# prepare
recent = df.tail(60).values
print('recent shape', recent.shape)
scaled = scaler.transform(recent)
print('scaled shape', scaled.shape)
X = np.array([scaled])
print('X shape', X.shape)

pred = model.predict(X)
print('pred shape', pred.shape)

n_features = getattr(scaler, 'n_features_in_', None)
print('scaler n_features_in_', n_features)
if n_features is None:
    raise SystemExit('Scaler missing n_features_in_ attribute')

CLOSE_COL_IDX = 1
dummy = np.zeros((1, n_features))
dummy[:, CLOSE_COL_IDX] = pred.flatten()[0]
inv_pred = scaler.inverse_transform(dummy)[0][CLOSE_COL_IDX]
print('Predicted price (INR):', inv_pred)
