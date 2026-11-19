"""Pooled, leakage-aware direct LSTM training for the configured equity panel."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from config import HORIZONS, NEWS_MAX_ARTICLES, WINDOW_SIZE, resolve_app_path
from src.artifacts import atomic_json, model_directory, sha256
from src.lstm_features import FEATURE_COLUMNS, build_lstm_features
from src.news.fetcher import fetch_ticker_articles

logger = logging.getLogger(__name__)
PIPELINE_VERSION = 1
MODEL_NAME = "PANEL_LSTM"
MODEL_DIR = model_directory()
ACTIVE_PATH = MODEL_DIR / f"active_{MODEL_NAME}.json"


def _load_panel(tickers: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    processed_dir = Path(resolve_app_path("data", "processed"))
    paths = sorted(processed_dir.glob("*_processed.csv"))
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        paths = [path for path in paths if path.name.removesuffix("_processed.csv").upper() in wanted]
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        ticker = path.name.removesuffix("_processed.csv").upper()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        frame.index = pd.to_datetime(frame.index, errors="raise").tz_localize(None)
        if frame.index.has_duplicates or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            raise ValueError(f"{ticker} has duplicate dates or non-finite processed values")
        engineered = build_lstm_features(frame)
        data = engineered.copy()
        data["__close"] = pd.to_numeric(frame["Close"], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURE_COLUMNS, "__close"])
        data = data[data["__close"] > 0]
        if len(data) <= WINDOW_SIZE + max(HORIZONS):
            logger.warning("Skipping %s: insufficient usable LSTM feature history", ticker)
            continue
        frames[ticker] = data
        hashes[ticker] = digest
    if len(frames) < 2:
        raise ValueError(f"Pooled LSTM training requires at least two usable tickers; found {len(frames)}")
    return frames, hashes


def _date_boundaries(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in frames.values()))))
    test_index = int(len(dates) * 0.80)
    calibration_index = int(test_index * 0.85)
    if calibration_index <= WINDOW_SIZE or test_index - calibration_index < max(HORIZONS) or len(dates) - test_index < max(HORIZONS) + 2:
        raise ValueError("Panel needs more dated history for fit, calibration and purged holdout windows")
    return dates[0], dates[calibration_index - 1], dates[calibration_index], dates[test_index]


def _make_samples(frames: dict[str, pd.DataFrame], horizon: int,
                  calibration_start: pd.Timestamp, test_start: pd.Timestamp):
    x_fit, y_fit = [], []
    x_cal, y_cal, meta_cal = [], [], []
    x_test, y_test, meta_test = [], [], []
    for ticker, frame in frames.items():
        features = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float32")
        close = frame["__close"].to_numpy(dtype=np.float64)
        dates = frame.index
        origins = np.arange(WINDOW_SIZE - 1, len(frame) - horizon, dtype=np.int64)
        if not len(origins):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            features, window_shape=WINDOW_SIZE, axis=0
        ).transpose(0, 2, 1)
        future = np.lib.stride_tricks.sliding_window_view(close[1:], window_shape=horizon, axis=0)
        actual_paths = future[origins]
        valid = (close[origins] > 0) & np.isfinite(actual_paths).all(axis=1) & (actual_paths > 0).all(axis=1)
        origins = origins[valid]
        actual_paths = actual_paths[valid]
        returns = np.log(actual_paths / close[origins, None]).astype("float32")
        starts = dates[origins + 1]
        ends = dates[origins + horizon]
        fit_mask = ends < calibration_start
        cal_mask = (starts >= calibration_start) & (ends < test_start)
        test_mask = starts >= test_start
        window_indices = origins - WINDOW_SIZE + 1
        if fit_mask.any():
            x_fit.append(windows[window_indices[fit_mask]])
            y_fit.append(returns[fit_mask])
        for mask, x_parts, y_parts, metadata in (
            (cal_mask, x_cal, y_cal, meta_cal),
            (test_mask, x_test, y_test, meta_test),
        ):
            if not mask.any():
                continue
            selected = origins[mask]
            x_parts.append(windows[window_indices[mask]])
            y_parts.append(returns[mask])
            metadata.extend({
                "ticker": ticker,
                "as_of": dates[index],
                "target_date": dates[index + horizon],
                "current_close": float(close[index]),
                "actual_path": actual_paths[position].astype("float64"),
                "actual_return_path": returns[mask][position].astype("float64"),
            } for position, index in enumerate(selected))
    def pack(xs, ys, metas):
        if not xs:
            return (np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)), dtype="float32"),
                    np.empty((0, horizon), dtype="float32"), [])
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), metas
    return pack(x_fit, y_fit, []), pack(x_cal, y_cal, meta_cal), pack(x_test, y_test, meta_test)


def _build_panel_lstm(horizon: int, compile_model: bool = True) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, len(FEATURE_COLUMNS)), name="technical_sequence")
    hidden = tf.keras.layers.LSTM(32, kernel_regularizer=regularizer, recurrent_regularizer=regularizer)(inputs)
    hidden = tf.keras.layers.Dropout(0.20)(hidden)
    hidden = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizer)(hidden)
    outputs = tf.keras.layers.Dense(horizon, kernel_initializer="zeros", name="standardized_log_return_path")(hidden)
    model = tf.keras.Model(inputs, outputs, name=f"pooled_lstm_{horizon}")
    if compile_model:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                     loss=tf.keras.losses.Huber(delta=1.0))
    return model


def build_panel_inference_model(horizon: int) -> tf.keras.Model:
    """Recreate the architecture for fast loading of weights-only artifacts."""
    return _build_panel_lstm(horizon, compile_model=False)


def _load_existing(hashes: dict[str, str]) -> dict[str, Any] | None:
    if not ACTIVE_PATH.exists():
        return None
    try:
        pointer = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
        bundle = (MODEL_DIR / pointer["bundle"]).resolve()
        if not bundle.is_relative_to(MODEL_DIR.resolve()):
            return None
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        return metadata if metadata.get("data_hashes") == hashes and all(
            str(horizon) in metadata.get("horizons", {})
            and (bundle / metadata["horizons"][str(horizon)]["model"]).is_file()
            for horizon in HORIZONS
        ) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def train_panel_model(epochs: int = 60, window_size: int = WINDOW_SIZE,
                      only_missing: bool = False, tickers: list[str] | None = None) -> dict[str, Any]:
    """Train one shared LSTM per horizon and atomically activate the panel bundle."""
    if window_size != WINDOW_SIZE:
        raise ValueError(f"The pooled feature contract currently requires window_size={WINDOW_SIZE}")
    frames, hashes = _load_panel(tickers)
    if only_missing:
        existing = _load_existing(hashes)
        if existing:
            return existing

    training_news: dict[str, dict[str, Any]] = {}
    insufficient_news = []
    for ticker in sorted(frames):
        articles = fetch_ticker_articles(
            ticker,
            max_articles=NEWS_MAX_ARTICLES,
            force_refresh=True,
            require_live=True,
        )
        training_news[ticker] = {
            "article_count": len(articles),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(articles) < NEWS_MAX_ARTICLES:
            insufficient_news.append(
                {"ticker": ticker, "articles": len(articles)}
            )
    if insufficient_news:
        raise RuntimeError(
            f"Panel training blocked before fitting any stock: every ticker requires "
            f"{NEWS_MAX_ARTICLES} fresh verified articles. "
            f"Insufficient news: {insufficient_news}"
        )

    data_start, fit_end, calibration_start, test_start = _date_boundaries(frames)
    feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
    fit_feature_rows = np.concatenate([
        frame.loc[frame.index < calibration_start, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        for frame in frames.values()
    ])
    feature_scaler.fit(fit_feature_rows)
    for ticker, frame in frames.items():
        frame.loc[:, list(FEATURE_COLUMNS)] = feature_scaler.transform(
            frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        ).astype("float32")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = MODEL_DIR / "versions" / MODEL_NAME / version
    bundle.mkdir(parents=True, exist_ok=False)
    joblib.dump(feature_scaler, bundle / "scaler.pkl")
    entries: dict[str, Any] = {}
    model_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        print(f"Preparing pooled {horizon}-session LSTM sequences...", flush=True)
        (x_fit, y_fit, _), (x_cal, y_cal, meta_cal), (x_test, y_test, meta_test) = _make_samples(
            frames, horizon, calibration_start, test_start
        )
        if len(x_fit) < 500 or len(x_cal) < 100 or len(x_test) < 100:
            raise ValueError(f"horizon={horizon}: too few pooled samples (fit={len(x_fit)}, calibration={len(x_cal)}, test={len(x_test)})")
        target_mean = y_fit.mean(axis=0).astype("float64")
        target_scale = y_fit.std(axis=0).astype("float64")
        target_scale = np.where(target_scale > 1e-6, target_scale, 1.0)
        fit_targets = ((y_fit - target_mean) / target_scale).astype("float32")
        cal_targets = ((y_cal - target_mean) / target_scale).astype("float32")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + horizon)
        model = _build_panel_lstm(horizon)
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                                 min_lr=1e-5, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, min_delta=1e-4,
                                             restore_best_weights=True, verbose=0),
        ]
        class EpochProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                values = logs or {}
                print(f"h{horizon} epoch {epoch + 1}/{epochs}: loss={values.get('loss', float('nan')):.6f}, val_loss={values.get('val_loss', float('nan')):.6f}", flush=True)
        history = model.fit(
            x_fit, fit_targets, validation_data=(x_cal, cal_targets),
            epochs=max(1, int(epochs)), batch_size=64, shuffle=True,
            callbacks=[*callbacks, EpochProgress()], verbose=0,
        )
        print(f"h{horizon} training complete; predicting calibration and holdout samples...", flush=True)
        cal_normalized = model.predict(x_cal, batch_size=2048, verbose=0)
        test_normalized = model.predict(x_test, batch_size=2048, verbose=0)
        cal_return = cal_normalized * target_scale + target_mean
        test_return = test_normalized * target_scale + target_mean
        cal_actual_return = y_cal.astype("float64")
        test_actual_return = y_test.astype("float64")
        shrinkage_grid = np.linspace(0.0, 1.0, 51)
        calibration_losses = np.asarray([
            np.mean(np.abs(cal_actual_return - cal_return * alpha))
            for alpha in shrinkage_grid
        ])
        return_shrinkage = float(shrinkage_grid[int(np.argmin(calibration_losses))])
        cal_return *= return_shrinkage
        test_return *= return_shrinkage
        residuals = cal_actual_return - cal_return
        lower_residual, upper_residual = np.quantile(residuals, [0.05, 0.95], axis=0)
        covered = ((test_actual_return >= test_return + lower_residual)
                   & (test_actual_return <= test_return + upper_residual))
        current = np.asarray([item["current_close"] for item in meta_test], dtype="float64")
        predicted_close = current * np.exp(test_return[:, -1])
        actual_close = current * np.exp(test_actual_return[:, -1])
        baseline_metrics = evaluate_panel_prices(current, current, actual_close)
        model_metrics = evaluate_panel_prices(current, predicted_close, actual_close)
        entry = {
            "model": f"h{horizon}.weights.h5",
            "metrics": model_metrics,
            "baseline_name": "zero_return_persistence",
            "baseline": baseline_metrics,
            "mae_improvement_vs_persistence": baseline_metrics["mae"] - model_metrics["mae"],
            "return_mae": float(np.mean(np.abs(test_return[:, -1] - test_actual_return[:, -1]))),
            "zero_return_mae": float(np.mean(np.abs(test_actual_return[:, -1]))),
            "return_shrinkage": return_shrinkage,
            "interval": {
                "nominal_coverage": 0.90,
                "lower_return_residual": lower_residual.tolist(),
                "upper_return_residual": upper_residual.tolist(),
                "test_coverage": float(covered.mean()),
                "calibration_samples": int(len(x_cal)),
            },
            "fit_samples": int(len(x_fit)), "calibration_samples": int(len(x_cal)),
            "test_samples": int(len(x_test)), "epochs_run": len(history.history["loss"]),
            "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        }
        model_path = bundle / entry["model"]
        print(f"h{horizon} holdout scored; saving model artifact...", flush=True)
        model.save_weights(model_path)
        entry["model_sha256"] = sha256(model_path)
        entries[str(horizon)] = entry
        for row_index, meta in enumerate(meta_test):
            ticker = meta["ticker"]
            step_pred = test_return[row_index, -1]
            step_actual = test_actual_return[row_index, -1]
            model_rows.append({
                "ticker": ticker, "horizon": horizon,
                "as_of": meta["as_of"], "target_date": meta["target_date"],
                "current_close": current[row_index],
                "predicted_close": predicted_close[row_index],
                "actual_close": actual_close[row_index],
                "predicted_return": step_pred, "actual_return": step_actual,
                "interval_covered": bool(covered[row_index, -1]),
            })
        print(f"PANEL {horizon} sessions: MAE={model_metrics['mae']:.4f}, persistence={baseline_metrics['mae']:.4f}, return-MAE={entry['return_mae']:.6f}, interval coverage={entry['interval']['test_coverage']:.1%}, epochs={entry['epochs_run']}, samples={len(x_test)}", flush=True)
        del model, x_fit, y_fit, x_cal, y_cal, x_test, y_test, fit_targets, cal_targets

    pd.DataFrame(model_rows).to_csv(bundle / "backtest_panel.csv", index=False)
    panel_backtests = pd.DataFrame(model_rows)
    for ticker, group in panel_backtests.groupby("ticker"):
        for horizon, horizon_group in group.groupby("horizon"):
            horizon_group.drop(columns=["ticker", "horizon"]).to_csv(
                bundle / f"backtest_h{int(horizon)}_{ticker}.csv", index=False
            )

    combined_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    metadata: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "ticker": MODEL_NAME,
        "version": version,
        "architecture": "pooled_residual_scale_invariant_lstm",
        "training_source": "pooled_processed_panel",
        "forecast_contract": {
            "target_column": "Close", "target_kind": "log_return_path",
            "cutoff": "after_close", "horizons": HORIZONS,
            "baseline": "zero_return_persistence",
            "evaluation_metrics": ["mae", "rmse", "mape", "lstm_directional_accuracy", "return_mae"],
        },
        "tensorflow_version": tf.__version__, "numpy_version": np.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_news": training_news,
        "features": list(FEATURE_COLUMNS), "data_hashes": hashes, "data_sha256": combined_hash,
        "data_start": str(data_start), "data_end": str(max(frame.index[-1] for frame in frames.values())),
        "fit_end": str(fit_end), "calibration_start": str(calibration_start),
        "test_start": str(test_start), "window_size": WINDOW_SIZE,
        "epochs": int(epochs), "seed": 42, "feature_scaler": "RobustScaler(10th-90th percentile)",
        "target": "per-session cumulative log-return path, standardized on fit-fold only",
        "calibration": "single return-shrinkage coefficient selected on calibration log-return MAE; holdout is not used for model selection",
        "training_tickers": sorted(frames), "training_ticker_count": len(frames),
        "horizons": entries,
    }
    atomic_json(bundle / "metadata.json", metadata)
    atomic_json(ACTIVE_PATH, {"bundle": str(bundle.relative_to(MODEL_DIR))})
    return metadata


def evaluate_panel_prices(current, predicted, actual) -> dict[str, float]:
    from src.evaluate_predictions import evaluate_predictions
    return evaluate_predictions(current, predicted, actual)
