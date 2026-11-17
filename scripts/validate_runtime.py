"""Read-only integrity and inference validation of active bundles."""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from config import ALL_DOWNLOAD_TICKERS, HORIZONS, PIPELINE_VERSION
from src.artifacts import resolve_artifact, model_directory, sha256, atomic_json
from src.forecast import predict_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-current", action="store_true", help="Fail when an intact bundle needs retraining on newer data")
    parser.add_argument("--skip-inference", action="store_true", help="Audit integrity only; report inference as skipped")
    args = parser.parse_args()
    failures, records = [], []
    processed = ROOT / "data" / "processed"
    tickers = list(dict.fromkeys([*ALL_DOWNLOAD_TICKERS, "NSEI"]))
    for ticker in tickers:
        source = processed / f"{ticker}_processed.csv"
        if not source.exists():
            failures.append({"ticker": ticker, "error": "Processed dataset is missing"})
            continue
        try:
            frame = pd.read_csv(source, index_col=0, parse_dates=True)
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError("Duplicate or unsorted dates")
            artifact = resolve_artifact(ticker)
            metadata = artifact["metadata"]
            if metadata.get("pipeline_version") != PIPELINE_VERSION:
                raise ValueError("Active bundle uses an outdated pipeline version")
            current = metadata.get("data_sha256") == sha256(source)
            if metadata.get("pipeline_version", 0) >= 3 and sha256(artifact["model"].parent / "processed.csv") != metadata["data_sha256"]:
                raise ValueError("Training snapshot hash mismatch")
            if metadata["features"] != list(frame.columns):
                raise ValueError("Feature order mismatch")
            if sha256(artifact["scaler"]) != metadata["scaler_sha256"]:
                raise ValueError("Scaler hash mismatch")
            for horizon in HORIZONS:
                entry = resolve_artifact(ticker, horizon)
                if sha256(entry["model"]) != entry["evaluation"]["model_sha256"]:
                    raise ValueError(f"Model hash mismatch: {horizon}")
            records.append({"ticker":ticker, "data_end":str(frame.index[-1]),
                            "training_data_end": metadata.get("data_end"), "version": metadata.get("version"),
                            "current_data_match": current, "retraining_required": not current,
                            "horizons":len(metadata["horizons"])})
            if args.require_current and not current:
                failures.append({"ticker":ticker, "error":"Intact bundle needs retraining on the latest dataset"})
        except Exception as exc:
            failures.append({"ticker":ticker, "error":str(exc)})
    predictions = []
    for record in ([] if args.skip_inference else records):
        ticker = record["ticker"]
        frame = pd.read_csv(ROOT / "data" / "processed" / f"{ticker}_processed.csv", index_col=0, parse_dates=True)
        for horizon in HORIZONS:
            try:
                result = predict_forecast(frame, ticker, horizon)
                assert len(result["prices"]) == horizon and np.isfinite(result["prices"]).all()
                assert (np.asarray(result["prices"]) > 0).all()
                assert np.isfinite(result["lower"]).all() and np.isfinite(result["upper"]).all()
                assert (np.asarray(result["lower"]) <= np.asarray(result["upper"])).all()
                predictions.append({"ticker":ticker,"horizon":horizon,"finite":True})
            except Exception as exc:
                failures.append({"ticker":ticker,"horizon":horizon,"error":f"Inference: {type(exc).__name__}: {exc}"})
    report = {"as_of":datetime.now(timezone.utc).isoformat(), "requested":len(tickers),
              "inference_skipped":args.skip_inference,
              "current_bundles":sum(record["current_data_match"] for record in records),
              "freshness_required":args.require_current,
              "datasets":records,"inference_checks":predictions,"failures":failures}
    atomic_json(ROOT / "data" / "evaluation" / "runtime_validation.json", report)
    print(json.dumps({"validated_datasets":len(records),"current_bundles":report["current_bundles"],
                      "inference_checks":len(predictions),"failures":failures},indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
