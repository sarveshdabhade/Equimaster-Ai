"""Read-only integrity and inference validation of active bundles."""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import numpy as np
import pandas as pd
from src.artifacts import resolve_artifact, model_directory, sha256, atomic_json
from src.forecast import predict_forecast


def main():
    failures, records = [], []
    for source in sorted((ROOT / "data" / "processed").glob("*_processed.csv")):
        ticker = source.name.removesuffix("_processed.csv")
        try:
            frame = pd.read_csv(source, index_col=0, parse_dates=True)
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError("Duplicate or unsorted dates")
            artifact = resolve_artifact(ticker)
            metadata = artifact["metadata"]
            if metadata.get("data_sha256") != sha256(source):
                raise ValueError("Active bundle does not match the processed dataset")
            if metadata.get("pipeline_version", 0) >= 3 and sha256(artifact["model"].parent / "processed.csv") != metadata["data_sha256"]:
                raise ValueError("Training snapshot hash mismatch")
            if metadata["features"] != list(frame.columns):
                raise ValueError("Feature order mismatch")
            if sha256(artifact["scaler"]) != metadata["scaler_sha256"]:
                raise ValueError("Scaler hash mismatch")
            for horizon in (1,5,22,88):
                entry = resolve_artifact(ticker, horizon)
                if sha256(entry["model"]) != entry["evaluation"]["model_sha256"]:
                    raise ValueError(f"Model hash mismatch: {horizon}")
            records.append({"ticker":ticker, "data_end":str(frame.index[-1]), "horizons":len(metadata["horizons"])})
        except Exception as exc:
            failures.append({"ticker":ticker, "error":str(exc)})
    predictions = []
    for ticker in ("TCS.NS", "NSEI"):
        frame = pd.read_csv(ROOT / "data" / "processed" / f"{ticker}_processed.csv", index_col=0, parse_dates=True)
        for horizon in (1,5,22,88):
            result = predict_forecast(frame, ticker, horizon)
            assert len(result["prices"]) == horizon and np.isfinite(result["prices"]).all()
            assert np.isfinite(result["lower"]).all() and np.isfinite(result["upper"]).all()
            predictions.append({"ticker":ticker,"horizon":horizon,"finite":True})
        result = predict_forecast(frame, ticker, 5, recursive=True)
        assert len(result["prices"]) == 5 and np.isfinite(result["prices"]).all()
        predictions.append({"ticker":ticker,"horizon":5,"recursive":True,"finite":True})
    report = {"datasets":records,"inference_checks":predictions,"failures":failures}
    atomic_json(ROOT / "data" / "evaluation" / "runtime_validation.json", report)
    print(json.dumps({"validated_datasets":len(records),"inference_checks":len(predictions),"failures":failures},indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
