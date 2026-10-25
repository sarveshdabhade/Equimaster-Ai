"""Immutable storage for experimental LLM/news fusion estimators."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import uuid
from typing import Any, Mapping

import joblib
import sklearn

from src.artifacts import atomic_json, model_directory, sha256
from src.fusion_dataset import (
    FUSION_DATASET_SCHEMA_VERSION,
    normalize_as_of,
    normalize_target_date,
    validate_fusion_dataset,
)
from src.fusion_model import FEATURE_NAMES
from src.fusion_evaluation import LLM_STANCE_SCORE

FUSION_ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILENAME = "fusion_model.joblib"


def _artifact_root(artifact_root: str | Path | None) -> Path:
    root = Path(artifact_root) if artifact_root is not None else model_directory() / "fusion"
    return root.resolve()


def save_fusion_artifact(
    model_result: Mapping[str, Any],
    split: Mapping[str, Any],
    artifact_root: str | Path | None = None,
) -> Path:
    """Save a fitted model and its evaluation/split provenance as an immutable bundle."""
    model = model_result.get("model")
    if model is None or not callable(getattr(model, "predict", None)):
        raise ValueError("model_result must contain a fitted estimator with predict()")
    if tuple(model_result.get("feature_names") or ()) != tuple(FEATURE_NAMES):
        raise ValueError("Fusion model feature names do not match the current contract")
    split_metadata = split.get("metadata")
    if not isinstance(split_metadata, Mapping):
        raise ValueError("A chronological split with metadata is required")
    if split_metadata.get("split_method") != "global_chronological_target_date_purged":
        raise ValueError("Fusion artifacts require the purged chronological split contract")
    partitions = {
        name: validate_fusion_dataset(split.get(name, []))
        for name in ("train", "validation", "test")
    }
    if len(partitions["train"]) != model_result.get("training_sample_count"):
        raise ValueError("Fusion model training count does not match the supplied train partition")
    all_rows = validate_fusion_dataset(
        partitions["train"] + partitions["validation"] + partitions["test"]
    )
    validation_start = normalize_as_of(split_metadata.get("validation_start"))
    test_start = normalize_as_of(split_metadata.get("test_start"))
    if any(
        normalize_as_of(row["as_of"]) >= validation_start
        or normalize_target_date(row["target_date"]) > validation_start.date()
        for row in partitions["train"]
    ):
        raise ValueError("Fusion training partition violates the validation boundary")
    if any(
        normalize_as_of(row["as_of"]) < validation_start
        or normalize_as_of(row["as_of"]) >= test_start
        or normalize_target_date(row["target_date"]) > test_start.date()
        for row in partitions["validation"]
    ):
        raise ValueError("Fusion validation partition violates the test boundary")
    if any(normalize_as_of(row["as_of"]) < test_start for row in partitions["test"]):
        raise ValueError("Fusion test partition violates the chronological boundary")

    root = _artifact_root(artifact_root)
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    staging = versions / f".{version}.tmp"
    bundle = versions / version
    staging.mkdir()
    try:
        model_path = staging / MODEL_FILENAME
        joblib.dump(model, model_path)
        dataset_payload = json.dumps(
            all_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        dataset_hash = hashlib.sha256(dataset_payload).hexdigest()
        model_ids_by_split = {
            name: {
                ticker: {
                    "llm": sorted({row["llm_model_id"] for row in rows if row["ticker"] == ticker}),
                    "lstm": sorted({row["lstm_model_id"] for row in rows if row["ticker"] == ticker}),
                }
                for ticker in sorted({row["ticker"] for row in rows})
            }
            for name, rows in partitions.items()
        }
        metadata = {
            "artifact_schema_version": FUSION_ARTIFACT_SCHEMA_VERSION,
            "fusion_dataset_schema_version": FUSION_DATASET_SCHEMA_VERSION,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "estimator": "standardized_ridge_log_return",
            "model_file": MODEL_FILENAME,
            "model_sha256": sha256(model_path),
            "dataset_sha256": dataset_hash,
            "sample_counts": {name: len(rows) for name, rows in partitions.items()},
            "model_ids_by_split": model_ids_by_split,
            "feature_names": list(FEATURE_NAMES),
            "llm_stance_mapping": dict(LLM_STANCE_SCORE),
            "alpha": float(model_result["alpha"]),
            "training_sample_count": int(model_result["training_sample_count"]),
            "split": dict(split_metadata),
            "validation_metrics": model_result["validation"],
            "test_metrics": model_result["test"],
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        }
        atomic_json(staging / "metadata.json", metadata)
        staging.replace(bundle)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return bundle


def load_fusion_artifact(
    version: str,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a locally stored fusion model after validating its bundle and hash."""
    if (
        not isinstance(version, str)
        or not version.strip()
        or version in {".", ".."}
        or Path(version).name != version
        or Path(version).is_absolute()
    ):
        raise ValueError("Fusion artifact version must be a single directory name")
    root = _artifact_root(artifact_root)
    bundle = (root / "versions" / version).resolve()
    if not bundle.is_relative_to((root / "versions").resolve()):
        raise ValueError("Fusion artifact must be inside the fusion model directory")
    metadata_path = bundle / "metadata.json"
    model_path = bundle / MODEL_FILENAME
    if not metadata_path.resolve().is_relative_to(bundle) or not model_path.resolve().is_relative_to(bundle):
        raise ValueError("Fusion artifact files must remain inside their version bundle")
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Incomplete fusion artifact bundle: {version}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Fusion artifact metadata must be a JSON object")
    if type(metadata.get("artifact_schema_version")) is not int or metadata.get("artifact_schema_version") != FUSION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported fusion artifact schema version")
    if type(metadata.get("fusion_dataset_schema_version")) is not int or metadata.get("fusion_dataset_schema_version") != FUSION_DATASET_SCHEMA_VERSION:
        raise ValueError("Fusion artifact dataset contract version is incompatible")
    if metadata.get("version") != version:
        raise ValueError("Fusion artifact version metadata does not match its directory")
    if metadata.get("feature_names") != list(FEATURE_NAMES) or metadata.get("llm_stance_mapping") != LLM_STANCE_SCORE:
        raise ValueError("Fusion artifact feature contract is incompatible")
    if metadata.get("model_file") != MODEL_FILENAME:
        raise ValueError("Fusion artifact model filename is invalid")
    if sha256(model_path) != metadata.get("model_sha256"):
        raise ValueError("Fusion model artifact checksum mismatch")

    model = joblib.load(model_path)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Fusion artifact does not contain a predictive estimator")
    return {"model": model, "metadata": metadata, "bundle": bundle}
