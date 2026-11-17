"""Resolve immutable model/scaler bundles, with explicit legacy fallback."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from config import MODEL_STORAGE_PATH_ENV, resolve_app_path


def model_directory() -> Path:
    return Path(os.getenv(MODEL_STORAGE_PATH_ENV, resolve_app_path("models")))


def canonical_ticker(ticker: str) -> str:
    return "NSEI" if ticker in ("NIFTY 50", "^NSEI") else ticker.upper()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def resolve_artifact(ticker: str, horizon: int = 1, prefer_panel: bool = True) -> dict:
    ticker = canonical_ticker(ticker)
    root = model_directory()
    panel_manifest = root / "active_PANEL_LSTM.json"
    if prefer_panel and panel_manifest.exists():
        pointer = json.loads(panel_manifest.read_text(encoding="utf-8"))
        bundle = (root / pointer["bundle"]).resolve()
        if not bundle.is_relative_to(root.resolve()):
            raise ValueError("Pooled model bundle must be inside the model directory")
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        entry = metadata.get("horizons", {}).get(str(horizon))
        if entry is None:
            raise FileNotFoundError(f"Pooled LSTM has no horizon {horizon} for {ticker}")
        return {"model": bundle / entry["model"], "scaler": bundle / "scaler.pkl",
                "metadata": metadata, "evaluation": entry, "legacy": False,
                "pooled": True, "bundle": bundle}
    manifest = root / f"active_{ticker}.json"
    if manifest.exists():
        pointer = json.loads(manifest.read_text(encoding="utf-8"))
        bundle = (root / pointer["bundle"]).resolve()
        if not bundle.is_relative_to(root.resolve()):
            raise ValueError("Model bundle must be inside the model directory")
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        entry = metadata["horizons"].get(str(horizon))
        if entry is None:
            raise FileNotFoundError(f"No active horizon {horizon} for {ticker}")
        return {"model": bundle / entry["model"], "scaler": bundle / "scaler.pkl",
                "metadata": metadata, "evaluation": entry, "legacy": False}
    suffix = "" if horizon == 1 else f"_h{horizon}"
    for extension in (".keras", ".h5"):
        path = root / f"lstm_{ticker}{suffix}{extension}"
        if path.exists():
            return {"model": path, "scaler": Path(resolve_app_path("data", "train_data", f"scaler_{ticker}.pkl")),
                    "metadata": {}, "evaluation": {}, "legacy": True}
    raise FileNotFoundError(f"No model for {ticker}, horizon {horizon}")
