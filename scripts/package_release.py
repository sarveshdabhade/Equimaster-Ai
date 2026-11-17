"""Package a clean source checkout with its validated local runtime artifacts."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_DOWNLOAD_TICKERS, HORIZONS
from scripts.train_manager import active_bundle_matches_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ci-report", type=Path, help="Require successful remote CI for this exact source commit")
    args = parser.parse_args()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode().strip()
    if dirty:
        raise SystemExit("Commit or resolve the reviewed source changes before packaging the release.")
    tickers = [*ALL_DOWNLOAD_TICKERS, "NSEI"]
    stale = [ticker for ticker in tickers if not active_bundle_matches_data(ticker, HORIZONS)]
    if stale:
        raise SystemExit("Missing or outdated model bundles: " + ", ".join(stale))
    files = {}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, tracked):
        path = ROOT / name
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    # Keep provider cookie databases, environment secrets and Python environments
    # outside the archive. Only the explicit runtime directories are selected.
    for directory in ("data/raw", "data/processed", "data/fundamentals", "data/news_cache",
                      "data/news_archive", "data/evaluation", "data/benchmarks", "models/fundamental_ml"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix != ".tmp":
                files[path.relative_to(ROOT).as_posix()] = path
    bundles = []
    for ticker in tickers:
        pointer = ROOT / "models" / f"active_{ticker}.json"
        files[pointer.relative_to(ROOT).as_posix()] = pointer
        bundle_name = json.loads(pointer.read_text())["bundle"]
        bundle = (ROOT / "models" / bundle_name).resolve()
        if not bundle.is_relative_to((ROOT / "models").resolve()):
            raise ValueError("Active bundle is outside the model directory")
        for path in bundle.rglob("*"):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
        metadata = json.loads((bundle / "metadata.json").read_text())
        bundles.append({"ticker": ticker, "bundle": bundle_name.replace("\\", "/"),
                        "version": metadata["version"], "data_sha256": metadata["data_sha256"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "model_count": len(bundles), "horizons": HORIZONS, "bundles": bundles, "files": []}
    if args.ci_report:
        runs = json.loads(args.ci_report.read_text(encoding="utf-8-sig"))
        matching = [run for run in runs if run.get("head_sha") == manifest["git_commit"] and run.get("conclusion") == "success"]
        if not matching:
            raise SystemExit("Successful remote CI for the current commit is required by --ci-report")
        manifest["remote_ci"] = matching[0]
    status_files = ["daily_status.json", "training_status.json", "training_news.json", "news_refresh.json",
                    "fundamentals_refresh.json", "fundamental_ml_status.json", *[f"progress_{ticker}.json" for ticker in tickers]]
    for name in status_files:
        path = ROOT / "logs" / name
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    with tempfile.TemporaryDirectory(prefix="equimaster-release-") as temporary:
        database = ROOT / "data/fundamental_history.sqlite3"
        if database.exists():
            snapshot = Path(temporary) / database.name
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source, sqlite3.connect(snapshot) as target:
                source.backup(target)
            files["data/fundamental_history.sqlite3"] = snapshot
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, "Equimaster-Ai/" + name)
                manifest["files"].append({"path": name, "bytes": path.stat().st_size,
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            archive.writestr("Equimaster-Ai/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError("Archive integrity failed: " + invalid)
        # Verify every published bundle is represented, including all horizons.
        entries = set(archive.namelist())
        for bundle in bundles:
            prefix = "Equimaster-Ai/models/" + bundle["bundle"] + "/"
            for name in ("metadata.json", "scaler.pkl", "processed.csv", *[f"h{h}.weights.h5" for h in HORIZONS]):
                if prefix + name not in entries:
                    raise RuntimeError("Archive is missing " + prefix + name)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest + "  " + args.output.name + "\n")
    print(json.dumps({"archive": str(args.output), "files": len(files), "models": len(bundles),
                      "bytes": args.output.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
