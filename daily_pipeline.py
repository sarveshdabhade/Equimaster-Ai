import sys
import subprocess
import time
import logging
import os
import glob
import argparse

import pandas as pd

# --- 1. SETUP LOGGING ---
LOG_FILE = "master_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Saves history to a file
        logging.StreamHandler()         # Prints to the terminal
    ]
)

# --- 2. PIPELINE LOGIC ---
def run_script(script_path: str, step_name: str, script_args=None) -> bool:
    """
    Runs a python script and returns True if successful, False otherwise.
    """
    logging.info(f"Starting Step: {step_name} ({script_path})")
    start_time = time.perf_counter()
    cmd = [sys.executable, script_path] + (script_args or [])
    
    try:
        # sys.executable ensures we use the exact python environment running this script
        result = subprocess.run(
            cmd,
            capture_output=True, 
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True # This forces it to raise an Exception if the script fails
        )
        
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"SUCCESS: {step_name} completed in {elapsed_time:.2f} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.perf_counter() - start_time
        logging.error(f"FAILED: {step_name} crashed after {elapsed_time:.2f} seconds.")
        logging.error(f"--- ERROR DETAILS ---\n{e.stderr}\n---------------------")
        return False


def _max_date_from_csv(path: str):
    """Safely returns the max date in a CSV index, or None."""
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df.index.max()
    except Exception:
        return None


def get_stale_or_missing_tickers(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """
    Returns a tuple:
      (stale_or_missing, up_to_date)

    A ticker is stale_or_missing when:
      - processed CSV does not exist, or
      - processed max date < raw max date.
    """
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    processed_files = sorted(glob.glob(os.path.join(processed_dir, "*_processed.csv")))
    processed_map = {
        os.path.basename(p).replace("_processed.csv", ""): p
        for p in processed_files
    }

    stale_or_missing = []
    up_to_date = []

    for raw_path in raw_files:
        ticker = os.path.basename(raw_path).replace(".csv", "")
        processed_path = processed_map.get(ticker)

        raw_max = _max_date_from_csv(raw_path)
        processed_max = _max_date_from_csv(processed_path) if processed_path else None

        if processed_path is None or processed_max is None:
            stale_or_missing.append(ticker)
            continue

        if raw_max is not None and processed_max < raw_max:
            stale_or_missing.append(ticker)
        else:
            up_to_date.append(ticker)

    return stale_or_missing, up_to_date

# --- 3. EXECUTION FLOW ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Equimaster daily auto-pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run preprocessing, sequencing, and training even if processed data appears up-to-date.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step (useful for quick data refresh).",
    )
    args = parser.parse_args()

    logging.info("="*50)
    logging.info("EQUIMASTER-AI MASTER PIPELINE INITIALIZED")
    logging.info("="*50)
    
    total_start = time.perf_counter()

    # Step 1 always runs so raw data is refreshed from market source.
    tasks = [
        {
            "path": "src/update_data.py",
            "name": "Fetch Fresh Market Data",
            "args": ["--skip-downstream"],
        },
    ]
    
    # Run initial task(s) sequentially
    for task in tasks:
        success = run_script(task["path"], task["name"], task.get("args"))
        
        # The "Fail-Fast" check
        if not success:
            logging.critical("PIPELINE ABORTED: A critical step failed. Halting further execution to protect data integrity.")
            sys.exit(1) # Exit with an error code

    stale_or_missing, up_to_date = get_stale_or_missing_tickers()
    logging.info(f"Up-to-date processed tickers: {len(up_to_date)}")
    logging.info(f"Stale/missing processed tickers: {len(stale_or_missing)}")

    if stale_or_missing and not args.force:
        preview = ", ".join(stale_or_missing[:8])
        suffix = " ..." if len(stale_or_missing) > 8 else ""
        logging.info(f"Detected stale tickers: {preview}{suffix}")

    should_run_downstream = args.force or len(stale_or_missing) > 0

    if should_run_downstream:
        downstream_tasks = [
            {"path": "src/preprocessor.py", "name": "Calculate Technical Indicators"},
            {"path": "src/data_sequencer.py", "name": "Generate LSTM Sequences"},
        ]

        if not args.skip_training:
            downstream_tasks.append({"path": "src/train_model.py", "name": "Train LSTM Models"})

        for task in downstream_tasks:
            success = run_script(task["path"], task["name"], task.get("args"))
            if not success:
                logging.critical("PIPELINE ABORTED: A critical step failed. Halting further execution to protect data integrity.")
                sys.exit(1)
    else:
        logging.info("No stale processed data detected. Skipping preprocess/sequence/train steps.")

    # If we make it here, everything worked!
    total_time = time.perf_counter() - total_start
    logging.info("="*50)
    logging.info("PIPELINE COMPLETE! All systems ready for tomorrow's market.")
    logging.info(f"Total Execution Time: {total_time:.2f} seconds")
    logging.info("="*50)