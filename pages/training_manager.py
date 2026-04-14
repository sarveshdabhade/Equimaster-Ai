import os
import sys
import json
import subprocess
import streamlit as st

st.set_page_config(page_title="Training Manager", layout="wide")

st.markdown("""
<h2 style='margin-bottom:0.1rem'>Bulk Training Manager</h2>
<p style='margin-top:0rem;color:#C7D2E0'>Run sequencing + training for all available processed tickers. This runs in background and may take a long time.</p>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Bulk Training Controls")
workers = st.number_input("Parallel Workers", min_value=1, max_value=16, value=2, step=1)
epochs = st.number_input("Epochs per model", min_value=1, max_value=100, value=3, step=1)
only_missing = st.checkbox("Only train missing models", value=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Run Sequencer + Train All Models (background)"):
        try:
            runner = os.path.join('scripts', 'train_manager.py')
            args = [sys.executable, runner, '--workers', str(workers), '--epochs', str(epochs)]
            if only_missing:
                args.append('--only-missing')
            p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.success(f"Training started in background (pid {p.pid}). Check logs/progress_*.json for per-ticker progress.")
        except Exception as e:
            st.error(f"Failed to start background training: {e}")

with col2:
    if st.button("Refresh Progress"):
        st.experimental_rerun()

st.markdown("---")

# Show a summary of per-ticker progress files
log_dir = os.path.join('logs')
if os.path.exists(log_dir):
    files = sorted([f for f in os.listdir(log_dir) if f.startswith('progress_') and f.endswith('.json')])
    if files:
        for fname in files:
            try:
                with open(os.path.join(log_dir, fname), 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                ticker_name = fname.replace('progress_', '').replace('.json', '')
                st.write(f"**{ticker_name}** — {data.get('status')} — {data.get('step')} — {data.get('message')}")
            except Exception:
                continue
    else:
        st.info('No progress files yet. Start a training run to generate progress.')
else:
    st.info('No logs directory found. Start a training run to create progress files.')
