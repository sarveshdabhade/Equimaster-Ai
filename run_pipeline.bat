@echo off
set PYTHONUTF8=1
set NUMBA_DISABLE_JIT=1
cd /d "%~dp0"
python daily_pipeline.py %*
