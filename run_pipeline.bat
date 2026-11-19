@echo off
set PYTHONUTF8=1
set NUMBA_DISABLE_JIT=1
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -u daily_pipeline.py %*
exit /b %errorlevel%
