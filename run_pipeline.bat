@echo off
chcp 65001
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
call .venv\Scripts\activate
python daily_pipeline.py
