@echo off
title DataSentry - Intelligent Dataset Quality Investigator
cd /d "%~dp0"
echo ======================================================================
echo             DataSentry - Launching Local Application (Fast Mode)
echo ======================================================================

:: Launch browser directly to localhost 127.0.0.1
start "" "http://127.0.0.1:8501"

:: Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo Launching with virtual environment...
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
) else (
    echo Running with system python...
    python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
)
pause
