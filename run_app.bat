@echo off
title DataSentry - Intelligent Dataset Quality Investigator
cd /d "%~dp0"
echo ======================================================================
echo             DataSentry - Launching Local Application
echo ======================================================================

:: Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo Activating .venv and launching Streamlit...
    start "" "http://localhost:8501"
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
) else (
    echo Running with system python...
    start "" "http://localhost:8501"
    python -m streamlit run app.py --server.port 8501
)
pause
