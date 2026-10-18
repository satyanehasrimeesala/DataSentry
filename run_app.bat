@echo off
title DataSentry - Intelligent Dataset Quality Investigator
cd /d "%~dp0"
echo ======================================================================
echo             DataSentry - Launching Local Application
echo ======================================================================
echo Checking Python Virtual Environment...
if exist ".venv\Scripts\Activate.ps1" (
    echo Activating .venv and launching Streamlit...
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo Virtual environment not found! Running with system python...
    python -m streamlit run app.py
)
pause
