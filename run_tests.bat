@echo off
title DataSentry - Automated Test Suite
cd /d "%~dp0"
echo ======================================================================
echo             DataSentry - Executing Automated Test Suite
echo ======================================================================
echo Running pytest across all unit tests...
".venv\Scripts\python.exe" -m pytest tests -v
echo ======================================================================
pause
