@echo off
title DataSentry - Push to GitHub
cd /d "%~dp0"
echo ======================================================================
echo             DataSentry - Pushing Updates to GitHub
echo ======================================================================
echo Adding all updated files to git stage...
git add .
echo Checking git status...
git status -s
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg="Update DataSentry project files and documentation"

git commit -m "%commit_msg%"
echo Pushing branch 'main' to origin (https://github.com/satyanehasrimeesala/DataSentry.git)...
git push -u origin main

echo ======================================================================
echo Done! Check your repository: https://github.com/satyanehasrimeesala/DataSentry
echo ======================================================================
pause
