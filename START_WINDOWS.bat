@echo off
cd /d "%~dp0"
echo Starting eBay Image Tool AUTO...
echo Tool folder: %cd%

where py >nul 2>nul
if %errorlevel% neq 0 (
  echo Python launcher not found. Please install Python 3.10+ from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist venv (
  echo Creating local environment...
  py -m venv venv
)

call venv\Scripts\activate
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
python app_auto.py
pause
