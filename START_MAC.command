#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "Starting eBay Image Tool AUTO..."
echo "Tool folder: $(pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Please install Python 3.10+ first."
  echo "Download: https://www.python.org/downloads/"
  read -p "Press Enter to close..."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating local environment..."
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt
python app_auto.py
