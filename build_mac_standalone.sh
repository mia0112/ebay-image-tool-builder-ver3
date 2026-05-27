#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv-build" ]; then
  python3 -m venv venv-build
fi
source venv-build/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

rm -rf build dist
pyinstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name eBayImageToolCore \
  --collect-all cv2 \
  --collect-all PIL \
  --collect-all numpy \
  --collect-all google.genai \
  --collect-all pydantic \
  app_auto.py

python build_tools/package_release.py --platform mac

echo "Done. Output zip is in release_artifacts/."
