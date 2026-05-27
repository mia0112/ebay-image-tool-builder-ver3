@echo off
cd /d "%~dp0"

if not exist venv-build (
  py -m venv venv-build
)
call venv-build\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --name eBayImageToolCore ^
  --collect-all cv2 ^
  --collect-all PIL ^
  --collect-all numpy ^
  --collect-all google.genai ^
  --collect-all pydantic ^
  app_auto.py

python build_tools\package_release.py --platform windows

echo Done. Output zip is in release_artifacts/.
pause
