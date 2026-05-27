from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist" / "eBayImageToolCore"
ARTIFACTS_DIR = ROOT / "release_artifacts"


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def build_release(platform_name: str) -> Path:
    if platform_name not in {"mac", "windows"}:
        raise ValueError("platform must be mac or windows")

    if not DIST_DIR.exists():
        raise FileNotFoundError(f"Built runtime folder not found: {DIST_DIR}")

    release_root = ARTIFACTS_DIR / ("eBayImageTool_macOS_standalone" if platform_name == "mac" else "eBayImageTool_Windows_standalone")
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True, exist_ok=True)

    runtime_dir = release_root / "runtime"
    shutil.copytree(DIST_DIR, runtime_dir)

    # Copy working folders for end users.
    for folder_name in ["INPUT", "FRAME", "CONFIG"]:
        src = ROOT / folder_name
        dst = release_root / folder_name
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)

    # Copy docs.
    for filename in ["README_VI_START_HERE.txt", "README_BUILD_VI.md"]:
        src = ROOT / filename
        if src.exists():
            shutil.copy2(src, release_root / filename)

    howto = (
        "eBay Image Tool - Standalone Package\n\n"
        "Nguoi dung cuoi KHONG can cai Python.\n"
        "Chi can:\n"
        "1. Mo CONFIG/api_key.txt va dan API key\n"
        "2. Bo folder san pham vao INPUT\n"
        "3. Double click file START\n"
        "4. Lay anh ket qua trong output\n"
    )
    write_text(release_root / "HOW_TO_USE.txt", howto)

    if platform_name == "mac":
        start_content = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
chmod +x runtime/eBayImageToolCore || true
./runtime/eBayImageToolCore
"""
        write_text(release_root / "START_MAC.command", start_content)
    else:
        start_content = """@echo off
cd /d "%~dp0"
"runtime\\eBayImageToolCore.exe"
pause
"""
        write_text(release_root / "START_WINDOWS.bat", start_content)

    zip_path = ARTIFACTS_DIR / (release_root.name + ".zip")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in release_root.rglob("*"):
            zf.write(path, path.relative_to(release_root.parent))

    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["mac", "windows"])
    args = parser.parse_args()
    zip_path = build_release(args.platform)
    print(f"Created release: {zip_path}")
