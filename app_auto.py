from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from core.config import AppConfig
from core.image_pipeline import find_product_folders, process_product_folder


def _get_app_dir() -> Path:
    # Source mode: project folder.
    # Frozen mode: executable is expected under <package>/runtime/.
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name.lower() == "runtime":
            return exe_path.parent.parent
        return exe_path.parent
    return Path(__file__).resolve().parent


APP_DIR = _get_app_dir()
INPUT_DIR = APP_DIR / "INPUT"
FRAME_DIR = APP_DIR / "FRAME"
CONFIG_DIR = APP_DIR / "CONFIG"
CONFIG_FILE = CONFIG_DIR / "config.json"
REMOVEBG_KEY_FILE = CONFIG_DIR / "api_key.txt"
GEMINI_KEY_FILE = CONFIG_DIR / "gemini_api_key.txt"
LOG_FILE = APP_DIR / "run_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg)
        if not msg.endswith("\n"):
            f.write("\n")


def _read_key(path: Path, placeholder: str) -> str:
    if not path.exists():
        return ""
    value = path.read_text(encoding="utf-8").strip()
    if not value or value == placeholder:
        return ""
    return value


def _load_raw_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def _find_frame_file() -> str:
    frame_candidates = [
        p for p in FRAME_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not frame_candidates:
        return ""
    frame_candidates = sorted(frame_candidates, key=lambda p: (p.suffix.lower() != ".png", p.name.lower()))
    return str(frame_candidates[0])


def load_auto_config() -> AppConfig:
    raw = _load_raw_config()
    allowed = AppConfig.__dataclass_fields__.keys()
    clean = {k: v for k, v in raw.items() if k in allowed}

    cfg = AppConfig(**clean)
    cfg.api_key = _read_key(REMOVEBG_KEY_FILE, "PASTE_YOUR_API_KEY_HERE")
    cfg.gemini_api_key = _read_key(GEMINI_KEY_FILE, "PASTE_YOUR_GEMINI_API_KEY_HERE")
    cfg.input_root_folder = str(INPUT_DIR)
    cfg.frame_file = _find_frame_file()

    # Do not force remove.bg key anymore. Gemini/OpenCV can run without it.
    if cfg.remove_background_flow == "removebg_api" and not cfg.api_key:
        raise RuntimeError("remove_background_flow=removebg_api but CONFIG/api_key.txt is empty.")
    return cfg


def _ensure_default_files() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    FRAME_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    if not GEMINI_KEY_FILE.exists():
        GEMINI_KEY_FILE.write_text("PASTE_YOUR_GEMINI_API_KEY_HERE\n", encoding="utf-8")
    if not REMOVEBG_KEY_FILE.exists():
        REMOVEBG_KEY_FILE.write_text("PASTE_YOUR_API_KEY_HERE\n", encoding="utf-8")


def main() -> int:
    LOG_FILE.write_text("", encoding="utf-8")
    print("=" * 70)
    print("eBay Image Tool - Gemini + rembg + OpenCV")
    print("=" * 70)
    print("Flow:")
    print("1) Put product folders into INPUT")
    print("2) Put optional frame into FRAME")
    print("3) Paste Gemini key into CONFIG/gemini_api_key.txt")
    print("4) Optional: paste remove.bg fallback key into CONFIG/api_key.txt")
    print("5) Open START file")
    print("6) Get final images in each product folder /output")
    print("=" * 70)

    try:
        _ensure_default_files()
        cfg = load_auto_config()
        log(f"Input folder: {INPUT_DIR}")
        log(f"Frame: {cfg.frame_file if cfg.frame_file else 'plain white canvas'}")
        log(f"Mode: {cfg.mode}")
        log(f"Flow: {cfg.remove_background_flow}")
        if cfg.remove_background_flow == "gemini_rembg_opencv":
            log(f"rembg model: {getattr(cfg, 'rembg_model', 'isnet-general-use')} | alpha_matting={getattr(cfg, 'rembg_alpha_matting', True)}")
        log(f"Gemini: {'enabled' if cfg.gemini_enabled else 'disabled'} | model={cfg.gemini_model}")
        log(f"remove.bg fallback: {'configured' if cfg.api_key else 'not configured'}")
        log(f"Component mode: {cfg.component_mode}")
        log(f"Auto-fit frame: {cfg.auto_fit_to_frame} | collision_check={cfg.frame_collision_check}")
        log(f"Budget per MPN/folder: ${cfg.current_budget():.2f}")

        folders = find_product_folders(INPUT_DIR)
        folders = [f for f in folders if f.name != cfg.output_subfolder_name]

        if not folders:
            log("\nNo images found.")
            log("Put one product folder inside INPUT, for example:")
            log("INPUT/A12345__51360-SDA-A01/main.jpg")
            return 2

        log(f"\nFound {len(folders)} product folder(s).")
        total_api_calls = 0
        total_cost = 0.0
        any_error = False

        for folder in folders:
            report = process_product_folder(folder, cfg, log)
            total_api_calls += report.api_calls
            total_cost += report.estimated_cost
            if report.status != "done":
                any_error = True
            log(
                f"Done folder: {report.folder_name} | status={report.status} | "
                f"api_calls={report.api_calls} | cost=${report.estimated_cost:.2f}\n"
            )

        log("=" * 70)
        log(f"ALL DONE. Total remove.bg API calls: {total_api_calls}. Estimated fallback cost: ${total_cost:.2f}")
        if any_error:
            log("Some images had errors. Check each /output/_processing_report.json and run_log.txt.")
        else:
            log("All images processed successfully.")
        log("=" * 70)
        return 0 if not any_error else 1

    except Exception as exc:
        log("\nERROR:")
        log(str(exc))
        log("\nDETAILS:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    code = main()
    if os.environ.get("EBAY_TOOL_NO_PAUSE") != "1":
        input("\nPress Enter to close...")
    sys.exit(code)
