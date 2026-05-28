from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict

APP_NAME = "eBay Image Tool"
CONFIG_DIR = Path.home() / ".ebay_image_tool"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class AppConfig:
    # remove.bg / generic background-removal API fallback
    api_endpoint: str = "https://api.remove.bg/v1.0/removebg"
    api_key: str = ""
    auth_header_name: str = "X-Api-Key"
    image_field_name: str = "image_file"
    extra_form_fields: Dict[str, Any] = field(default_factory=lambda: {"size": "auto", "format": "png"})

    # Gemini vision analysis
    gemini_enabled: bool = True
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_retry_model: str = "gemini-3-flash-preview"
    gemini_preview_max_size: int = 1280
    gemini_confidence_threshold: float = 0.65
    gemini_retry_on_low_confidence: bool = True
    gemini_timeout_seconds: int = 90

    # Main folders and rendering
    input_root_folder: str = ""
    frame_file: str = ""
    canvas_size: int = 1600
    product_max_size: int = 1400
    mode: str = "standard"  # standard | test
    cost_per_api_call: float = 0.05
    max_cost_per_mpn_standard: float = 0.30
    max_cost_per_mpn_test: float = 0.50

    # Background processing
    remove_background_flow: str = "gemini_rembg_opencv"  # gemini_rembg_opencv | gemini_guided_opencv | removebg_api | opencv_local
    fallback_mode: str = "removebg_api_then_opencv"      # removebg_api | opencv_local | removebg_api_then_opencv
    local_bg_threshold: int = 28
    local_bg_enabled: bool = False
    component_mode: str = "auto"  # auto | multi_part | single_part
    preclean_text_regions: bool = True
    preserve_text_on_product: bool = True
    keep_small_accessories: bool = True
    opencv_bbox_expand_ratio: float = 0.08
    source_trim_uniform_border: bool = False

    # Open-source AI background-removal engine. Used when remove_background_flow=gemini_rembg_opencv.
    rembg_model: str = "isnet-general-use"  # isnet-general-use | u2net | u2netp
    rembg_alpha_matting: bool = True
    rembg_foreground_threshold: int = 240
    rembg_background_threshold: int = 10
    rembg_erode_size: int = 10

    # Frame fitting
    create_output_subfolder: bool = True
    output_subfolder_name: str = "output"
    frame_safe_margin_left: int = 130
    frame_safe_margin_top: int = 160
    frame_safe_margin_right: int = 230
    frame_safe_margin_bottom: int = 320
    safe_area_padding_ratio: float = 0.94
    auto_fit_to_frame: bool = True
    target_product_width_ratio: float = 0.72
    target_product_height_ratio: float = 0.68
    max_product_width_ratio: float = 0.78
    max_product_height_ratio: float = 0.72
    frame_collision_check: bool = True
    frame_clearance_px: int = 24
    scale_down_step: float = 0.96
    max_fit_attempts: int = 12
    allowed_frame_overlap_ratio: float = 0.0005

    # QA and debug
    qa_after_compose: bool = True
    qa_retry_once: bool = True
    save_debug_files: bool = True

    def current_budget(self) -> float:
        return self.max_cost_per_mpn_test if self.mode == "test" else self.max_cost_per_mpn_standard


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    ensure_config_dir()
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        save_config(cfg)
        return cfg
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = AppConfig.__dataclass_fields__.keys()
    clean = {k: v for k, v in data.items() if k in allowed}
    return AppConfig(**clean)


def save_config(config: AppConfig) -> None:
    ensure_config_dir()
    CONFIG_PATH.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
