from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class RembgEngineError(Exception):
    pass


@dataclass
class RembgConfig:
    model: str = "isnet-general-use"
    alpha_matting: bool = True
    foreground_threshold: int = 240
    background_threshold: int = 10
    erode_size: int = 10
    post_morph_close: int = 3
    post_morph_open: int = 0


_SESSION_CACHE: dict[str, object] = {}


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _get_session(model: str):
    try:
        from rembg import new_session  # type: ignore
    except Exception as exc:
        raise RembgEngineError(
            "Package rembg is not installed. Install with: pip install rembg onnxruntime"
        ) from exc

    model = (model or "isnet-general-use").strip()
    if model not in _SESSION_CACHE:
        _SESSION_CACHE[model] = new_session(model)
    return _SESSION_CACHE[model]


def rembg_alpha_mask(image: Image.Image, cfg: Optional[RembgConfig] = None) -> np.ndarray:
    """Return an alpha mask from rembg while preserving original RGB elsewhere.

    rembg may output its own RGB values. This function uses only its alpha channel.
    The caller should combine the returned alpha with the original source RGB.
    """
    cfg = cfg or RembgConfig()
    try:
        from rembg import remove  # type: ignore
    except Exception as exc:
        raise RembgEngineError(
            "Package rembg is not installed. Install with: pip install rembg onnxruntime"
        ) from exc

    session = _get_session(cfg.model)
    try:
        out_bytes = remove(
            _png_bytes(image),
            session=session,
            alpha_matting=bool(cfg.alpha_matting),
            alpha_matting_foreground_threshold=int(cfg.foreground_threshold),
            alpha_matting_background_threshold=int(cfg.background_threshold),
            alpha_matting_erode_size=int(cfg.erode_size),
        )
        rgba = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    except Exception as exc:
        raise RembgEngineError(f"rembg failed: {exc}") from exc

    if rgba.size != image.size:
        rgba = rgba.resize(image.size, Image.LANCZOS)

    alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]

    # Light alpha-only cleanup. This is NOT RGB editing.
    binary = np.where(alpha > 12, 255, 0).astype(np.uint8)
    if cfg.post_morph_close and cfg.post_morph_close > 1:
        k = int(cfg.post_morph_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if cfg.post_morph_open and cfg.post_morph_open > 1:
        k = int(cfg.post_morph_open)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Preserve antialias by keeping the stronger of original soft alpha and binary shell.
    alpha = np.maximum(alpha, cv2.GaussianBlur(binary, (3, 3), 0)).astype(np.uint8)
    return alpha
