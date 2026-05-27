from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class FitResult:
    image: Image.Image
    x: int
    y: int
    scale: float
    overlap_ratio: float
    note: str


def crop_alpha(rgba: Image.Image, padding: int = 8) -> Image.Image:
    rgba = rgba.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Product alpha mask is empty after background removal.")
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgba.width, right + padding)
    bottom = min(rgba.height, bottom + padding)
    return rgba.crop((left, top, right, bottom))


def _frame_protected_mask(frame: Image.Image, canvas_size: int, clearance_px: int) -> np.ndarray:
    frame = frame.convert("RGBA").resize((canvas_size, canvas_size), Image.LANCZOS)
    alpha = np.array(frame)[:, :, 3]
    mask = (alpha > 8).astype(np.uint8) * 255
    if clearance_px > 0:
        k = max(1, int(clearance_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _overlap_ratio(product: Image.Image, x: int, y: int, protected: np.ndarray, canvas_size: int) -> float:
    alpha = np.array(product.convert("RGBA"))[:, :, 3]
    ph, pw = alpha.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas_size, x + pw), min(canvas_size, y + ph)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    pa = alpha[y1 - y : y2 - y, x1 - x : x2 - x] > 10
    if not pa.any():
        return 1.0
    prot = protected[y1:y2, x1:x2] > 0
    return float(np.logical_and(pa, prot).sum() / max(1, pa.sum()))


def _candidate_positions(safe_rect: Tuple[int, int, int, int], product_size: Tuple[int, int]) -> list[Tuple[int, int]]:
    sx, sy, sw, sh = safe_rect
    pw, ph = product_size
    cx = sx + (sw - pw) // 2
    cy = sy + (sh - ph) // 2
    offsets = [
        (0, 0), (0, -40), (0, 40), (-40, 0), (40, 0),
        (0, -80), (0, 80), (-80, 0), (80, 0),
        (-40, -40), (40, -40), (-40, 40), (40, 40),
    ]
    positions = []
    for ox, oy in offsets:
        x = min(max(sx, cx + ox), sx + max(0, sw - pw))
        y = min(max(sy, cy + oy), sy + max(0, sh - ph))
        pos = (int(x), int(y))
        if pos not in positions:
            positions.append(pos)
    return positions


def compose_on_canvas(
    product_rgba: Image.Image,
    *,
    canvas_size: int,
    product_max_size: int,
    frame_image: Optional[Image.Image] = None,
    frame_safe_margin_left: int = 130,
    frame_safe_margin_top: int = 160,
    frame_safe_margin_right: int = 230,
    frame_safe_margin_bottom: int = 320,
    safe_area_padding_ratio: float = 0.94,
    auto_fit_to_frame: bool = True,
    target_product_width_ratio: float = 0.72,
    target_product_height_ratio: float = 0.68,
    max_product_width_ratio: float = 0.78,
    max_product_height_ratio: float = 0.72,
    frame_collision_check: bool = True,
    frame_clearance_px: int = 24,
    scale_down_step: float = 0.96,
    max_fit_attempts: int = 12,
    allowed_frame_overlap_ratio: float = 0.0005,
) -> FitResult:
    product_rgba = crop_alpha(product_rgba, padding=max(4, canvas_size // 220))

    if frame_image is None:
        scale = min(product_max_size / product_rgba.width, product_max_size / product_rgba.height)
        new_size = (max(1, int(product_rgba.width * scale)), max(1, int(product_rgba.height * scale)))
        product = product_rgba.resize(new_size, Image.LANCZOS)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
        x = (canvas_size - product.width) // 2
        y = (canvas_size - product.height) // 2
        canvas.alpha_composite(product, (x, y))
        return FitResult(canvas.convert("RGB"), x, y, scale, 0.0, "plain_canvas")

    frame = frame_image.convert("RGBA").resize((canvas_size, canvas_size), Image.LANCZOS)
    safe_x1 = max(0, int(frame_safe_margin_left))
    safe_y1 = max(0, int(frame_safe_margin_top))
    safe_x2 = min(canvas_size, max(safe_x1 + 1, canvas_size - int(frame_safe_margin_right)))
    safe_y2 = min(canvas_size, max(safe_y1 + 1, canvas_size - int(frame_safe_margin_bottom)))
    safe_w = max(1, safe_x2 - safe_x1)
    safe_h = max(1, safe_y2 - safe_y1)

    if auto_fit_to_frame:
        max_w = min(safe_w * safe_area_padding_ratio, canvas_size * max_product_width_ratio, product_max_size)
        max_h = min(safe_h * safe_area_padding_ratio, canvas_size * max_product_height_ratio, product_max_size)
    else:
        max_w = safe_w * safe_area_padding_ratio
        max_h = safe_h * safe_area_padding_ratio

    scale = min(max_w / product_rgba.width, max_h / product_rgba.height)
    scale = max(scale, 0.01)
    protected = _frame_protected_mask(frame, canvas_size, frame_clearance_px) if frame_collision_check else np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    best = None
    best_overlap = 999.0
    best_note = ""
    attempts = max(1, int(max_fit_attempts))
    step = min(0.99, max(0.80, float(scale_down_step)))

    for i in range(attempts):
        trial_scale = scale * (step ** i)
        new_w = max(1, int(product_rgba.width * trial_scale))
        new_h = max(1, int(product_rgba.height * trial_scale))
        if new_w > safe_w or new_h > safe_h:
            continue
        product = product_rgba.resize((new_w, new_h), Image.LANCZOS)
        for x, y in _candidate_positions((safe_x1, safe_y1, safe_w, safe_h), (new_w, new_h)):
            overlap = _overlap_ratio(product, x, y, protected, canvas_size) if frame_collision_check else 0.0
            if overlap < best_overlap:
                best = (product, x, y, trial_scale)
                best_overlap = overlap
                best_note = f"fit_attempt={i+1}"
            if overlap <= allowed_frame_overlap_ratio:
                canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
                canvas.alpha_composite(product, (x, y))
                canvas.alpha_composite(frame, (0, 0))
                return FitResult(canvas.convert("RGB"), x, y, trial_scale, overlap, f"auto_fit_ok; attempt={i+1}")

    if best is None:
        raise ValueError("Could not fit product into frame safe area.")

    product, x, y, final_scale = best
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    canvas.alpha_composite(product, (x, y))
    canvas.alpha_composite(frame, (0, 0))
    note = f"auto_fit_best_effort; {best_note}; overlap={best_overlap:.5f}"
    return FitResult(canvas.convert("RGB"), x, y, final_scale, best_overlap, note)
