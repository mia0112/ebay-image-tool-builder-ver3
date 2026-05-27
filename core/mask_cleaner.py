from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

BBox = Tuple[int, int, int, int]


def load_image_rgb(path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def corner_average_rgb(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    patch = max(5, min(h, w) // 35)
    patches = [
        arr[:patch, :patch],
        arr[:patch, w - patch : w],
        arr[h - patch : h, :patch],
        arr[h - patch : h, w - patch : w],
    ]
    return np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).mean(axis=0)


def trim_uniform_border(image: Image.Image, tolerance: int = 14) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    bg = corner_average_rgb(arr)
    diff = np.linalg.norm(arr.astype(np.int16) - bg.astype(np.int16), axis=2)
    mask = diff > tolerance
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    width, height = image.size
    pad = max(3, min(width, height) // 180)
    return image.crop((max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)))


def make_preview(image: Image.Image, max_size: int) -> Tuple[Image.Image, float, float]:
    w, h = image.size
    if max(w, h) <= max_size:
        return image.copy(), 1.0, 1.0
    scale = max_size / float(max(w, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    preview = image.resize((new_w, new_h), Image.LANCZOS)
    return preview, w / new_w, h / new_h


def clamp_bbox(box: Sequence[float | int], width: int, height: int) -> BBox:
    if len(box) != 4:
        return (0, 0, width, height)
    x, y, w, h = [int(round(float(v))) for v in box]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return (x, y, w, h)


def expand_bbox(box: BBox, width: int, height: int, ratio: float = 0.08, min_pad: int = 8) -> BBox:
    x, y, w, h = box
    pad_x = max(min_pad, int(w * ratio))
    pad_y = max(min_pad, int(h * ratio))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def bbox_intersection_area(a: BBox, b: BBox) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def bbox_area(box: BBox) -> int:
    return max(0, box[2]) * max(0, box[3])


def scale_analysis(analysis: Dict[str, Any], sx: float, sy: float, width: int, height: int) -> Dict[str, Any]:
    def scale_box(raw) -> BBox:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return (0, 0, width, height)
        x, y, w, h = raw
        return clamp_bbox([float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy], width, height)

    out = dict(analysis or {})
    out["product_bbox"] = scale_box(out.get("product_bbox", [0, 0, width, height]))
    for key in ["keep_regions", "remove_regions", "text_on_product"]:
        regions = []
        for item in out.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            new_item["bbox"] = scale_box(item.get("bbox", [0, 0, 1, 1]))
            regions.append(new_item)
        out[key] = regions
    return out


def validate_analysis(analysis: Dict[str, Any], width: int, height: int, confidence_threshold: float) -> Tuple[bool, str]:
    if not isinstance(analysis, dict):
        return False, "Gemini analysis is not a JSON object."
    if analysis.get("product_type") not in {"single_object", "multi_part_object"}:
        return False, "Gemini product_type is invalid."
    try:
        box = clamp_bbox(analysis.get("product_bbox", []), width, height)
    except Exception:
        return False, "Gemini product_bbox is invalid."
    if bbox_area(box) < max(80, int(width * height * 0.005)):
        return False, "Gemini product_bbox is too small."
    confidence = float(analysis.get("confidence") or 0.0)
    if confidence < confidence_threshold:
        return False, f"Gemini confidence too low: {confidence:.2f}."
    return True, "ok"


def bbox_mask(shape: Tuple[int, int], box: BBox, value: int = 255) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : min(h, y + bh), x : min(w, x + bw)] = value
    return mask


def preclean_remove_regions(image: Image.Image, analysis: Dict[str, Any]) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    product_box = clamp_bbox(analysis.get("product_bbox", [0, 0, w, h]), w, h)
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in analysis.get("remove_regions", []) or []:
        box = clamp_bbox(region.get("bbox", []), w, h)
        overlap = bbox_intersection_area(box, product_box) / max(1, bbox_area(box))
        reason = str(region.get("reason", "")).lower()
        if overlap < 0.20 or any(k in reason for k in ["background", "disclaimer", "watermark", "overlay"]):
            x, y, bw, bh = expand_bbox(box, w, h, ratio=0.12, min_pad=3)
            mask[y : y + bh, x : x + bw] = 255

    if not mask.any():
        return image
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned, mode="RGB")


def _remove_regions_from_mask(mask: np.ndarray, analysis: Optional[Dict[str, Any]]) -> np.ndarray:
    if not analysis:
        return mask
    h, w = mask.shape
    product_box = clamp_bbox(analysis.get("product_bbox", [0, 0, w, h]), w, h)
    out = mask.copy()
    for region in analysis.get("remove_regions", []) or []:
        box = clamp_bbox(region.get("bbox", []), w, h)
        reason = str(region.get("reason", "")).lower()
        overlap = bbox_intersection_area(box, product_box) / max(1, bbox_area(box))
        if overlap < 0.35 or any(k in reason for k in ["disclaimer", "background", "watermark", "overlay"]):
            x, y, bw, bh = expand_bbox(box, w, h, ratio=0.08, min_pad=2)
            out[y : y + bh, x : x + bw] = 0
    return out


def _include_keep_regions(mask: np.ndarray, image: Image.Image, analysis: Optional[Dict[str, Any]], threshold: int) -> np.ndarray:
    if not analysis:
        return mask
    arr = np.array(image.convert("RGB"))
    bg = corner_average_rgb(arr)
    diff = np.linalg.norm(arr.astype(np.int16) - bg.astype(np.int16), axis=2)
    h, w = mask.shape
    out = mask.copy()
    for region in analysis.get("keep_regions", []) or []:
        box = clamp_bbox(region.get("bbox", []), w, h)
        x, y, bw, bh = box
        roi = (diff[y : y + bh, x : x + bw] > max(10, threshold - 8)).astype(np.uint8) * 255
        out[y : y + bh, x : x + bw] = np.maximum(out[y : y + bh, x : x + bw], roi)
    return out


def initial_white_bg_mask(image: Image.Image, threshold: int = 28) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    bg = corner_average_rgb(arr)
    diff = np.linalg.norm(arr.astype(np.int16) - bg.astype(np.int16), axis=2)
    return (diff > threshold).astype(np.uint8) * 255


def remove_projection_border_lines(mask: np.ndarray) -> np.ndarray:
    """Remove listing/scanner lines even when they are connected to the product mask.

    Some marketplace source images contain a 1-3 px grey vertical or horizontal
    border inside the image, not exactly at the outer edge. Component filtering
    cannot remove it if it touches the product, so we remove isolated long line
    runs by projection first.
    """
    out = mask.copy()
    h, w = out.shape
    edge = max(4, int(min(w, h) * 0.025))

    def remove_long_column_runs() -> None:
        col_sum = (out > 0).sum(axis=0)
        strong = col_sum >= h * 0.45
        start = None
        for i, flag in enumerate(list(strong) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                end = i
                run_w = end - start
                if run_w <= max(4, int(w * 0.012)):
                    left = max(0, start - 1)
                    right = min(w, end + 1)
                    out[:, left:right] = 0
                start = None

    def remove_long_row_runs() -> None:
        row_sum = (out > 0).sum(axis=1)
        strong = row_sum >= w * 0.45
        start = None
        for i, flag in enumerate(list(strong) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                end = i
                run_h = end - start
                if run_h <= max(4, int(h * 0.012)):
                    top = max(0, start - 1)
                    bottom = min(h, end + 1)
                    out[top:bottom, :] = 0
                start = None

    # Outer edge lines.
    col_sum = (out > 0).sum(axis=0)
    row_sum = (out > 0).sum(axis=1)
    for x in list(range(edge)) + list(range(max(0, w - edge), w)):
        if col_sum[x] >= h * 0.22:
            out[:, max(0, x - 1) : min(w, x + 2)] = 0
    for y in list(range(edge)) + list(range(max(0, h - edge), h)):
        if row_sum[y] >= w * 0.22:
            out[max(0, y - 1) : min(h, y + 2), :] = 0

    remove_long_column_runs()
    remove_long_row_runs()
    return out


def remove_edge_border_lines(mask: np.ndarray) -> np.ndarray:
    """Remove thin border lines as full components and by edge projection."""
    mask = remove_projection_border_lines(mask)
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return mask
    out = mask.copy()
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        near_edge = x <= 6 or y <= 6 or (x + bw) >= w - 6 or (y + bh) >= h - 6
        vertical_line = bw <= max(4, int(w * 0.018)) and bh >= h * 0.25
        horizontal_line = bh <= max(4, int(h * 0.018)) and bw >= w * 0.25
        very_long_line = (bw <= max(4, int(w * 0.012)) and bh >= h * 0.50) or (bh <= max(4, int(h * 0.012)) and bw >= w * 0.50)
        sparse_line = area <= max(18, int((bw + bh) * 6))
        if ((near_edge and (vertical_line or horizontal_line)) or very_long_line) and sparse_line:
            out[labels == label] = 0
    return remove_projection_border_lines(out)


def remove_bottom_disclaimer_like_noise(mask: np.ndarray, *, aggressive: bool = True) -> np.ndarray:
    """Remove common tiny disclaimer text near the lower part of marketplace photos.

    This is deterministic and runs even when Gemini misses the text. It does not remove
    larger accessories such as washers/clips because it targets only very low, short,
    sparse components or grouped text-line components.
    """
    h, w = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return mask

    out = mask.copy()
    candidates = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        aspect = bw / max(1, bh)
        in_bottom_text_band = cy >= h * (0.78 if aggressive else 0.84)
        short = bh <= max(14, int(h * 0.045))
        small_area = area <= max(900, int(h * w * 0.012))
        text_like = short and small_area and (aspect >= 1.8 or bw <= w * 0.25)
        # Also remove long thin horizontal rules that often sit under disclaimer text.
        horizontal_rule = cy >= h * 0.80 and bh <= max(8, int(h * 0.02)) and bw >= w * 0.35
        if (in_bottom_text_band and text_like) or horizontal_rule:
            candidates.append((label, x, y, bw, bh, area))

    # Remove only if there are multiple tiny bits or a long thin line. This protects
    # single legitimate accessories in the lower half.
    if not candidates:
        return out
    if len(candidates) >= 3 or any(c[3] >= w * 0.35 for c in candidates):
        for label, *_ in candidates:
            out[labels == label] = 0
    return out


def guided_opencv_mask(
    image: Image.Image,
    analysis: Optional[Dict[str, Any]],
    *,
    threshold: int = 28,
    bbox_expand_ratio: float = 0.08,
    component_mode: str = "auto",
    keep_small_accessories: bool = True,
) -> np.ndarray:
    image = image.convert("RGB")
    w, h = image.size
    mask = initial_white_bg_mask(image, threshold=threshold)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, aggressive=True)

    if analysis:
        # Gemini is a hint source, not a hard crop. Do NOT clip the whole mask to
        # product_bbox; a bad bbox was the reason products were cut in previous builds.
        mask = _include_keep_regions(mask, image, analysis, threshold)
        mask = _remove_regions_from_mask(mask, analysis)

    product_type = str(analysis.get("product_type", "multi_part_object")) if analysis else ("multi_part_object" if component_mode != "single_part" else "single_object")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if product_type == "single_object" and component_mode != "multi_part":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    mask = clean_components(mask, analysis, component_mode=component_mode, keep_small_accessories=keep_small_accessories)
    mask = _remove_regions_from_mask(mask, analysis)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, aggressive=True)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def clean_components(
    mask: np.ndarray,
    analysis: Optional[Dict[str, Any]],
    *,
    component_mode: str = "auto",
    keep_small_accessories: bool = True,
) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    image_area = h * w
    product_type = "multi_part_object"
    product_box: BBox = (0, 0, w, h)
    keep_boxes: List[BBox] = []
    if analysis:
        product_type = str(analysis.get("product_type", product_type))
        product_box = clamp_bbox(analysis.get("product_bbox", [0, 0, w, h]), w, h)
        # Use a larger tolerance because Gemini bbox can be imperfect.
        product_box = expand_bbox(product_box, w, h, ratio=0.25, min_pad=max(12, min(w, h) // 18))
        for r in analysis.get("keep_regions", []) or []:
            keep_boxes.append(expand_bbox(clamp_bbox(r.get("bbox", []), w, h), w, h, ratio=0.18, min_pad=5))

    if component_mode == "single_part":
        product_type = "single_object"
    elif component_mode == "multi_part":
        product_type = "multi_part_object"

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        box = (x, y, bw, bh)
        inside_product = bbox_intersection_area(box, product_box) > 0
        inside_keep = any(bbox_intersection_area(box, kb) > 0 for kb in keep_boxes)
        cy = float(centroids[label][1])
        # Exclude obvious disclaimer pieces before choosing dominant components.
        bottom_text_like = cy >= h * 0.78 and bh <= max(14, int(h * 0.045)) and area <= max(900, int(image_area * 0.012))
        components.append({
            "label": label,
            "area": area,
            "box": box,
            "inside_product": inside_product,
            "inside_keep": inside_keep,
            "bottom_text_like": bottom_text_like,
        })

    kept: List[int] = []
    if product_type == "single_object":
        candidates = [c for c in components if not c["bottom_text_like"]]
        if analysis:
            inside_candidates = [c for c in candidates if c["inside_product"] or c["inside_keep"]]
            # Trust the inside filter only if it still leaves a reasonably large object.
            if inside_candidates and max(c["area"] for c in inside_candidates) >= image_area * 0.02:
                candidates = inside_candidates
        if not candidates:
            candidates = components
        best = max(candidates, key=lambda c: c["area"])
        kept.append(best["label"])
        min_accessory_area = max(20, int(image_area * 0.00003))
        for c in candidates:
            if c["inside_keep"] and c["area"] >= min_accessory_area and not c["bottom_text_like"]:
                kept.append(c["label"])
    else:
        min_area = 8 if keep_small_accessories else max(30, int(image_area * 0.00004))
        for c in components:
            if c["area"] < min_area or c["bottom_text_like"]:
                continue
            # For multi-part kits, keep all legitimate non-text components. Gemini
            # can miss small screws/washers, so do not require inside_product.
            kept.append(c["label"])

    if not kept:
        kept = [max(components, key=lambda c: c["area"])["label"]]

    out = np.isin(labels, kept).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out.astype(np.uint8)


def rgba_from_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    if mask.shape != arr.shape[:2]:
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    arr[:, :, 3] = mask.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def refine_alpha(rgba: Image.Image, analysis: Optional[Dict[str, Any]] = None, component_mode: str = "auto", keep_small_accessories: bool = True) -> Image.Image:
    rgba = rgba.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    _, binary = cv2.threshold(alpha, 12, 255, cv2.THRESH_BINARY)
    binary = remove_edge_border_lines(binary)
    binary = remove_bottom_disclaimer_like_noise(binary, aggressive=True)
    binary = clean_components(binary, analysis, component_mode=component_mode, keep_small_accessories=keep_small_accessories)
    binary = _remove_regions_from_mask(binary, analysis)
    binary = remove_edge_border_lines(binary)
    binary = remove_bottom_disclaimer_like_noise(binary, aggressive=True)
    arr[:, :, 3] = cv2.GaussianBlur(binary, (3, 3), 0)
    return Image.fromarray(arr, mode="RGBA")


def plain_opencv_remove(image: Image.Image, *, threshold: int = 28, component_mode: str = "multi_part") -> Image.Image:
    mask = initial_white_bg_mask(image, threshold=threshold)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, aggressive=True)
    mode = "multi_part" if component_mode == "auto" else component_mode
    mask = clean_components(mask, None, component_mode=mode, keep_small_accessories=True)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, aggressive=True)
    return rgba_from_mask(image, mask)
