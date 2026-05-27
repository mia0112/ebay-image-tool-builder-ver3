from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

BBox = Tuple[int, int, int, int]


def load_image_rgb(path) -> Image.Image:
    """Load an image without changing its RGB pixels except EXIF orientation."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def corner_average_rgb(arr: np.ndarray) -> np.ndarray:
    """Robust background colour estimate from the four corners.

    Median is deliberately used instead of mean so a small logo/border at one
    corner does not pull the background estimate away from white.
    """
    h, w = arr.shape[:2]
    patch = max(5, min(h, w) // 35)
    patches = [
        arr[:patch, :patch],
        arr[:patch, w - patch : w],
        arr[h - patch : h, :patch],
        arr[h - patch : h, w - patch : w],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(pixels, axis=0)


def trim_uniform_border(image: Image.Image, tolerance: int = 14, max_crop_fraction: float = 0.18) -> Image.Image:
    """Optional conservative border trim.

    The production pipeline keeps this disabled by default. When enabled, this
    refuses large crops so the source product cannot be cut just because Gemini
    or thresholding found a wrong bbox. The final product crop is still based on
    alpha only, not on Gemini boxes.
    """
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
    # Safety: a trim that removes too much can indicate a bad background estimate.
    if (
        left > width * max_crop_fraction
        or top > height * max_crop_fraction
        or (width - right) > width * max_crop_fraction
        or (height - bottom) > height * max_crop_fraction
    ):
        return image
    pad = max(6, min(width, height) // 120)
    return image.crop((max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)))


def make_preview(image: Image.Image, max_size: int) -> Tuple[Image.Image, float, float]:
    w, h = image.size
    if max(w, h) <= max_size:
        return image.copy(), 1.0, 1.0
    scale = max_size / float(max(w, h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
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
    pad_x = max(min_pad, int(round(w * ratio)))
    pad_y = max(min_pad, int(round(h * ratio)))
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


def _bbox_center(box: BBox) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _point_in_box(pt: Tuple[float, float], box: BBox) -> bool:
    px, py = pt
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def _touches_any(box: BBox, boxes: Sequence[BBox]) -> bool:
    return any(bbox_intersection_area(box, b) > 0 for b in boxes)


def _center_in_any(box: BBox, boxes: Sequence[BBox]) -> bool:
    center = _bbox_center(box)
    return any(_point_in_box(center, b) for b in boxes)


def _analysis_boxes(
    analysis: Optional[Dict[str, Any]],
    width: int,
    height: int,
    *,
    product_expand_ratio: float = 0.16,
    keep_expand_ratio: float = 0.20,
) -> Tuple[BBox, List[BBox], List[BBox]]:
    """Return product box, keep boxes and text-on-product boxes as soft hints."""
    product_box: BBox = (0, 0, width, height)
    keep_boxes: List[BBox] = []
    text_boxes: List[BBox] = []
    if not analysis:
        return product_box, keep_boxes, text_boxes

    product_box = expand_bbox(
        clamp_bbox(analysis.get("product_bbox", [0, 0, width, height]), width, height),
        width,
        height,
        ratio=product_expand_ratio,
        min_pad=max(8, min(width, height) // 80),
    )
    for r in analysis.get("keep_regions", []) or []:
        if not isinstance(r, dict):
            continue
        keep_boxes.append(
            expand_bbox(
                clamp_bbox(r.get("bbox", []), width, height),
                width,
                height,
                ratio=keep_expand_ratio,
                min_pad=max(4, min(width, height) // 180),
            )
        )
    for r in analysis.get("text_on_product", []) or []:
        if not isinstance(r, dict):
            continue
        text_boxes.append(
            expand_bbox(
                clamp_bbox(r.get("bbox", []), width, height),
                width,
                height,
                ratio=0.10,
                min_pad=3,
            )
        )
    return product_box, keep_boxes, text_boxes


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
    """Validate Gemini JSON while treating Gemini as a soft hint.

    Low confidence is a warning, not an automatic failure, because OpenCV is the
    authority for the alpha mask. Invalid structure/bbox still fails.
    """
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
    if confidence < max(0.15, confidence_threshold * 0.30):
        return False, f"Gemini confidence unusably low: {confidence:.2f}."
    if confidence < confidence_threshold:
        return True, f"ok_with_low_confidence_warning={confidence:.2f}"
    return True, "ok"


def bbox_mask(shape: Tuple[int, int], box: BBox, value: int = 255) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : min(h, y + bh), x : min(w, x + bw)] = value
    return mask


def preclean_remove_regions(image: Image.Image, analysis: Dict[str, Any]) -> Image.Image:
    """Deprecated compatibility stub.

    v3 intentionally does not inpaint/edit RGB before final output. Unwanted text
    and borders are removed from alpha mask only via _remove_regions_from_mask().
    """
    return image


def _remove_regions_from_mask(mask: np.ndarray, analysis: Optional[Dict[str, Any]]) -> np.ndarray:
    if not analysis:
        return mask
    h, w = mask.shape
    product_box, keep_boxes, text_boxes = _analysis_boxes(analysis, w, h, product_expand_ratio=0.04, keep_expand_ratio=0.12)
    protected_boxes = [product_box] + keep_boxes + text_boxes
    out = mask.copy()
    remove_keywords = ("disclaimer", "background", "watermark", "overlay", "border", "unrelated")

    for region in analysis.get("remove_regions", []) or []:
        if not isinstance(region, dict):
            continue
        box = clamp_bbox(region.get("bbox", []), w, h)
        reason = str(region.get("reason", "")).lower()
        x, y, bw, bh = expand_bbox(box, w, h, ratio=0.10, min_pad=2)
        region_box = (x, y, bw, bh)
        overlap_product = bbox_intersection_area(region_box, product_box) / max(1, bbox_area(region_box))
        overlaps_keep = _touches_any(region_box, keep_boxes + text_boxes)

        # Remove only clear non-product regions. If Gemini accidentally marks real
        # product marking as removable, text_on_product/keep_regions protect it.
        strong_reason = any(k in reason for k in remove_keywords)
        mostly_outside_product = overlap_product < 0.18
        if strong_reason and (mostly_outside_product or not overlaps_keep):
            out[y : y + bh, x : x + bw] = 0
        elif mostly_outside_product and not _touches_any(region_box, protected_boxes):
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
        if not isinstance(region, dict):
            continue
        box = expand_bbox(clamp_bbox(region.get("bbox", []), w, h), w, h, ratio=0.08, min_pad=3)
        x, y, bw, bh = box
        roi_diff = diff[y : y + bh, x : x + bw]
        # Keep regions use a lower threshold to protect small clips/screws, but
        # still do not fill the whole rectangle with alpha.
        roi = (roi_diff > max(5, threshold - 18)).astype(np.uint8) * 255
        if roi.size and roi.sum() == 0:
            roi = (roi_diff > 3).astype(np.uint8) * 255
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        out[y : y + bh, x : x + bw] = np.maximum(out[y : y + bh, x : x + bw], roi)
    return out


def initial_white_bg_mask(image: Image.Image, threshold: int = 28) -> np.ndarray:
    """Create a foreground alpha candidate from a light/white marketplace bg.

    This returns a mask only; RGB pixels are never edited. The mask keeps product
    pixels that differ from the corner background and leaves white holes (for
    example hub center holes) transparent.
    """
    arr = np.array(image.convert("RGB"))
    bg = corner_average_rgb(arr)
    diff = np.linalg.norm(arr.astype(np.int16) - bg.astype(np.int16), axis=2)
    max_abs = np.max(np.abs(arr.astype(np.int16) - bg.astype(np.int16)), axis=2)

    # Foreground if it is meaningfully different from the corner background.
    mask = ((diff > threshold) | (max_abs > max(14, int(threshold * 0.65)))).astype(np.uint8) * 255
    return mask


def remove_projection_border_lines(mask: np.ndarray) -> np.ndarray:
    """Remove listing/scanner lines even when they are connected to the product mask."""
    out = mask.copy()
    h, w = out.shape
    edge = max(4, int(min(w, h) * 0.025))

    def remove_long_column_runs() -> None:
        col_sum = (out > 0).sum(axis=0)
        strong = col_sum >= h * 0.42
        start = None
        for i, flag in enumerate(list(strong) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                end = i
                run_w = end - start
                if run_w <= max(5, int(w * 0.014)):
                    left = max(0, start - 1)
                    right = min(w, end + 1)
                    out[:, left:right] = 0
                start = None

    def remove_long_row_runs() -> None:
        row_sum = (out > 0).sum(axis=1)
        strong = row_sum >= w * 0.42
        start = None
        for i, flag in enumerate(list(strong) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                end = i
                run_h = end - start
                if run_h <= max(5, int(h * 0.014)):
                    top = max(0, start - 1)
                    bottom = min(h, end + 1)
                    out[top:bottom, :] = 0
                start = None

    col_sum = (out > 0).sum(axis=0)
    row_sum = (out > 0).sum(axis=1)
    for x in list(range(edge)) + list(range(max(0, w - edge), w)):
        if col_sum[x] >= h * 0.18:
            out[:, max(0, x - 1) : min(w, x + 2)] = 0
    for y in list(range(edge)) + list(range(max(0, h - edge), h)):
        if row_sum[y] >= w * 0.18:
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
        near_edge = x <= 8 or y <= 8 or (x + bw) >= w - 8 or (y + bh) >= h - 8
        vertical_line = bw <= max(5, int(w * 0.018)) and bh >= h * 0.22
        horizontal_line = bh <= max(5, int(h * 0.018)) and bw >= w * 0.22
        very_long_line = (bw <= max(5, int(w * 0.014)) and bh >= h * 0.45) or (bh <= max(5, int(h * 0.014)) and bw >= w * 0.45)
        sparse_line = area <= max(24, int((bw + bh) * 7))
        if ((near_edge and (vertical_line or horizontal_line)) or very_long_line) and sparse_line:
            out[labels == label] = 0
    return remove_projection_border_lines(out)


def remove_bottom_disclaimer_like_noise(
    mask: np.ndarray,
    analysis: Optional[Dict[str, Any]] = None,
    *,
    aggressive: bool = True,
) -> np.ndarray:
    """Remove common tiny disclaimer text near the lower source-image area.

    This targets text-line components, not real accessories. Gemini keep/product
    regions are treated as soft protection so screws/clips at the bottom are not
    removed when Gemini correctly reports them.
    """
    h, w = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return mask

    product_box, keep_boxes, text_boxes = _analysis_boxes(analysis, w, h, product_expand_ratio=0.06, keep_expand_ratio=0.16)
    protected = [product_box] + keep_boxes + text_boxes if analysis else []
    image_area = h * w
    out = mask.copy()
    candidates = []

    # If Gemini gives a product box, anything clearly below that box is likely a
    # disclaimer. Otherwise use only the very low source-image band.
    product_bottom = product_box[1] + product_box[3] if analysis else int(h * 0.82)
    low_start = min(int(h * 0.86), max(int(h * (0.64 if aggressive else 0.72)), product_bottom + max(4, h // 120)))

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        box = (x, y, bw, bh)
        if protected and (_touches_any(box, protected) or _center_in_any(box, protected)):
            continue

        aspect = bw / max(1, bh)
        in_low_band = cy >= low_start
        short = bh <= max(16, int(h * 0.040))
        tiny_short = bh <= max(10, int(h * 0.026))
        small_area = area <= max(700, int(image_area * 0.0045))
        sparse = area <= max(24, int(bw * bh * 0.55))
        text_like = short and small_area and sparse and (aspect >= 1.35 or bw <= w * 0.12)
        horizontal_rule = cy >= h * 0.70 and bh <= max(8, int(h * 0.018)) and bw >= w * 0.25
        long_text_line = in_low_band and short and bw >= w * 0.18 and sparse
        tiny_letter = in_low_band and tiny_short and area <= max(80, int(image_area * 0.00030))
        if horizontal_rule or long_text_line or (in_low_band and text_like) or tiny_letter:
            row_bin = int(cy // max(8, h // 80))
            candidates.append((label, x, y, bw, bh, area, row_bin, horizontal_rule or long_text_line))

    if not candidates:
        return out

    row_counts: Dict[int, int] = {}
    for *_, row_bin, _strong in candidates:
        row_counts[row_bin] = row_counts.get(row_bin, 0) + 1

    for label, x, y, bw, bh, area, row_bin, strong in candidates:
        if strong or row_counts.get(row_bin, 0) >= 3 or len(candidates) >= 5:
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
    mask = initial_white_bg_mask(image, threshold=threshold)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, analysis=analysis, aggressive=True)

    if analysis:
        # Gemini is a hint source only. We use keep/remove regions to adjust mask;
        # we never clip/crop the mask to Gemini product_bbox.
        mask = _include_keep_regions(mask, image, analysis, threshold)
        mask = _remove_regions_from_mask(mask, analysis)

    product_type = str(analysis.get("product_type", "multi_part_object")) if analysis else ("multi_part_object" if component_mode != "single_part" else "single_object")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if product_type == "single_object" and component_mode != "multi_part":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    mask = clean_components(mask, analysis, component_mode=component_mode, keep_small_accessories=keep_small_accessories)
    if product_type == "single_object" and component_mode != "multi_part":
        mask = rebuild_single_object_mask(mask, analysis, keep_small_accessories=keep_small_accessories)
    mask = _remove_regions_from_mask(mask, analysis)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, analysis=analysis, aggressive=True)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _build_component_table(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = (mask > 0).astype(np.uint8) * 255
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def _preserve_large_internal_holes(main_mask: np.ndarray, *, min_area: int) -> np.ndarray:
    h, w = main_mask.shape
    inv = ((main_mask == 0).astype(np.uint8)) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(main_mask)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_edge = x <= 0 or y <= 0 or (x + bw) >= w - 1 or (y + bh) >= h - 1
        if touches_edge:
            continue
        if area >= min_area:
            keep[labels == label] = 255
    return keep


def rebuild_single_object_mask(
    mask: np.ndarray,
    analysis: Optional[Dict[str, Any]] = None,
    *,
    keep_small_accessories: bool = True,
) -> np.ndarray:
    """Rebuild the silhouette for single-object products.

    Purpose:
    - fill patchy / lem-nhem holes created inside shiny products
    - remove thin leftover edge slivers / black side borders

    Important: this still only edits the alpha mask, never RGB.
    """
    num_labels, labels, stats, centroids = _build_component_table(mask)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    image_area = h * w
    product_box: BBox = (0, 0, w, h)
    keep_boxes: List[BBox] = []
    text_boxes: List[BBox] = []
    if analysis:
        product_box, keep_boxes, text_boxes = _analysis_boxes(
            analysis,
            w,
            h,
            product_expand_ratio=0.24,
            keep_expand_ratio=0.20,
        )

    components: List[Dict[str, Any]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        box = (x, y, bw, bh)
        comp = {
            "label": label,
            "area": area,
            "box": box,
            "centroid": (float(centroids[label][0]), float(centroids[label][1])),
            "inside_product": bbox_intersection_area(box, product_box) > 0,
            "center_in_product": _point_in_box(_bbox_center(box), product_box),
            "inside_keep": _touches_any(box, keep_boxes + text_boxes),
        }
        comp.update(_component_features(comp, h, w, image_area))
        components.append(comp)

    if not components:
        return mask

    candidates = [c for c in components if not c["border_like"] and not c["bottom_text_like"]]
    if analysis:
        inside = [c for c in candidates if c["inside_product"] or c["inside_keep"]]
        if inside:
            candidates = inside
    if not candidates:
        candidates = components

    main = max(candidates, key=lambda c: c["area"])
    main_mask = np.where(labels == main["label"], 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    rebuilt = np.zeros_like(mask)
    cv2.drawContours(rebuilt, contours, -1, 255, thickness=-1)

    # Preserve only large structural holes (for example center bores), and fill
    # smaller accidental holes caused by weak white-background thresholding.
    keep_holes = _preserve_large_internal_holes(
        main_mask,
        min_area=max(12000, int(main["area"] * 0.008), int(image_area * 0.0022)),
    )
    rebuilt[keep_holes > 0] = 0

    # Keep clearly intentional accessory components that Gemini/cleanup already
    # protected, without bringing back edge-border residue.
    min_extra_area = 8 if keep_small_accessories else max(20, int(image_area * 0.00003))
    for comp in components:
        if comp["label"] == main["label"]:
            continue
        if comp["area"] < min_extra_area:
            continue
        if comp["border_like"] or comp["bottom_text_like"]:
            continue
        if comp["inside_keep"] or (keep_small_accessories and comp["center_in_product"] and not comp["text_like"]):
            rebuilt[labels == comp["label"]] = 255

    rebuilt = cv2.morphologyEx(rebuilt, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return rebuilt.astype(np.uint8)


def _component_features(component: Dict[str, Any], h: int, w: int, image_area: int) -> Dict[str, bool]:
    x, y, bw, bh = component["box"]
    area = component["area"]
    cx, cy = component["centroid"]
    aspect = bw / max(1, bh)
    thin_h = bh <= max(8, int(h * 0.018)) and bw >= w * 0.18
    thin_v = bw <= max(8, int(w * 0.018)) and bh >= h * 0.18
    edge_touch = x <= 6 or y <= 6 or x + bw >= w - 6 or y + bh >= h - 6
    sparse = area <= max(24, int(bw * bh * 0.58))
    short = bh <= max(16, int(h * 0.040))
    small = area <= max(800, int(image_area * 0.0045))
    bottom_text_like = cy >= h * 0.74 and short and small and sparse and (aspect >= 1.35 or bw <= w * 0.14)
    border_like = edge_touch and sparse and (thin_h or thin_v)
    text_like = short and small and sparse and (aspect >= 1.35 or (bw <= w * 0.12 and bh <= h * 0.035))
    return {
        "bottom_text_like": bottom_text_like,
        "border_like": border_like,
        "text_like": text_like,
        "line_like": thin_h or thin_v,
    }


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
    text_boxes: List[BBox] = []
    if analysis:
        product_type = str(analysis.get("product_type", product_type))
        product_box, keep_boxes, text_boxes = _analysis_boxes(
            analysis,
            w,
            h,
            # generous because Gemini bbox is only a hint, never a hard crop
            product_expand_ratio=0.30,
            keep_expand_ratio=0.22,
        )

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
        component = {
            "label": label,
            "area": area,
            "box": box,
            "centroid": (float(centroids[label][0]), float(centroids[label][1])),
            "inside_product": bbox_intersection_area(box, product_box) > 0,
            "center_in_product": _point_in_box(_bbox_center(box), product_box),
            "inside_keep": _touches_any(box, keep_boxes + text_boxes),
        }
        component.update(_component_features(component, h, w, image_area))
        components.append(component)

    if not components:
        return mask

    non_noise_components = [c for c in components if not c["border_like"] and not c["bottom_text_like"]]
    if not non_noise_components:
        non_noise_components = components
    main_area = max(c["area"] for c in non_noise_components)
    min_accessory_area = 6 if keep_small_accessories else max(25, int(image_area * 0.000035))

    kept: List[int] = []
    if product_type == "single_object":
        candidates = [c for c in non_noise_components if not c["text_like"] or c["inside_product"] or c["inside_keep"]]
        if analysis:
            inside_candidates = [c for c in candidates if c["inside_product"] or c["inside_keep"]]
            if inside_candidates and max(c["area"] for c in inside_candidates) >= image_area * 0.012:
                candidates = inside_candidates
        if not candidates:
            candidates = non_noise_components
        best = max(candidates, key=lambda c: c["area"])
        kept.append(best["label"])
        for c in candidates:
            if c["label"] == best["label"]:
                continue
            if c["inside_keep"] and c["area"] >= min_accessory_area and not c["bottom_text_like"]:
                kept.append(c["label"])
            elif keep_small_accessories and c["center_in_product"] and c["area"] >= max(8, int(main_area * 0.00008)) and not c["text_like"]:
                kept.append(c["label"])
    else:
        for c in components:
            if c["area"] < min_accessory_area:
                continue
            if c["border_like"] or c["bottom_text_like"]:
                continue
            # Outside Gemini product box can still be a missed accessory, so do
            # not remove it unless its shape is clearly text/line noise.
            outside_soft_product = analysis is not None and not (c["inside_product"] or c["inside_keep"])
            if outside_soft_product and (c["text_like"] or c["line_like"]):
                continue
            kept.append(c["label"])

    if not kept:
        kept = [max(non_noise_components, key=lambda c: c["area"])["label"]]

    out = np.isin(labels, list(dict.fromkeys(kept))).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out.astype(np.uint8)


def rgba_from_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    if mask.shape != arr.shape[:2]:
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Only alpha is replaced. RGB remains exactly the source image RGB.
    arr[:, :, 3] = mask.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def refine_alpha(rgba: Image.Image, analysis: Optional[Dict[str, Any]] = None, component_mode: str = "auto", keep_small_accessories: bool = True) -> Image.Image:
    rgba = rgba.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    _, binary = cv2.threshold(alpha, 12, 255, cv2.THRESH_BINARY)
    binary = remove_edge_border_lines(binary)
    binary = remove_bottom_disclaimer_like_noise(binary, analysis=analysis, aggressive=True)
    binary = clean_components(binary, analysis, component_mode=component_mode, keep_small_accessories=keep_small_accessories)
    if analysis and str(analysis.get("product_type", "")) == "single_object" and component_mode != "multi_part":
        binary = rebuild_single_object_mask(binary, analysis, keep_small_accessories=keep_small_accessories)
    elif component_mode == "single_part":
        binary = rebuild_single_object_mask(binary, analysis, keep_small_accessories=keep_small_accessories)
    binary = _remove_regions_from_mask(binary, analysis)
    binary = remove_edge_border_lines(binary)
    binary = remove_bottom_disclaimer_like_noise(binary, analysis=analysis, aggressive=True)
    # Feather alpha only. RGB is not altered.
    arr[:, :, 3] = cv2.GaussianBlur(binary, (3, 3), 0)
    return Image.fromarray(arr, mode="RGBA")


def plain_opencv_remove(image: Image.Image, *, threshold: int = 28, component_mode: str = "multi_part") -> Image.Image:
    mask = initial_white_bg_mask(image, threshold=threshold)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, analysis=None, aggressive=True)
    mode = "multi_part" if component_mode == "auto" else component_mode
    mask = clean_components(mask, None, component_mode=mode, keep_small_accessories=True)
    if mode == "single_part":
        mask = rebuild_single_object_mask(mask, None, keep_small_accessories=True)
    mask = remove_edge_border_lines(mask)
    mask = remove_bottom_disclaimer_like_noise(mask, analysis=None, aggressive=True)
    return rgba_from_mask(image, mask)
