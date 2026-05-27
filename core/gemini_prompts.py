from __future__ import annotations

GEMINI_ANALYSIS_PROMPT = """
You are analyzing an eBay auto-parts product photo before computer-vision background removal.
Return STRICT JSON only. No markdown, no comments.

Coordinate system:
- Return all boxes in pixels relative to the provided image.
- bbox format is [x, y, width, height].

Task:
1. Identify the real product object and all real accessories/parts that must be kept.
2. Identify ALL background text, bottom disclaimer text, watermark, overlay text, thin grey/black listing border lines, edge slivers, or unrelated objects that must be removed. Pay special attention to the bottom 35% region and the outer 8% edge band.
3. Do NOT remove embossed, engraved, printed, stamped, or physical markings on the actual product.
4. Do NOT remove small real accessories such as screws, clips, washers, springs, rubber boots, brackets, pins, bolts, caps.
5. product_bbox must contain the FULL visible product/kit, including all accessories. Do not return a tight bbox around only the central/main part. product_bbox is a soft hint only, never a hard crop.
6. If the product is a kit with multiple separate parts, set product_type to "multi_part_object".
7. If the image contains one main connected object, set product_type to "single_object". If several small words form one disclaimer sentence, include boxes that cover the whole line. If a thin border line runs near an edge, include a rectangle covering that visible line segment.

Return this JSON object:
{
  "product_type": "single_object | multi_part_object",
  "product_bbox": [x, y, width, height],
  "keep_regions": [
    {"bbox": [x, y, width, height], "reason": "main_product | accessory | screw | clip | washer | spring | bracket | bolt | rubber_boot | other_product_part"}
  ],
  "remove_regions": [
    {"bbox": [x, y, width, height], "reason": "background_text | disclaimer_text | watermark | overlay_text | unrelated_object"}
  ],
  "text_on_product": [
    {"bbox": [x, y, width, height], "should_keep": true, "reason": "real_product_marking"}
  ],
  "background_type": "plain | gradient | complex | unknown",
  "confidence": 0.0
}
""".strip()

GEMINI_QA_PROMPT = """
Check this final eBay framed product image.
Return STRICT JSON only. No markdown.

Pass only if:
- no unwanted source-image background text, bottom disclaimer, watermark, listing border line, or overlay text remains inside the white product area;
- product is not cut off by the background-removal/cropping pipeline;
- frame graphics are not covering the product;
- small accessories still appear when the original image is a multi-part kit.

Important: Ignore text/graphics that are part of the fixed selling frame itself, such as logo, BEST QUALITY ribbon, FREESHIP voucher, eBay badge, and warranty banner.

Return:
{
  "pass": true,
  "issues": [
    {"type": "leftover_text | product_cut_off | frame_overlap | missing_accessory | bad_crop | other", "bbox": [x, y, width, height], "severity": "low | medium | high", "note": "short note"}
  ],
  "recommend_retry": false
}
""".strip()

GEMINI_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "product_type": {"type": "string", "enum": ["single_object", "multi_part_object"]},
        "product_bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "keep_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "reason": {"type": "string"},
                },
                "required": ["bbox", "reason"],
            },
        },
        "remove_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "reason": {"type": "string"},
                },
                "required": ["bbox", "reason"],
            },
        },
        "text_on_product": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "should_keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["bbox", "should_keep", "reason"],
            },
        },
        "background_type": {"type": "string", "enum": ["plain", "gradient", "complex", "unknown"]},
        "confidence": {"type": "number"},
    },
    "required": ["product_type", "product_bbox", "keep_regions", "remove_regions", "text_on_product", "background_type", "confidence"],
}

GEMINI_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "severity": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["type", "bbox", "severity", "note"],
            },
        },
        "recommend_retry": {"type": "boolean"},
    },
    "required": ["pass", "issues", "recommend_retry"],
}


GEMINI_ANALYSIS_PROMPT_STRICT_SWEEP = """
You are doing a STRICT CLEANUP SWEEP for an eBay auto-parts source photo before OpenCV alpha-mask cleanup.
Return STRICT JSON only. No markdown, no comments.

Coordinate system:
- Return all boxes in pixels relative to the provided image.
- bbox format is [x, y, width, height].

Primary goal:
Find ALL non-product text/lines that must be removed from alpha. Missing a small disclaimer or border is worse than slightly over-marking blank background.

Rules:
1. product_bbox must contain the FULL visible product / kit, including all real accessories and all extremities. It is a soft hint only, not a crop.
2. keep_regions must include every real product part, even tiny screws, clips, washers, springs, boots, bolts, brackets, pins, caps, seals.
3. remove_regions must include ALL non-product artifacts such as:
   - bottom disclaimer text
   - faint grey / black text near the bottom
   - tiny text lines under the product
   - watermark text
   - overlay text
   - thin listing border lines or slivers on the left / right / top / bottom margins
   - isolated black/grey border remnants touching an image edge
4. VERY IMPORTANT: inspect carefully the bottom 35% of the image and the outer 8% edge band on all four sides. If you see any faint text line, tiny disclaimer, or thin border sliver there, return it in remove_regions.
5. If several small words form one disclaimer sentence, return one or more remove_regions that fully cover the whole sentence / line.
6. If a thin border line runs near an image edge, return a long rectangle covering that entire visible line segment.
7. Do NOT mark embossed, engraved, printed, stamped, or molded markings that are physically on the real product surface.
8. If uncertain whether a tiny dark mark in the lower blank area is product or background text, prefer marking it as remove_regions unless it clearly touches the product body.
9. If the image contains one main connected object, set product_type to "single_object". If it is a kit with multiple separated parts, set product_type to "multi_part_object".

Return this JSON object:
{
  "product_type": "single_object | multi_part_object",
  "product_bbox": [x, y, width, height],
  "keep_regions": [
    {"bbox": [x, y, width, height], "reason": "main_product | accessory | screw | clip | washer | spring | bracket | bolt | rubber_boot | other_product_part"}
  ],
  "remove_regions": [
    {"bbox": [x, y, width, height], "reason": "background_text | disclaimer_text | watermark | overlay_text | listing_border | unrelated_object"}
  ],
  "text_on_product": [
    {"bbox": [x, y, width, height], "should_keep": true, "reason": "real_product_marking"}
  ],
  "background_type": "plain | gradient | complex | unknown",
  "confidence": 0.0
}
""".strip()
