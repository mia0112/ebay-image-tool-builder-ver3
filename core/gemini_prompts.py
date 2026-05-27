from __future__ import annotations

GEMINI_ANALYSIS_PROMPT = """
You are analyzing an eBay auto-parts product photo before computer-vision background removal.
Return STRICT JSON only. No markdown, no comments.

Coordinate system:
- Return all boxes in pixels relative to the provided image.
- bbox format is [x, y, width, height].

Task:
1. Identify the real product object and all real accessories/parts that must be kept.
2. Identify background text, bottom disclaimer text, watermark, overlay text, thin grey listing border lines, or unrelated objects that must be removed.
3. Do NOT remove embossed, engraved, printed, stamped, or physical markings on the actual product.
4. Do NOT remove small real accessories such as screws, clips, washers, springs, rubber boots, brackets, pins, bolts, caps.
5. product_bbox must contain the FULL visible product/kit, including all accessories. Do not return a tight bbox around only the central/main part.
6. If the product is a kit with multiple separate parts, set product_type to "multi_part_object".
7. If the image contains one main connected object, set product_type to "single_object".

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
