from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from .gemini_prompts import (
    GEMINI_ANALYSIS_PROMPT,
    GEMINI_ANALYSIS_PROMPT_STRICT_SWEEP,
    GEMINI_ANALYSIS_SCHEMA,
    GEMINI_QA_PROMPT,
    GEMINI_QA_SCHEMA,
)


class GeminiVisionError(Exception):
    pass


@dataclass
class GeminiVisionClient:
    api_key: str
    model: str = "gemini-3.1-flash-lite"
    retry_model: str = "gemini-3-flash-preview"
    timeout_seconds: int = 90

    def __post_init__(self) -> None:
        if not self.api_key:
            raise GeminiVisionError("Gemini API key is empty.")
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise GeminiVisionError(
                "Package google-genai is not installed. Run: pip install google-genai"
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def api_key_from_env_or_value(value: str = "") -> str:
        value = (value or "").strip()
        if value and value != "PASTE_YOUR_GEMINI_API_KEY_HERE":
            return value
        return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    @staticmethod
    def image_to_jpeg_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)
        try:
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            return str(response)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise GeminiVisionError(f"Gemini did not return JSON: {text[:300]}")
            return json.loads(match.group(0))

    def _generate_json(self, *, image: Image.Image, prompt: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
        try:
            from google.genai import types  # type: ignore
        except Exception as exc:
            raise GeminiVisionError("google-genai types import failed.") from exc

        image_bytes = self.image_to_jpeg_bytes(image)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Prefer structured output. If a future SDK rejects JSON Schema in this exact shape,
        # fall back to JSON mime type without schema.
        try:
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = self._client.models.generate_content(
                model=model,
                contents=[prompt, image_part],
                config=config,
            )
        except Exception:
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            )
            response = self._client.models.generate_content(
                model=model,
                contents=[prompt, image_part],
                config=config,
            )

        return self._parse_json(self._extract_text(response))

    def analyze(self, preview_image: Image.Image) -> Tuple[Dict[str, Any], str]:
        data = self._generate_json(
            image=preview_image,
            prompt=GEMINI_ANALYSIS_PROMPT,
            schema=GEMINI_ANALYSIS_SCHEMA,
            model=self.model,
        )
        return data, self.model


    def analyze_strict_sweep(self, preview_image: Image.Image) -> Tuple[Dict[str, Any], str]:
        data = self._generate_json(
            image=preview_image,
            prompt=GEMINI_ANALYSIS_PROMPT_STRICT_SWEEP,
            schema=GEMINI_ANALYSIS_SCHEMA,
            model=self.model,
        )
        return data, self.model

    def analyze_with_retry(self, preview_image: Image.Image, threshold: float = 0.65, retry_on_low_confidence: bool = True) -> Tuple[Dict[str, Any], str]:
        data, used_model = self.analyze(preview_image)
        confidence = float(data.get("confidence") or 0.0)
        if retry_on_low_confidence and confidence < threshold and self.retry_model and self.retry_model != self.model:
            retry_data = self._generate_json(
                image=preview_image,
                prompt=GEMINI_ANALYSIS_PROMPT,
                schema=GEMINI_ANALYSIS_SCHEMA,
                model=self.retry_model,
            )
            retry_confidence = float(retry_data.get("confidence") or 0.0)
            if retry_confidence >= confidence:
                return retry_data, self.retry_model
        return data, used_model

    def qa(self, final_image: Image.Image) -> Dict[str, Any]:
        return self._generate_json(
            image=final_image,
            prompt=GEMINI_QA_PROMPT,
            schema=GEMINI_QA_SCHEMA,
            model=self.model,
        )
