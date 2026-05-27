from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

import requests
from PIL import Image


class BackgroundRemovalError(Exception):
    pass


@dataclass
class BackgroundRemovalAPIClient:
    endpoint: str
    api_key: str
    auth_header_name: str = "X-Api-Key"
    image_field_name: str = "image_file"
    extra_form_fields: Optional[Dict[str, Any]] = None
    timeout: int = 120

    def remove_background(self, image_bytes: bytes, filename: str = "input.png") -> Image.Image:
        if not self.endpoint:
            raise BackgroundRemovalError("API endpoint is empty.")
        if not self.api_key:
            raise BackgroundRemovalError("API key is empty.")

        headers = {self.auth_header_name: self.api_key}
        files = {self.image_field_name: (filename, image_bytes)}
        data = self.extra_form_fields or {}

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackgroundRemovalError(f"API request failed: {exc}") from exc

        if response.status_code >= 400:
            body = response.text[:500]
            raise BackgroundRemovalError(f"API error {response.status_code}: {body}")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            raise BackgroundRemovalError(
                "API returned JSON instead of an image. If your provider returns image URLs in JSON, adapt core/api_bg_remove.py."
            )

        try:
            image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            return image
        except Exception as exc:
            raise BackgroundRemovalError(f"Could not decode API image response: {exc}") from exc
