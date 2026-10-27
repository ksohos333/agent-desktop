"""OpenRouter API client for cheap vision/planning inference."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
from PIL import Image


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VISION_MODEL = "openrouter/google/gemini-2.5-flash-preview"
TIMEOUT_SECONDS = 120


class OpenRouterError(Exception):
    """Raised when an OpenRouter request fails or returns an error."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class OpenRouterClient:
    """Minimal OpenAI-compatible client for OpenRouter.

    Supports text and vision (image + prompt) chat completions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        vision_model: str = DEFAULT_VISION_MODEL,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.vision_model = vision_model
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY not configured")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> tuple[str, Usage]:
        """Run a chat completion and return (text, usage)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "desktop-mcp-vision",
        }
        payload: dict[str, Any] = {
            "model": model or self.vision_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenRouterError(f"OpenRouter HTTP {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.RequestError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        data = resp.json()
        if data.get("error"):
            raise OpenRouterError(f"OpenRouter error: {data['error']}")
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        tokens = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=payload["model"],
        )
        return content, tokens

    def visual_query(
        self,
        image_base64: str,
        prompt: str,
        mime_type: str = "image/png",
        model: str | None = None,
        system: str | None = None,
    ) -> tuple[str, Usage]:
        """Send a base64 image with a prompt to a vision model."""
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}",
                    "detail": "low",
                },
            }
        )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat(messages, model=model)


# ---------------------------------------------------------------------------
# Cost-control helpers
# ---------------------------------------------------------------------------

def resize_image(
    image_base64: str,
    max_width: int = 1280,
    max_height: int = 1280,
    max_bytes: int = 1_500_000,
    mime_type: str = "image/png",
) -> tuple[str, dict[str, int | str]]:
    """Resize and/or re-encode an image to keep vision costs bounded.

    Returns the resized base64 string plus metadata about what changed.
    """
    data = base64.b64decode(image_base64)
    original_len = len(data)
    img = Image.open(BytesIO(data))
    original_size = img.size

    if original_len <= max_bytes and img.width <= max_width and img.height <= max_height:
        return image_base64, {
            "action": "unchanged",
            "original_bytes": original_len,
            "final_bytes": original_len,
            "size": f"{img.width}x{img.height}",
        }

    # Downscale preserving aspect ratio
    ratio = min(max_width / img.width, max_height / img.height, 1.0)
    if ratio < 1.0:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    fmt = "PNG" if mime_type == "image/png" else "JPEG"
    quality = 85
    for _ in range(5):
        buf = BytesIO()
        if fmt == "JPEG":
            img.save(buf, format=fmt, quality=quality, optimize=True)
        else:
            img.save(buf, format=fmt, optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            break
        # Reduce size again
        quality = max(30, quality - 15)
        ratio *= 0.75
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    final_b64 = base64.b64encode(out).decode("ascii")
    return final_b64, {
        "action": "resized",
        "original_bytes": original_len,
        "final_bytes": len(out),
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "final_size": f"{img.width}x{img.height}",
    }
