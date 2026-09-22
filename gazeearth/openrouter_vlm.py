from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


def _image_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    media_type = "image/png"
    if len(payload) > 29_000_000:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        payload = buffer.getvalue()
        media_type = "image/jpeg"
    return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"


class OpenRouterFrozenVLM:
    """GPT-4o through OpenRouter's Chat Completions interface."""

    def __init__(
        self,
        *,
        model_name: str = "openai/gpt-4o-2024-11-20",
        api_key_env: str = "OPENROUTER_API_KEY",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        image_detail: str = "high",
        timeout_sec: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("image_detail must be auto, low, or high")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Set {api_key_env} before using the OpenRouter backend")
        self.api_key = api_key
        self.model_id = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.image_detail = image_detail
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(0, int(max_retries))

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image] | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": _image_data_url(image), "detail": self.image_detail}}
            for image in (images or [])
        ]
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": int(max_new_tokens or self.max_new_tokens),
            }
        ).encode("utf-8")
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    result = json.load(response)
                message = result["choices"][0]["message"]["content"]
                if isinstance(message, str):
                    return message
                if isinstance(message, list):
                    return "\n".join(str(part.get("text", "")) for part in message if isinstance(part, dict))
                raise RuntimeError("OpenRouter response has no text content")
            except (HTTPError, URLError) as exc:
                retryable = not isinstance(exc, HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")
