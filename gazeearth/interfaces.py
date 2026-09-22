from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image


class FrozenVLM(Protocol):
    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image] | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        ...
