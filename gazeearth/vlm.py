from __future__ import annotations

from typing import Sequence

from PIL import Image


class QwenFrozenVLM:
    """Frozen multimodal generator. No training path is exposed by this wrapper."""

    def __init__(
        self,
        model_path: str,
        *,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise ImportError("Install torch and transformers to use QwenFrozenVLM") from exc

        self.torch = torch
        self.model_id = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
        candidates = []
        for name in (attn_implementation, "sdpa", "eager"):
            if name not in candidates:
                candidates.append(name)
        last_error = None
        self.model = None
        for name in candidates:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    attn_implementation=name,
                    device_map="auto",
                )
                self.attn_implementation = name
                break
            except Exception as exc:  # pragma: no cover - hardware/model specific
                last_error = exc
        if self.model is None:
            raise RuntimeError(f"Unable to load reasoning model: {last_error}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Frozen reasoning model unexpectedly has trainable parameters")

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image] | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        images = list(images or [])
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [[{"role": "user", "content": content}]]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        if images:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
