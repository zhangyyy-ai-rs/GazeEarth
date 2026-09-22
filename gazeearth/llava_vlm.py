from __future__ import annotations

from typing import Sequence

from PIL import Image


class LlavaNextFrozenVLM:
    """Frozen Hugging Face LLaVA-NeXT/LLaVA-v1.6 adapter.

    The adapter owns only the model-native chat template and image processor.
    GazeEarth remains responsible for constructing the visual evidence canvas.
    One source image or multiple source images are passed in source order.
    """

    def __init__(
        self,
        model_path: str,
        *,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        dtype: str = "float16",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise ImportError("Install torch and transformers to use LlavaNextFrozenVLM") from exc

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = str(dtype).strip().lower()
        if key not in dtype_map:
            raise ValueError(f"Unsupported dtype={dtype!r}; choose from {sorted(dtype_map)}")

        self.torch = torch
        self.model_id = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
            local_files_only=bool(local_files_only),
        )

        candidates = []
        for name in (attn_implementation, "sdpa", "eager"):
            if name not in candidates:
                candidates.append(name)
        self.model = None
        last_error: Exception | None = None
        for name in candidates:
            try:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_id,
                    dtype=dtype_map[key],
                    attn_implementation=name,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=bool(local_files_only),
                )
                self.attn_implementation = name
                break
            except Exception as exc:  # pragma: no cover - hardware/model specific
                last_error = exc
        if self.model is None:
            raise RuntimeError(f"Unable to load LLaVA-v1.6 model: {last_error}")

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Frozen LLaVA model unexpectedly has trainable parameters")

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image] | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        images = list(images or [])
        if not images:
            raise ValueError("LLaVA-v1.6 requires at least one visual input")

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": str(prompt)}]
        messages = [{"role": "user", "content": content}]
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=formatted,
            images=images,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_length:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
