from __future__ import annotations

from typing import Sequence

from PIL import Image


class InternS1FrozenVLM:
    """Frozen Intern-S1-mini adapter with the same generate contract as Qwen.

    Thinking mode follows the model configuration; the paper's two prompts and
    image construction remain shared across backbones.
    """

    def __init__(
        self,
        model_path: str,
        *,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise ImportError(
                "Intern-S1-mini requires torch and transformers>=4.55.2"
            ) from exc

        self.torch = torch
        self.model_id = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.enable_thinking = bool(enable_thinking)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
        candidates = []
        for name in (attn_implementation, "sdpa", "eager", None):
            if name not in candidates:
                candidates.append(name)
        self.model = None
        last_error: Exception | None = None
        for name in candidates:
            kwargs = dict(load_kwargs)
            if name is not None:
                kwargs["attn_implementation"] = name
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                self.attn_implementation = name or "model_default"
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
        if self.model is None:
            raise RuntimeError(f"Unable to load Intern-S1-mini: {last_error}")

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Frozen Intern-S1-mini unexpectedly has trainable parameters")

    def _input_device(self):
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        raise RuntimeError("Intern-S1-mini has no materialized parameter device")

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image] | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        content = [
            {"type": "image", "image": image.convert("RGB")}
            for image in list(images or [])
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Intern-S1's multimodal processor treats arbitrary template variables
        # as processor kwargs in some Transformers releases and silently drops
        # ``enable_thinking``. Render with the underlying tokenizer, whose Jinja
        # template owns that variable, then let the processor bind the same PIL
        # images to the emitted <IMG_CONTEXT> placeholders.
        rendered = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        image_list = [image.convert("RGB") for image in list(images or [])]
        inputs = self.processor(
            text=[rendered],
            images=image_list or None,
            padding=True,
            return_tensors="pt",
        )

        device = self._input_device()
        for key, value in list(inputs.items()):
            if not hasattr(value, "to"):
                continue
            value = value.to(device)
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=self.torch.bfloat16)
            inputs[key] = value

        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": do_sample,
        }
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = int(pad_token_id)
        # Some Intern-S1-mini remote-code revisions dereference
        # ``cache_position[0]`` while newer Transformers releases may leave the
        # argument as None for custom models. Supplying the canonical prefill
        # positions is an interface-compatibility fix; it does not change the
        # prompt, pixels, decoding policy, or generated token budget.
        generation_kwargs["cache_position"] = self.torch.arange(
            int(inputs["input_ids"].shape[1]),
            device=inputs["input_ids"].device,
            dtype=self.torch.long,
        )
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        input_length = int(inputs["input_ids"].shape[1])
        generated = generated[:, input_length:]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""
