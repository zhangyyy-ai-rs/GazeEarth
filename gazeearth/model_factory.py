from __future__ import annotations

import importlib
from typing import Any

from .vlm import QwenFrozenVLM


def resolve_model_path(config: dict[str, Any]) -> str:
    model_path = config.get("path")
    if not model_path:
        raise ValueError("Missing model path. Set model.path in the YAML config.")
    return str(model_path)


def build_vlm(config: dict[str, Any]):
    """Build one of the frozen backbones used in the paper."""
    backend = str(config.get("backend", "qwen")).strip().lower()
    if backend == "qwen":
        return QwenFrozenVLM(
            resolve_model_path(config),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            max_new_tokens=config.get("max_new_tokens", 128),
            temperature=config.get("temperature", 0.0),
        )
    if backend in {"llava", "llava_next", "llava-v1.6"}:
        from .llava_vlm import LlavaNextFrozenVLM

        return LlavaNextFrozenVLM(
            resolve_model_path(config),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            max_new_tokens=config.get("max_new_tokens", 1024),
            temperature=config.get("temperature", 0.0),
            dtype=config.get("dtype", "float16"),
            local_files_only=bool(config.get("local_files_only", False)),
        )
    if backend in {"intern_s1", "intern-s1", "interns1"}:
        from .intern_s1_vlm import InternS1FrozenVLM

        return InternS1FrozenVLM(
            resolve_model_path(config),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            max_new_tokens=config.get("max_new_tokens", 1024),
            temperature=config.get("temperature", 0.0),
            enable_thinking=bool(config.get("enable_thinking", False)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
    if backend in {"openrouter", "gpt4o"}:
        from .openrouter_vlm import OpenRouterFrozenVLM

        return OpenRouterFrozenVLM(
            model_name=str(config.get("model_name", "openai/gpt-4o-2024-11-20")),
            api_key_env=str(config.get("api_key_env", "OPENROUTER_API_KEY")),
            max_new_tokens=int(config.get("max_new_tokens", 1024)),
            temperature=float(config.get("temperature", 0.0)),
            image_detail=str(config.get("image_detail", "high")),
        )
    if backend == "custom":
        target = str(config.get("class_path", ""))
        if ":" not in target:
            raise ValueError("custom model.class_path must be formatted as module:Class")
        module_name, class_name = target.split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**dict(config.get("kwargs", {})))
    raise ValueError(f"Unsupported model backend {backend!r}; use qwen, llava, intern_s1, openrouter, or custom")
