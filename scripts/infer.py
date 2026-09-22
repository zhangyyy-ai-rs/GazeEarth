#!/usr/bin/env python3
"""Run GazeEarth on one question without benchmark annotations."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="GazeEarth single-question inference")
    parser.add_argument("--config", default="configs/xlrs_bench.yaml")
    parser.add_argument("--model-config", default=None, help="Optional YAML with a model mapping")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", default=[])
    parser.add_argument("--method", choices=("direct", "gazeearth", "overview"), default=None)
    args = parser.parse_args()

    import numpy as np

    from benchmarks.base import DatasetSample
    from gazeearth import AnchorConfig, FixedGridSelector, FocusConfig, GazeEarthPipeline, SelectorConfig
    from gazeearth.config import load_config
    from gazeearth.model_factory import build_vlm

    for path in args.images:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    cfg = load_config(args.config)
    if args.model_config:
        model_config = load_config(args.model_config)
        if not isinstance(model_config.get("model"), dict):
            raise ValueError("--model-config must contain a model mapping")
        cfg["model"] = model_config["model"]
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    if cfg["model"].get("backend", "qwen") == "qwen":
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    vlm = build_vlm(cfg["model"])
    pipeline = GazeEarthPipeline(
        vlm=vlm,
        selector=FixedGridSelector(vlm, SelectorConfig(**cfg.get("selector", {}))),
        focus_config=FocusConfig(**cfg.get("focus", {})),
        anchor_config=AnchorConfig(**cfg.get("anchor", {})),
        method=args.method or cfg.get("method", "gazeearth"),
    )
    sample = DatasetSample("example", args.images, args.question, options=args.options)
    result = pipeline.infer(args.images, args.question, sample.prompt_question)
    print(json.dumps(result.trace_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
