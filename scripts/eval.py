#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _focus_config(cfg: dict) -> dict:
    """Read the paper's focus configuration."""

    return dict(cfg.get("focus", {}) or {})


def main() -> None:
    parser = argparse.ArgumentParser(description="GazeEarth evaluation")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--model-config", default=None, help="Optional YAML with a model mapping")
    parser.add_argument("--method", choices=("direct", "gazeearth", "overview"), default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue a verified JSONL prefix")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace an existing run")
    args = parser.parse_args()
    if args.limit < -1 or args.limit == 0:
        parser.error("--limit must be -1 (all samples) or a positive integer")

    from benchmarks.factory import build_dataset
    from evaluation.metrics import CHOICE_PROTOCOL, build_scorer, multiple_choice_match
    from evaluation.runner import EvaluationRunner, preflight_output_dir
    from gazeearth.anchors import AnchorConfig
    from gazeearth.config import load_config
    from gazeearth.model_factory import build_vlm
    from gazeearth.pipeline import ANSWER_PROTOCOL_ID, STAGES, FocusConfig, GazeEarthPipeline
    from gazeearth.selector import FixedGridSelector, SelectorConfig
    from gazeearth.topology_focus import PROTOCOL_ID

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.model_config:
        model_config = load_config(Path(args.model_config).resolve())
        if not isinstance(model_config.get("model"), dict):
            raise ValueError("--model-config must contain a model mapping")
        cfg["model"] = model_config["model"]
    method = args.method or cfg.get("method", "gazeearth")
    output_dir = Path(args.output or cfg.get("output_dir", f"outputs/{method}"))
    output = preflight_output_dir(output_dir, resume=args.resume, overwrite=args.overwrite)

    focus_cfg = FocusConfig(**_focus_config(cfg))
    selector_cfg = SelectorConfig(**cfg.get("selector", {}))
    anchor_cfg = AnchorConfig(**cfg.get("anchor", {}))
    if method not in GazeEarthPipeline.VALID_METHODS:
        raise ValueError(f"Unknown method: {method}")
    dataset_cfg = cfg["dataset"]
    dataset = list(build_dataset(dataset_cfg["name"], dataset_cfg["annotation_path"], dataset_cfg["root"]))
    if not dataset:
        raise ValueError("No evaluation samples found")
    scorer, scoring_protocol = build_scorer(dataset_cfg["name"])
    requested = dataset if args.limit < 0 else dataset[:args.limit]
    for sample in requested:
        if not sample.question or not sample.answer or not sample.image_paths:
            raise ValueError(f"Incomplete annotation for {sample.sample_id}")
        for image_path in sample.image_paths:
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
        if scoring_protocol == CHOICE_PROTOCOL:
            multiple_choice_match(sample.answer, sample.answer, sample.options)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    answer_template = GazeEarthPipeline.answer_prompt("{question}")
    selector_template = FixedGridSelector(None, selector_cfg).prompt("{question}", 1)
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("gazeearth", "benchmarks", "evaluation", "scripts")
        for path in sorted((ROOT / folder).glob("*.py"))
    }
    runtime_versions = {}
    for package in ("torch", "torchvision", "transformers", "tokenizers", "qwen-vl-utils",
                    "Pillow", "numpy", "nltk", "PyYAML", "accelerate"):
        try:
            runtime_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[package] = "not_installed"
    manifest = {
        "project": "GazeEarth",
        "release_protocol": "gazeearth-paper-release",
        "stages": list(STAGES),
        "seed": seed,
        "scoring_protocol": scoring_protocol,
        "annotation_sha256": hashlib.sha256(Path(dataset_cfg["annotation_path"]).read_bytes()).hexdigest(),
        "answer_prompt_template": answer_template,
        "answer_prompt_sha256": hashlib.sha256(answer_template.encode()).hexdigest(),
        "selector_prompt_template_single_source": selector_template,
        "selector_prompt_sha256": hashlib.sha256(selector_template.encode()).hexdigest(),
        "source_sha256": source_hashes,
        "runtime_versions": runtime_versions,
        "method": method,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_config_path": str(Path(args.model_config).resolve()) if args.model_config else None,
        "model_config_sha256": (
            hashlib.sha256(Path(args.model_config).read_bytes()).hexdigest() if args.model_config else None
        ),
        "limit": args.limit,
        "model": cfg.get("model", {}),
        "selector": selector_cfg.__dict__,
        "focus": focus_cfg.__dict__,
        "anchor": anchor_cfg.__dict__,
        "dataset": cfg.get("dataset", {}),
        "protocol": PROTOCOL_ID,
        "answer_protocol": ANSWER_PROTOCOL_ID,
        "training_free": True,
        "dataset_specific_inference_branch": False,
    }
    manifest_path = output / "run_manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("--resume requires run_manifest.json from the original run")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "release_protocol",
            "seed",
            "scoring_protocol",
            "annotation_sha256",
            "answer_prompt_sha256",
            "selector_prompt_sha256",
            "source_sha256",
            "runtime_versions",
            "model",
            "method",
            "protocol",
            "answer_protocol",
            "config_sha256",
            "model_config_sha256",
            "limit",
            "selector",
            "focus",
            "anchor",
            "dataset",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"resume manifest mismatch for {key!r}; use the original command/config")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Only load the large model after every cheap preflight check passes.
    if cfg["model"].get("backend", "qwen") == "qwen":
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    vlm = build_vlm(cfg["model"])
    pipeline = GazeEarthPipeline(
        vlm=vlm,
        selector=FixedGridSelector(vlm, selector_cfg),
        focus_config=focus_cfg,
        anchor_config=anchor_cfg,
        method=method,
    )
    summary = EvaluationRunner(
        pipeline,
        dataset,
        str(output),
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        scorer=scorer,
        scoring_protocol=scoring_protocol,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
