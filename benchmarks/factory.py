from __future__ import annotations

from .lrs_gro import LRSGROAdapter
from .mme_realworld_rs import MMERealWorldRSAdapter
from .xlrs_bench import XLRSBenchAdapter


def build_dataset(name: str, annotation_path: str, root: str):
    key = name.strip().lower().replace("_", "-")
    if key in {"lrs-gro", "lrsgro"}:
        return LRSGROAdapter(annotation_path, root)
    if key in {"xlrs-bench", "xlrs", "xlrs-bench-lite"}:
        return XLRSBenchAdapter(annotation_path, root)
    if key in {"mme-realworld-rs", "mme-rs", "mme-realworld"}:
        return MMERealWorldRSAdapter(annotation_path, root)
    raise ValueError(f"Unknown dataset adapter: {name}")
