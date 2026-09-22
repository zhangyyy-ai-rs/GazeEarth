from __future__ import annotations

from .base import BaseDatasetAdapter, DatasetSample, resolve_path
from .common import first, load_records, parse_images, parse_options


class XLRSBenchAdapter(BaseDatasetAdapter):
    """Robust local-file adapter for XLRS-Bench / XLRS-Bench-lite exports."""

    def __init__(self, annotation_path: str, root: str) -> None:
        self.records = load_records(annotation_path)
        self.root = root

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        for idx, item in enumerate(self.records):
            raw_images = first(item, "image_path", "image_paths", "image", "images", default=[])
            image_paths = [resolve_path(self.root, x) for x in parse_images(raw_images)]
            yield DatasetSample(
                sample_id=str(first(item, "index", "id", "question_id", default=idx)),
                image_paths=image_paths,
                question=str(first(item, "question", "text", "query", default="")).strip(),
                answer=str(first(item, "answer", "ground_truth", "label", default="")).strip(),
                options=parse_options(first(item, "multi-choice options", "options", "choices", default=[])),
                category=str(first(item, "l2-category", "category", "task", default="")),
                metadata={"raw_category": item.get("category", "")},
            )
