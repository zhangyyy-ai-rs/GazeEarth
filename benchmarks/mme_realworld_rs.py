from __future__ import annotations

from .base import BaseDatasetAdapter, DatasetSample, resolve_path
from .common import first, load_records, parse_images, parse_options


class MMERealWorldRSAdapter(BaseDatasetAdapter):
    """Adapter for local exports of the Remote-Sensing subset of MME-RealWorld."""

    def __init__(self, annotation_path: str, root: str) -> None:
        self.records = load_records(annotation_path)
        self.root = root

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        for idx, item in enumerate(self.records):
            raw_images = first(item, "image_path", "image", "images", "image_paths", default=[])
            yield DatasetSample(
                sample_id=str(first(item, "id", "index", "question_id", default=idx)),
                image_paths=[resolve_path(self.root, x) for x in parse_images(raw_images)],
                question=str(first(item, "question", "text", "query", default="")).strip(),
                answer=str(first(item, "answer", "ground_truth", "label", default="")).strip(),
                options=parse_options(first(item, "options", "choices", "multi-choice options", default=[])),
                category=str(first(item, "category", "task", "sub_task", default="RS")),
                metadata={"domain": first(item, "domain", default="RS")},
            )
