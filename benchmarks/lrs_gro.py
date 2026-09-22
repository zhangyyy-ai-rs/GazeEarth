from __future__ import annotations

from .base import BaseDatasetAdapter, DatasetSample, resolve_path
from .common import first, load_records


class LRSGROAdapter(BaseDatasetAdapter):
    """Read the official LRS-GRO test split without exposing ROI annotations.

    The benchmark contains ``bbox``, ``cut``, ``label`` and reasoning fields for
    supervision and retrospective localization evaluation.  They are retained
    only as audit metadata and never enter ``question`` or the GazeEarth
    inference pipeline.
    """

    def __init__(self, annotation_path: str, root: str) -> None:
        records = load_records(annotation_path)
        self.records = [item for item in records if str(item.get("split", "test")).lower() == "test"]
        if not self.records:
            raise ValueError(f"No LRS-GRO test records found in {annotation_path}")
        self.root = root

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        for idx, item in enumerate(self.records):
            image_rel = str(first(item, "image_name", "image", "image_path", default=""))
            yield DatasetSample(
                sample_id=str(first(item, "question_id", "id", "index", default=idx)),
                image_paths=[resolve_path(self.root, image_rel)],
                question=str(first(item, "question", "text", "query", default="")).strip(),
                answer=str(first(item, "ground_truth", "answer", "label_answer", default="")).strip(),
                category=str(first(item, "category", default="")),
                metadata={
                    "split": str(item.get("split", "test")),
                    "spatial_level": str(item.get("type", "")),
                    # Audit-only fields. The runner never passes metadata to the model.
                    "cut": bool(item.get("cut", False)),
                    "gt_bbox": item.get("bbox", []),
                },
            )
