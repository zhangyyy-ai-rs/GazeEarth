from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class SelectionResult:
    """The complete output of the frozen, single-step selector."""

    status: str
    region_ids: tuple[str, ...] = ()
    raw_output: str = ""
    parse_success: bool = True
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["region_ids"] = list(self.region_ids)
        return out


@dataclass(frozen=True)
class EvidenceRegion:
    """A deterministic source-space region derived from selected evidence."""

    id: str
    source_index: int
    source_label: str
    cell_ids: tuple[str, ...]
    bbox: BBox

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cell_ids"] = list(self.cell_ids)
        out["bbox"] = list(self.bbox)
        return out


@dataclass
class InferenceResult:
    answer: str
    raw_answer: str
    method: str
    timings: dict[str, float]
    selection: SelectionResult | None = None
    regions: list[EvidenceRegion] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def trace_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "method": self.method,
            "timings": self.timings,
            "selection": self.selection.to_dict() if self.selection else None,
            "regions": [region.to_dict() for region in self.regions],
            "extras": self.extras,
        }
