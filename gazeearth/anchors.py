from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from .schemas import BBox, EvidenceRegion

_NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"

_SIZE_PATTERNS = (
    re.compile(
        r"(?:image\s+)?(?:resolution|size|dimensions?)\s*(?:is|are|=|:)?\s*"
        r"(?P<width>\d+)\s*[xX×]\s*(?P<height>\d+)",
        flags=re.I,
    ),
    re.compile(
        r"(?:\"|')?(?:image[_\s-]*)?(?:width|w)(?:\"|')?\s*[=:]\s*(?P<width>\d+)"
        r"[\s,;]+(?:\"|')?(?:image[_\s-]*)?(?:height|h)(?:\"|')?\s*[=:]\s*(?P<height>\d+)",
        flags=re.I,
    ),
)

_BOX_PATTERN = re.compile(
    rf"(?:\"|')?(?P<label>bounding\s*box|bbox|target\s*box|reference\s*box|"
    rf"region\s*coordinates?|target\s*coordinates?|reference\s*coordinates?|roi)"
    rf"(?:\"|')?\s*(?:is|are|=|:)?\s*[\[(]\s*"
    rf"(?P<x1>{_NUMBER})\s*[,;]\s*(?P<y1>{_NUMBER})\s*[,;]\s*"
    rf"(?P<x2>{_NUMBER})\s*[,;]\s*(?P<y2>{_NUMBER})\s*[\])]",
    flags=re.I,
)

_SOURCE_PATTERNS = (
    re.compile(r"\b(?:image|source)\s*(?:index|id)?\s*[#:=]?\s*(?P<index>\d+)\b", flags=re.I),
    re.compile(r"\bI(?P<index>\d+)\b", flags=re.I),
)


@dataclass(frozen=True)
class AnchorConfig:
    """One scale-normalized policy shared by every dataset and question."""

    enabled: bool = True
    context_scale: float = 4.0
    min_focus_fraction: float = 0.04
    max_focus_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.context_scale <= 0:
            raise ValueError("anchor.context_scale must be positive")
        if not 0 < self.min_focus_fraction <= self.max_focus_fraction <= 1:
            raise ValueError("Require 0 < min_focus_fraction <= max_focus_fraction <= 1")


@dataclass(frozen=True)
class SpatialHint:
    """A format-neutral spatial reference extracted only from the user query."""

    bbox: tuple[float, float, float, float]
    coordinate_mode: str
    declared_width: int | None = None
    declared_height: int | None = None
    source_index: int | None = None
    hint_format: str = "query_xyxy"


@dataclass(frozen=True)
class AnchorResolution:
    regions: tuple[EvidenceRegion, ...] = ()
    source: str = "none"
    reason: str = "not_present"
    declared_size: tuple[int, int] | None = None
    declared_bbox: tuple[float, float, float, float] | None = None
    mapped_bbox: BBox | None = None
    source_index: int | None = None
    coordinate_mode: str | None = None
    hint_format: str | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.regions)

    def to_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "source": self.source,
            "reason": self.reason,
            "declared_size": list(self.declared_size) if self.declared_size else None,
            "declared_bbox": list(self.declared_bbox) if self.declared_bbox else None,
            "mapped_bbox": list(self.mapped_bbox) if self.mapped_bbox else None,
            "source_index": self.source_index,
            "coordinate_mode": self.coordinate_mode,
            "hint_format": self.hint_format,
        }


def _parse_declared_size(question: str) -> tuple[int, int] | None:
    for pattern in _SIZE_PATTERNS:
        match = pattern.search(question)
        if match:
            width, height = int(match.group("width")), int(match.group("height"))
            if width > 0 and height > 0:
                return width, height
    return None


def _parse_source_index(question: str, box_start: int) -> int | None:
    # A source reference must be near the spatial expression. This avoids
    # treating unrelated numbers elsewhere in a long question as image IDs.
    context = question[max(0, box_start - 160) : box_start]
    candidates: list[tuple[int, int]] = []
    for pattern in _SOURCE_PATTERNS:
        for match in pattern.finditer(context):
            candidates.append((match.start(), int(match.group("index")) - 1))
    if not candidates:
        return None
    return max(candidates)[1]


def parse_spatial_hint(question: str) -> SpatialHint | None:
    """Extract a generic XYXY spatial hint from question text.

    Supported forms include resolution/size/dimensions, bbox/bounding box/ROI/
    target coordinates, either order, absolute or normalized coordinates, and
    optional image/source indices. A label is required, so arbitrary numeric
    lists in counting or route-planning questions are not interpreted as boxes.
    """

    text = str(question or "")
    box_match = _BOX_PATTERN.search(text)
    if not box_match:
        return None
    bbox = tuple(float(box_match.group(name)) for name in ("x1", "y1", "x2", "y2"))
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    normalized = all(0.0 <= value <= 1.0 for value in bbox)
    declared_size = _parse_declared_size(text)
    if not normalized and declared_size is None:
        # Absolute coordinates can still be interpreted against the actual
        # source dimensions, but only after the image is opened.
        coordinate_mode = "absolute_source_pixels"
    else:
        coordinate_mode = "normalized" if normalized else "absolute_declared_pixels"
    return SpatialHint(
        bbox=bbox,
        coordinate_mode=coordinate_mode,
        declared_width=declared_size[0] if declared_size else None,
        declared_height=declared_size[1] if declared_size else None,
        source_index=_parse_source_index(text, box_match.start()),
        hint_format=f"query_{box_match.group('label').casefold().replace(' ', '_')}_xyxy",
    )


def map_anchor_to_image(anchor: SpatialHint, image: Image.Image) -> BBox:
    if anchor.coordinate_mode == "normalized":
        sx, sy = image.width, image.height
    elif anchor.declared_width and anchor.declared_height:
        sx = image.width / anchor.declared_width
        sy = image.height / anchor.declared_height
    else:
        sx = sy = 1.0
    x1, y1, x2, y2 = anchor.bbox
    mapped = (
        max(0, min(image.width - 1, round(x1 * sx))),
        max(0, min(image.height - 1, round(y1 * sy))),
        max(1, min(image.width, round(x2 * sx))),
        max(1, min(image.height, round(y2 * sy))),
    )
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("Question-provided bounding box maps to an empty image region")
    return mapped


def _square_box(center_x: float, center_y: float, side: int, width: int, height: int) -> BBox:
    side = max(1, min(int(side), width, height))
    x1 = max(0, min(round(center_x - side / 2), width - side))
    y1 = max(0, min(round(center_y - side / 2), height - side))
    return (x1, y1, x1 + side, y1 + side)


def contextual_focus_box(target: BBox, image: Image.Image, config: AnchorConfig) -> BBox:
    """Return a context-expanded focus box; this function does not crop pixels."""
    x1, y1, x2, y2 = target
    target_long = max(x2 - x1, y2 - y1)
    image_long = max(image.size)
    lower = max(target_long, round(config.min_focus_fraction * image_long))
    upper = max(lower, round(config.max_focus_fraction * image_long))
    side = max(lower, min(round(config.context_scale * target_long), upper))
    return _square_box((x1 + x2) / 2, (y1 + y2) / 2, side, image.width, image.height)


def resolve_question_anchor(
    question: str,
    images: Sequence[Image.Image],
    config: AnchorConfig,
) -> AnchorResolution:
    """Resolve query-supplied spatial evidence into the common EvidenceRegion type."""

    if not config.enabled:
        return AnchorResolution(reason="disabled")
    hint = parse_spatial_hint(question)
    if hint is None:
        return AnchorResolution(reason="not_present")
    if hint.source_index is None:
        if len(images) != 1:
            return AnchorResolution(reason="ambiguous_source_count")
        source_index = 0
    else:
        source_index = hint.source_index
        if source_index < 0 or source_index >= len(images):
            return AnchorResolution(reason="invalid_source_index", source_index=source_index)
    try:
        mapped = map_anchor_to_image(hint, images[source_index])
        focus_box = contextual_focus_box(mapped, images[source_index], config)
    except ValueError:
        return AnchorResolution(reason="invalid_mapping", source_index=source_index)
    region_id = f"I{source_index + 1}:R1"
    region = EvidenceRegion(
        id=region_id,
        source_index=source_index,
        source_label=f"I{source_index + 1}",
        cell_ids=("R1",),
        bbox=focus_box,
    )
    declared_size = (
        (hint.declared_width, hint.declared_height)
        if hint.declared_width is not None and hint.declared_height is not None
        else None
    )
    return AnchorResolution(
        regions=(region,),
        source="question_coordinates",
        reason="resolved",
        declared_size=declared_size,
        declared_bbox=hint.bbox,
        mapped_bbox=mapped,
        source_index=source_index,
        coordinate_mode=hint.coordinate_mode,
        hint_format=hint.hint_format,
    )
