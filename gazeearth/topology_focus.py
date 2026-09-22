from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from .image_ops import build_overview_views, resize_long_side
from .schemas import BBox, EvidenceRegion

PROTOCOL_ID = "gazeearth-source-resolved-topology-focus"
DEFAULT_FOCUS_SCALE = 1.5
DEFAULT_MAX_FOCUS_FRACTION = 0.55


def _union(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("cannot form a focus union from no boxes")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _expanded_interval(
    start: int,
    end: int,
    length: int,
    *,
    focus_scale: float,
    max_focus_fraction: float,
) -> tuple[int, int]:
    """Expand one interval while keeping the image endpoints fixed.

    The selected interval is enlarged around its original center whenever
    possible. Near an image boundary it is shifted only as far as required to
    remain inside the canvas. The mapping remains monotonic and continuous.
    """

    if not 0 <= start < end <= length:
        raise ValueError(f"invalid interval {(start, end)} for axis length {length}")
    source_span = end - start
    target_span = min(
        length,
        max(source_span, round(source_span * focus_scale)),
        max(source_span, round(length * max_focus_fraction)),
    )
    # A selected interval near (but not touching) an image edge can otherwise
    # expand all the way to that edge and map a non-empty context strip to zero
    # output pixels. Reserve one output pixel for every non-empty outside
    # segment. This is the minimal discrete analogue of a strictly monotonic
    # continuous mapping and does not introduce a tunable visual parameter.
    minimum_target_start = 1 if start > 0 else 0
    maximum_target_end = length - 1 if end < length else length
    available_span = maximum_target_end - minimum_target_start
    if available_span < source_span:
        raise ValueError("unable to preserve non-empty context segments on the target axis")
    target_span = min(target_span, available_span)
    center = (start + end) / 2.0
    target_start = round(center - target_span / 2.0)
    target_start = max(
        minimum_target_start,
        min(maximum_target_end - target_span, target_start),
    )
    return target_start, target_start + target_span


def _resample_source_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
) -> Image.Image:
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"invalid source-resampled piece size={size}")
    # Pillow can resample directly from a source-space box. This avoids
    # materialising a potentially very large UHR crop before resizing it.
    return image.resize(size, Image.Resampling.BILINEAR, box=box)


def _overview_size(image: Image.Image, target: int) -> tuple[int, int]:
    """Return exactly the size produced by ``resize_long_side``."""

    if target <= 0 or max(image.size) <= target:
        return image.size
    scale = target / max(image.size)
    return max(1, round(image.width * scale)), max(1, round(image.height * scale))


def _source_resolved_horizontal(
    image: Image.Image,
    source_interval: tuple[int, int],
    target_interval: tuple[int, int],
    *,
    output_width: int,
) -> Image.Image:
    """Map source x segments directly to their final canvas widths.

    The intermediate retains the original source height, so no vertical detail
    is discarded before the vertical mapping is applied.
    """

    width, height = image.size
    sx1, sx2 = source_interval
    tx1, tx2 = target_interval
    out = Image.new("RGB", (output_width, height))
    pieces = (
        ((0, 0, sx1, height), (0, tx1)),
        ((sx1, 0, sx2, height), (tx1, tx2)),
        ((sx2, 0, width, height), (tx2, output_width)),
    )
    for source_box, (target_start, target_end) in pieces:
        source_width = source_box[2] - source_box[0]
        target_width = target_end - target_start
        if source_width == 0 and target_width == 0:
            continue
        if source_width <= 0 or target_width <= 0:
            raise ValueError("source-resolved focus produced a collapsed horizontal segment")
        out.paste(
            _resample_source_box(image, source_box, (target_width, height)),
            (target_start, 0),
        )
    return out


def _source_resolved_vertical(
    image: Image.Image,
    source_interval: tuple[int, int],
    target_interval: tuple[int, int],
    *,
    output_height: int,
) -> Image.Image:
    """Map source y segments directly to their final canvas heights."""

    width, height = image.size
    sy1, sy2 = source_interval
    ty1, ty2 = target_interval
    out = Image.new("RGB", (width, output_height))
    pieces = (
        ((0, 0, width, sy1), (0, ty1)),
        ((0, sy1, width, sy2), (ty1, ty2)),
        ((0, sy2, width, height), (ty2, output_height)),
    )
    for source_box, (target_start, target_end) in pieces:
        source_height = source_box[3] - source_box[1]
        target_height = target_end - target_start
        if source_height == 0 and target_height == 0:
            continue
        if source_height <= 0 or target_height <= 0:
            raise ValueError("source-resolved focus produced a collapsed vertical segment")
        out.paste(
            _resample_source_box(image, source_box, (width, target_height)),
            (0, target_start),
        )
    return out


def _same_half(before: float, after: float, length: int) -> bool:
    midpoint = length / 2.0
    if before == midpoint:
        return abs(after - midpoint) <= 0.5
    return (before < midpoint) == (after < midpoint)


def _focus_one(
    image: Image.Image,
    boxes: Sequence[BBox],
    *,
    overview_long_side: int,
    focus_scale: float,
    max_focus_fraction: float,
) -> tuple[Image.Image, dict]:
    overview_size = _overview_size(image, overview_long_side)
    if not boxes:
        overview = resize_long_side(image, overview_long_side)
        return overview, {
            "focused": False,
            "source_size": list(image.size),
            "overview_size": list(overview.size),
            "source_resolved_sampling": False,
            "pre_downsample_before_focus": False,
        }

    source_union = _union(boxes)
    output_width, output_height = overview_size
    scale_x = output_width / image.width
    scale_y = output_height / image.height
    x1, y1, x2, y2 = source_union
    rendered = (
        max(0, min(output_width - 1, round(x1 * scale_x))),
        max(0, min(output_height - 1, round(y1 * scale_y))),
        max(1, min(output_width, round(x2 * scale_x))),
        max(1, min(output_height, round(y2 * scale_y))),
    )
    rx1, ry1, rx2, ry2 = rendered
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError(f"focus region collapsed after Overview resizing: {rendered}")

    tx1, tx2 = _expanded_interval(
        rx1,
        rx2,
        output_width,
        focus_scale=focus_scale,
        max_focus_fraction=max_focus_fraction,
    )
    ty1, ty2 = _expanded_interval(
        ry1,
        ry2,
        output_height,
        focus_scale=focus_scale,
        max_focus_fraction=max_focus_fraction,
    )
    # Snap a source boundary to the source endpoint exactly when ordinary
    # Overview rounding has already snapped its rendered counterpart there.
    # This keeps source and target segment emptiness paired while retaining the
    # complete source extent in the single output coordinate frame.
    source_x_interval = (0 if rx1 == 0 else x1, image.width if rx2 == output_width else x2)
    source_y_interval = (0 if ry1 == 0 else y1, image.height if ry2 == output_height else y2)

    # Crucially, resampling starts from the original source, not from a 2048
    # Overview. Horizontal and vertical mappings are separable; the temporary
    # image retains the untouched source height until the second pass.
    warped = _source_resolved_horizontal(
        image,
        source_x_interval,
        (tx1, tx2),
        output_width=output_width,
    )
    warped = _source_resolved_vertical(
        warped,
        source_y_interval,
        (ty1, ty2),
        output_height=output_height,
    )

    source_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
    target_center = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    rendered_x_span = rx2 - rx1
    rendered_y_span = ry2 - ry1
    target_x_span = tx2 - tx1
    target_y_span = ty2 - ty1
    x_scale = target_x_span / rendered_x_span
    y_scale = target_y_span / rendered_y_span
    return warped, {
        "focused": x_scale > 1.0 or y_scale > 1.0,
        "source_size": list(image.size),
        "overview_size": list(overview_size),
        "source_union_bbox": list(source_union),
        "overview_focus_bbox_before": list(rendered),
        "overview_focus_bbox_after": [tx1, ty1, tx2, ty2],
        "focus_center_before": list(source_center),
        "focus_center_after": list(target_center),
        "horizontal_scale": x_scale,
        "vertical_scale": y_scale,
        "area_scale": x_scale * y_scale,
        "horizontal_half_preserved": _same_half(source_center[0], target_center[0], output_width),
        "vertical_half_preserved": _same_half(source_center[1], target_center[1], output_height),
        "source_resolved_sampling": True,
        "pre_downsample_before_focus": False,
        "effective_source_focus_bbox": [
            source_x_interval[0],
            source_y_interval[0],
            source_x_interval[1],
            source_y_interval[1],
        ],
        "sampling_density_gain": {
            "horizontal": x_scale,
            "vertical": y_scale,
            "area": x_scale * y_scale,
        },
        "axis_mapping": {
            "x_source_breaks": [0, rx1, rx2, output_width],
            "x_target_breaks": [0, tx1, tx2, output_width],
            "y_source_breaks": [0, ry1, ry2, output_height],
            "y_target_breaks": [0, ty1, ty2, output_height],
            "x_original_source_breaks": [0, source_x_interval[0], source_x_interval[1], image.width],
            "y_original_source_breaks": [0, source_y_interval[0], source_y_interval[1], image.height],
        },
    }


def build_topology_preserving_focus_views(
    images: Sequence[Image.Image],
    regions: Sequence[EvidenceRegion],
    *,
    overview_long_side: int,
    focus_scale: float = DEFAULT_FOCUS_SCALE,
    max_focus_fraction: float = DEFAULT_MAX_FOCUS_FRACTION,
) -> tuple[list[Image.Image], dict]:
    """Create one source-resolved, coordinate-continuous view per source.

    Frozen selected regions are sampled directly from the original source into
    a deterministic separable piecewise-linear canvas. Surrounding context is
    compressed, but no spatial region is cropped away, no second local
    coordinate frame is introduced, and no region-specific marker, label, or
    dataset/question-specific rule is used.
    """

    if overview_long_side <= 0:
        raise ValueError("overview_long_side must be positive")
    if not 1.0 < focus_scale <= 3.0:
        raise ValueError("focus_scale must be within (1.0, 3.0]")
    if not 0.25 <= max_focus_fraction <= 0.80:
        raise ValueError("max_focus_fraction must be within [0.25, 0.80]")

    grouped: dict[int, list[BBox]] = {index: [] for index in range(len(images))}
    for region in regions:
        if region.source_index not in grouped:
            raise ValueError(f"region {region.id!r} points outside the source list")
        grouped[region.source_index].append(region.bbox)

    warped_sources: list[Image.Image] = []
    mappings: list[dict] = []
    for source_index, image in enumerate(images):
        warped, mapping = _focus_one(
            image,
            grouped[source_index],
            overview_long_side=overview_long_side,
            focus_scale=focus_scale,
            max_focus_fraction=max_focus_fraction,
        )
        warped_sources.append(warped)
        mappings.append({"source": f"I{source_index + 1}", **mapping})

    # Use the same wrapper as the matched Overview control so headers and
    # model-facing image dimensions remain identical across methods.
    views, base_layout = build_overview_views(
        warped_sources,
        overview_long_side=overview_long_side,
    )
    return views, {
        **base_layout,
        "organization": "single_coordinate_source_resolved_topology_focus",
        "protocol_id": PROTOCOL_ID,
        "focus_scale": focus_scale,
        "max_focus_fraction": max_focus_fraction,
        "region_policy": "union selected regions independently per source",
        "mapping": mappings,
        "separate_detail_views": False,
        "region_markers_or_labels_added": False,
        "neutral_source_header_matches_overview": True,
        "full_source_extent_retained": True,
        "single_coordinate_frame_per_source": True,
        "piecewise_mapping_monotonic": True,
        "source_resolved_sampling": True,
        "pre_downsample_before_focus": False,
    }
