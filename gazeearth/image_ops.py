from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .schemas import BBox, EvidenceRegion

Image.MAX_IMAGE_PIXELS = None


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_long_side(image: Image.Image, target: int) -> Image.Image:
    if target <= 0 or max(image.size) <= target:
        return image.copy()
    scale = target / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=max(10, size))
    except OSError:
        return ImageFont.load_default()


def cell_id(row: int, col: int, cols: int) -> str:
    return f"C{row * cols + col + 1:02d}"


def cell_bbox(width: int, height: int, rows: int, cols: int, cid: str) -> BBox:
    index = int(cid[1:]) - 1
    row, col = divmod(index, cols)
    if index < 0 or row >= rows:
        raise ValueError(f"Cell {cid!r} is outside a {rows}x{cols} grid")
    return (
        round(col * width / cols),
        round(row * height / rows),
        round((col + 1) * width / cols),
        round((row + 1) * height / rows),
    )


def expand_bbox(bbox: BBox, width: int, height: int, ratio: float) -> BBox:
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * max(0.0, ratio)
    dy = (y2 - y1) * max(0.0, ratio)
    return (
        max(0, round(x1 - dx)),
        max(0, round(y1 - dy)),
        min(width, round(x2 + dx)),
        min(height, round(y2 + dy)),
    )


def _intersects(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _union(boxes: Sequence[BBox]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def selected_regions(
    images: Sequence[Image.Image],
    region_ids: Sequence[str],
    *,
    rows: int,
    cols: int,
    expansion_ratio: float,
    merge_overlaps: bool = True,
) -> list[EvidenceRegion]:
    """Map normalized grid IDs to source-space focus boxes."""
    pending: list[tuple[int, str, BBox]] = []
    for rid in region_ids:
        source_text, cid = rid.split(":", 1)
        source_index = int(source_text[1:]) - 1
        image = images[source_index]
        base = cell_bbox(image.width, image.height, rows, cols, cid)
        pending.append((source_index, cid, expand_bbox(base, image.width, image.height, expansion_ratio)))

    if not merge_overlaps:
        groups = [[item] for item in pending]
    else:
        groups: list[list[tuple[int, str, BBox]]] = []
        for item in pending:
            matches = [
                i
                for i, group in enumerate(groups)
                if group[0][0] == item[0] and any(_intersects(x[2], item[2]) for x in group)
            ]
            if not matches:
                groups.append([item])
                continue
            first = matches[0]
            groups[first].append(item)
            for index in reversed(matches[1:]):
                groups[first].extend(groups.pop(index))

    out: list[EvidenceRegion] = []
    for group in groups:
        source_index = group[0][0]
        cids = tuple(x[1] for x in group)
        rid = f"I{source_index + 1}:" + "+".join(cids)
        out.append(
            EvidenceRegion(
                id=rid,
                source_index=source_index,
                source_label=f"I{source_index + 1}",
                cell_ids=cids,
                bbox=_union([x[2] for x in group]),
            )
        )
    return out


def draw_indexed_overview(image: Image.Image, source_label: str, rows: int, cols: int, long_side: int) -> Image.Image:
    overview = resize_long_side(image, long_side)
    draw = ImageDraw.Draw(overview)
    line_width = max(2, round(min(overview.size) / 450))
    font = _font(round(min(overview.size) / 34))
    for row in range(rows):
        for col in range(cols):
            x1, y1, x2, y2 = cell_bbox(overview.width, overview.height, rows, cols, cell_id(row, col, cols))
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 215, 0), width=line_width)
            label = f"{source_label}:{cell_id(row, col, cols)}"
            box = draw.textbbox((x1 + 4, y1 + 3), label, font=font)
            draw.rectangle(box, fill=(15, 15, 15))
            draw.text((x1 + 4, y1 + 3), label, fill=(255, 235, 80), font=font)
    return overview


def _fit_panel(image: Image.Image, width: int, height: int, background=(245, 245, 245)) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    panel = Image.new("RGB", (width, height), background)
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return panel


def build_selector_board(
    images: Sequence[Image.Image], *, rows: int, cols: int, overview_long_side: int
) -> tuple[Image.Image, dict]:
    indexed = [
        draw_indexed_overview(image, f"I{i + 1}", rows, cols, overview_long_side)
        for i, image in enumerate(images)
    ]
    pad, header = 12, 34
    panel_width = max(image.width for image in indexed)
    panel_height = max(image.height for image in indexed)
    ncols = min(2, len(indexed))
    nrows = math.ceil(len(indexed) / ncols)
    board = Image.new(
        "RGB",
        (ncols * panel_width + (ncols + 1) * pad, nrows * (panel_height + header) + (nrows + 1) * pad),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    font = _font(20)
    placements = []
    for index, image in enumerate(indexed):
        row, col = divmod(index, ncols)
        x = pad + col * panel_width
        y = pad + row * (panel_height + header)
        draw.text((x + 4, y + 5), f"SOURCE I{index + 1}", fill=(25, 25, 25), font=font)
        board.paste(_fit_panel(image, panel_width, panel_height), (x, y + header))
        placements.append({"source": f"I{index + 1}", "row": row, "col": col})
    return board, {"rows": nrows, "cols": ncols, "placements": placements}


def build_overview_views(
    images: Sequence[Image.Image], *, overview_long_side: int
) -> tuple[list[Image.Image], dict]:
    """Wrap each resized source in the matched Overview interface."""

    if overview_long_side <= 0:
        raise ValueError("overview_long_side must be positive")
    views: list[Image.Image] = []
    layout: dict = {
        "organization": "separate_overviews",
        "overview_count": len(images),
        "detail_count": 0,
        "coordinate_synchronized": False,
        "views": [],
    }
    for source_index, image in enumerate(images):
        resized = resize_long_side(image, overview_long_side)
        header = 36
        overview = Image.new("RGB", (resized.width, resized.height + header), (245, 245, 245))
        draw = ImageDraw.Draw(overview)
        draw.text((8, 7), f"OVERVIEW I{source_index + 1}", fill=(25, 25, 25), font=_font(20))
        overview.paste(resized, (0, header))
        views.append(overview)
        layout["views"].append(
            {
                "kind": "overview",
                "source": f"I{source_index + 1}",
                "size": list(overview.size),
                "marked": False,
            }
        )
    return views, layout
