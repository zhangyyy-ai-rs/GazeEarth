from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .json_utils import extract_json_object
from .schemas import SelectionResult


@dataclass(frozen=True)
class SelectorConfig:
    grid_rows: int = 4
    grid_cols: int = 4
    overview_long_side: int = 1536
    max_regions: int = 2
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        if self.max_regions not in {1, 2}:
            raise ValueError("GazeEarth deliberately supports only one or two selected regions")


class FixedGridSelector:
    """A frozen one-call selector with a fixed, dataset-independent action space."""

    VALID_STATUS = {"global_only", "selected", "brief_incomplete"}

    def __init__(self, vlm, config: SelectorConfig) -> None:
        self.vlm = vlm
        self.config = config

    def prompt(self, question: str, source_count: int) -> str:
        sources = ", ".join(f"I{i + 1}" for i in range(source_count))
        return (
            "You are a visual relevance selector for a remote-sensing question, not an answerer.\n\n"
            f"Question: {question}\n"
            f"Indexed source overview(s): {sources}. Each source uses the same "
            f"{self.config.grid_rows}x{self.config.grid_cols} neutral grid.\n\n"
            "Choose the smallest visual brief that would let another expert answer:\n"
            "- global_only: the overview already contains sufficient visual evidence;\n"
            "- selected: magnify one or two cells needed for a local detail, local relation, or two-region "
            "comparison;\n"
            "- brief_incomplete: the question requires evidence scattered over more than two regions, exhaustive "
            "counting/distribution, or cannot be represented by this brief.\n\n"
            "Return JSON only. Never answer the question, describe objects, or provide reasoning.\n"
            "Valid forms:\n"
            '{"status":"global_only","region_ids":[]}\n'
            '{"status":"selected","region_ids":["I1:C07"]}\n'
            '{"status":"selected","region_ids":["I1:C04","I1:C12"]}\n'
            '{"status":"brief_incomplete","region_ids":[]}'
        )

    def _allowed(self, source_count: int) -> set[str]:
        return {
            f"I{source + 1}:C{cell + 1:02d}"
            for source in range(source_count)
            for cell in range(self.config.grid_rows * self.config.grid_cols)
        }

    def parse(self, raw: str, source_count: int) -> SelectionResult:
        obj = extract_json_object(raw)
        if obj is None:
            return SelectionResult(
                "brief_incomplete", raw_output=raw, parse_success=False, failure_reason="invalid_json"
            )
        status = str(obj.get("status", "")).strip().lower()
        values = obj.get("region_ids", obj.get("cell_ids", []))
        if not isinstance(values, list):
            return SelectionResult(
                "brief_incomplete",
                raw_output=raw,
                parse_success=False,
                failure_reason="region_ids_not_list",
            )
        normalized = []
        for value in values:
            rid = str(value).strip().upper()
            if ":" not in rid and source_count == 1:
                rid = f"I1:{rid}"
            if rid not in normalized:
                normalized.append(rid)
        allowed = self._allowed(source_count)
        if status not in self.VALID_STATUS:
            return SelectionResult(
                "brief_incomplete", raw_output=raw, parse_success=False, failure_reason="invalid_status"
            )
        if status == "selected":
            if not 1 <= len(normalized) <= self.config.max_regions or any(rid not in allowed for rid in normalized):
                return SelectionResult(
                    "brief_incomplete",
                    raw_output=raw,
                    parse_success=False,
                    failure_reason="invalid_selection",
                )
            return SelectionResult(status, tuple(normalized), raw_output=raw)
        if normalized:
            return SelectionResult(
                "brief_incomplete",
                raw_output=raw,
                parse_success=False,
                failure_reason="regions_for_non_selected",
            )
        return SelectionResult(status, (), raw_output=raw)

    def select(self, question: str, selector_board: Image.Image, source_count: int) -> SelectionResult:
        raw = self.vlm.generate(
            self.prompt(question, source_count),
            images=[selector_board],
            max_new_tokens=self.config.max_new_tokens,
        )
        return self.parse(raw, source_count)
