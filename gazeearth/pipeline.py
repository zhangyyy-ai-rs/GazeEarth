from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .anchors import AnchorConfig, resolve_question_anchor
from .image_ops import build_overview_views, build_selector_board, open_rgb, selected_regions
from .json_utils import clean_short_answer
from .schemas import InferenceResult, SelectionResult
from .selector import FixedGridSelector
from .topology_focus import build_topology_preserving_focus_views

# The public full-scene template has its own identity for safe resume checks.
ANSWER_PROTOCOL_ID = "gazeearth-paper-answer"
STAGES = (
    "Question-guided gaze",
    "Topology-preserving foveated mapping",
    "Evidence-grounded answering",
)


@dataclass(frozen=True)
class FocusConfig:
    """One frozen visual interface shared by every model and dataset."""

    expansion_ratio: float = 0.25
    overview_long_side: int = 2048
    answer_max_new_tokens: int = 1024
    focus_scale: float = 1.5
    max_focus_fraction: float = 0.55

    def __post_init__(self) -> None:
        if self.expansion_ratio < 0:
            raise ValueError("expansion_ratio must be non-negative")
        if self.overview_long_side <= 0 or self.answer_max_new_tokens <= 0:
            raise ValueError("visual size and answer token budget must be positive")
        if not 1.0 < self.focus_scale <= 3.0:
            raise ValueError("focus_scale must be within (1.0, 3.0]")
        if not 0.25 <= self.max_focus_fraction <= 0.80:
            raise ValueError("max_focus_fraction must be within [0.25, 0.80]")


class GazeEarthPipeline:
    """GazeEarth: question-guided gaze, foveated mapping, and grounded answering."""

    VALID_METHODS = {"direct", "gazeearth", "overview"}

    def __init__(
        self,
        *,
        vlm,
        selector: FixedGridSelector,
        focus_config: FocusConfig,
        anchor_config: AnchorConfig | None = None,
        method: str = "gazeearth",
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method={method!r}; choose from {sorted(self.VALID_METHODS)}")
        self.vlm = vlm
        self.selector = selector
        self.focus_config = focus_config
        self.anchor_config = anchor_config or AnchorConfig()
        self.method = method

    @staticmethod
    def answer_prompt(question: str) -> str:
        # Shared verbatim by GazeEarth and the matched Overview control.  The
        # wording describes the real model-facing input without claiming that
        # separate crops or detail views exist.
        return (
            "The visual input contains one full-scene view for each remote-sensing "
            "source. Any labels are neutral correspondence cues only.\n"
            "Answer the visual question using only visible evidence. Explain your reasoning before giving the final answer.\n"
            f"Question: {question}\n"
            "Enclose only the final answer in <answer>...</answer> tags."
        )

    def _answer(self, question: str, images) -> tuple[str, str, float]:
        started = time.perf_counter()
        raw = self.vlm.generate(
            self.answer_prompt(question),
            images=images,
            max_new_tokens=self.focus_config.answer_max_new_tokens,
        )
        return clean_short_answer(raw), raw, time.perf_counter() - started

    def infer(
        self,
        image_paths: Sequence[str],
        evidence_question: str,
        answer_question: str | None = None,
    ) -> InferenceResult:
        if not image_paths:
            raise ValueError("At least one image path is required")
        source_images = [open_rgb(path) for path in image_paths]
        answer_question = answer_question or evidence_question
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        if self.method == "direct":
            # The Direct control passes native source images to the backbone's
            # own image processor, without an overview or focus operation.
            answer, raw_answer, timings["answer"] = self._answer(answer_question, source_images)
            timings["total"] = time.perf_counter() - total_start
            return InferenceResult(
                answer=answer,
                raw_answer=raw_answer,
                method=self.method,
                timings=timings,
                extras={
                    "evidence_question": evidence_question,
                    "answer_question": answer_question,
                    "view_layout": {
                        "organization": "native_direct",
                        "source_count": len(source_images),
                        "views": [
                            {"kind": "source", "source": f"I{index + 1}", "size": list(image.size)}
                            for index, image in enumerate(source_images)
                        ],
                    },
                    "visual_input_count": len(source_images),
                    "source_count": len(source_images),
                    "evidence_organization": "native_direct",
                    "answer_protocol": ANSWER_PROTOCOL_ID,
                    "training_free": True,
                },
            )

        if self.method == "overview":
            started = time.perf_counter()
            answer_images, view_layout = build_overview_views(
                source_images,
                overview_long_side=self.focus_config.overview_long_side,
            )
            timings["view_rendering"] = time.perf_counter() - started
            answer, raw_answer, timings["answer"] = self._answer(answer_question, answer_images)
            timings["total"] = time.perf_counter() - total_start
            return InferenceResult(
                answer=answer,
                raw_answer=raw_answer,
                method=self.method,
                timings=timings,
                extras={
                    "evidence_question": evidence_question,
                    "answer_question": answer_question,
                    "view_layout": view_layout,
                    "visual_input_count": len(answer_images),
                    "source_count": len(source_images),
                    "overview_long_side": self.focus_config.overview_long_side,
                    "evidence_organization": "overview_only",
                    "answer_protocol": ANSWER_PROTOCOL_ID,
                    "training_free": True,
                },
            )

        # Stage 1: Question-guided gaze.
        started = time.perf_counter()
        anchor = resolve_question_anchor(evidence_question, source_images, self.anchor_config)
        timings["anchor_resolution"] = time.perf_counter() - started
        selector_layout = None
        if anchor.resolved:
            selection = SelectionResult("selected", tuple(region.id for region in anchor.regions))
            regions = list(anchor.regions)
            selection_source = anchor.source
            timings["index_rendering"] = 0.0
            timings["selection"] = 0.0
        else:
            started = time.perf_counter()
            selector_board, selector_layout = build_selector_board(
                source_images,
                rows=self.selector.config.grid_rows,
                cols=self.selector.config.grid_cols,
                overview_long_side=self.selector.config.overview_long_side,
            )
            timings["index_rendering"] = time.perf_counter() - started
            started = time.perf_counter()
            selection = self.selector.select(evidence_question, selector_board, len(source_images))
            timings["selection"] = time.perf_counter() - started
            regions = selected_regions(
                source_images,
                selection.region_ids,
                rows=self.selector.config.grid_rows,
                cols=self.selector.config.grid_cols,
                expansion_ratio=self.focus_config.expansion_ratio,
            )
            selection_source = "visual_grid_selector"

        started = time.perf_counter()
        # Stage 2: Topology-preserving foveated mapping from ORIGINAL pixels.
        answer_images, view_layout = build_topology_preserving_focus_views(
            source_images,
            regions,
            overview_long_side=self.focus_config.overview_long_side,
            focus_scale=self.focus_config.focus_scale,
            max_focus_fraction=self.focus_config.max_focus_fraction,
        )
        timings["view_rendering"] = time.perf_counter() - started
        if len(answer_images) != len(source_images):
            raise AssertionError("topology focus must emit exactly one view per source image")

        # Stage 3: Evidence-grounded answering, without selector dialogue history.
        answer, raw_answer, timings["answer"] = self._answer(answer_question, answer_images)
        timings["total"] = time.perf_counter() - total_start
        return InferenceResult(
            answer=answer,
            raw_answer=raw_answer,
            method=self.method,
            timings=timings,
            selection=selection,
            regions=regions,
            extras={
                "evidence_question": evidence_question,
                "answer_question": answer_question,
                "selector_layout": selector_layout,
                "stages": list(STAGES),
                "anchor_resolution": anchor.to_dict(),
                "selection_source": selection_source,
                "view_layout": view_layout,
                "visual_input_count": len(answer_images),
                "source_count": len(source_images),
                "overview_long_side": self.focus_config.overview_long_side,
                "focus_scale": self.focus_config.focus_scale,
                "max_focus_fraction": self.focus_config.max_focus_fraction,
                "evidence_organization": "single_coordinate_source_resolved_topology_focus",
                "answer_protocol": ANSWER_PROTOCOL_ID,
                "model_interface": "FrozenVLM.generate(prompt, images)",
                "training_free": True,
                "dataset_or_question_type_branch": False,
            },
        )
