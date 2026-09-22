from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from .metrics import multiple_choice_match, open_ended_match

OUTPUT_FILES = ("predictions.jsonl", "traces.jsonl", "results.json", "run_manifest.json")


def preflight_output_dir(output_dir: str | Path, *, resume: bool, overwrite: bool) -> Path:
    """Reject accidental output reuse before the 8B model is loaded."""

    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output = Path(output_dir)
    existing = [output / name for name in OUTPUT_FILES if (output / name).exists()]
    if existing and not resume and not overwrite:
        raise FileExistsError(
            "output already contains a run; use --resume, --overwrite, or a new --output: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite and existing:
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        for path in existing:
            shutil.copy2(path, path.with_name(f"{path.name}.backup-{stamp}"))
    if resume and not (output / "predictions.jsonl").exists() and not (output / "traces.jsonl").exists():
        raise FileNotFoundError("--resume found neither predictions.jsonl nor traces.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_valid_prefix(path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    truncated_tail = False
    raw_lines = path.read_bytes().splitlines()
    for index, raw in enumerate(raw_lines):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(raw_lines) - 1:
                raise ValueError(f"malformed JSONL before final row in {path}: line {index + 1}") from exc
            truncated_tail = True
    return rows, truncated_tail


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _backup(path: Path, stamp: str) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.backup-{stamp}"))


def _prepare_output_pair(output: Path, *, resume: bool, overwrite: bool) -> list[dict[str, Any]]:
    pred_path = output / "predictions.jsonl"
    trace_path = output / "traces.jsonl"
    if overwrite:
        pred_path.write_text("", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return []
    if not resume:
        pred_path.write_text("", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return []

    if not pred_path.exists() or not trace_path.exists():
        raise ValueError("--resume requires both predictions.jsonl and traces.jsonl")
    predictions, pred_truncated = _read_valid_prefix(pred_path)
    traces, trace_truncated = _read_valid_prefix(trace_path)
    if abs(len(predictions) - len(traces)) > 1:
        raise ValueError("resume audit failed: prediction/trace counts differ by more than one")
    completed = min(len(predictions), len(traces))
    for index in range(completed):
        if predictions[index].get("example_uid") != traces[index].get("example_uid"):
            raise ValueError(f"resume audit failed at row {index}")
    unterminated = any(p.stat().st_size and not p.read_bytes().endswith(b"\n")
                       for p in (pred_path, trace_path))
    if pred_truncated or trace_truncated or len(predictions) != len(traces) or unterminated:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _backup(pred_path, stamp)
        _backup(trace_path, stamp)
        predictions = predictions[:completed]
        traces = traces[:completed]
        _rewrite_jsonl(pred_path, predictions)
        _rewrite_jsonl(trace_path, traces)
    return predictions[:completed]


class EvaluationRunner:
    def __init__(
        self,
        pipeline,
        dataset,
        output_dir: str,
        *,
        limit: int = -1,
        resume: bool = False,
        overwrite: bool = False,
        scorer=None,
        scoring_protocol: str = "normalized-exact-unit-test-v1",
    ) -> None:
        self.pipeline = pipeline
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.resume = resume
        self.overwrite = overwrite
        self.scorer = scorer
        self.scoring_protocol = scoring_protocol

    def run(self) -> dict:
        pred_path = self.output_dir / "predictions.jsonl"
        trace_path = self.output_dir / "traces.jsonl"
        records = _prepare_output_pair(
            self.output_dir,
            resume=self.resume,
            overwrite=self.overwrite,
        )
        dataset_rows = list(self.dataset)
        target_count = len(dataset_rows) if self.limit < 0 else min(self.limit, len(dataset_rows))
        if len(records) > target_count:
            raise ValueError("resume output is longer than the requested dataset/limit")
        for index, record in enumerate(records):
            sample = dataset_rows[index]
            expected_uid = f"{index}:{sample.sample_id}"
            if record.get("example_uid") != expected_uid or int(record.get("dataset_position", -1)) != index:
                raise ValueError(f"resume output is not a prefix of the current dataset at row {index}")
            if record.get("prompt_question") not in (None, sample.prompt_question):
                raise ValueError(f"resume question mismatch at row {index}")

        started = time.time()
        preexisting = len(records)
        iterator = enumerate(dataset_rows[preexisting:target_count], start=preexisting)
        for idx, sample in tqdm(iterator, total=target_count, initial=preexisting, desc="GazeEarth"):
            example_uid = f"{idx}:{sample.sample_id}"
            missing = [path for path in sample.image_paths if not os.path.exists(path)]
            if not sample.image_paths or missing:
                raise FileNotFoundError(f"Missing input image for {example_uid}: {missing}")
            # Ground truth is used only after generation, never by the pipeline.
            result = self.pipeline.infer(
                sample.image_paths,
                evidence_question=sample.question,
                answer_question=sample.prompt_question,
            )
            correct, match_method = self.scorer(result.answer, sample) if self.scorer else (
                multiple_choice_match(result.answer, sample.answer, sample.options)
                if sample.options
                else open_ended_match(result.answer, sample.answer)
            )
            record = {
                "example_uid": example_uid,
                "dataset_position": idx,
                "sample_id": sample.sample_id,
                "question": sample.question,
                "prompt_question": sample.prompt_question,
                "category": sample.category,
                "image_paths": sample.image_paths,
                "answer": result.answer,
                "gt": sample.answer,
                "correct": bool(correct),
                "options": sample.options,
                "match_method": match_method,
                "method": result.method,
                "timings": result.timings,
                "selection_status": result.selection.status if result.selection else None,
                "selection_source": result.extras.get("selection_source"),
                "selected_region_count": len(result.selection.region_ids) if result.selection else 0,
                "final_region_count": len(result.regions),
                "visual_input_count": result.extras.get("visual_input_count", len(sample.image_paths)),
                "evidence_organization": result.extras.get("evidence_organization", ""),
            }
            trace_payload = {
                "example_uid": example_uid,
                "dataset_position": idx,
                "sample_id": sample.sample_id,
                **result.trace_dict(),
            }
            records.append(record)
            record["metadata"] = sample.metadata
            with pred_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")

        valid = records
        correct = sum(int(record["correct"]) for record in valid)
        by_category = defaultdict(lambda: [0, 0])
        by_level = defaultdict(lambda: [0, 0])
        for record in valid:
            key = record.get("category", "")
            by_category[key][1] += 1
            by_category[key][0] += int(record["correct"])
            level = record.get("metadata", {}).get("spatial_level")
            if level:
                by_level[level][0] += int(record["correct"])
                by_level[level][1] += 1
        invocation_wall_time = time.time() - started
        summary = {
            "scoring_protocol": self.scoring_protocol,
            "method": self.pipeline.method,
            "n_total": len(records),
            "n_evaluated": len(valid),
            "n_correct": correct,
            "accuracy": correct / len(valid) if valid else 0.0,
            "accuracy_percent": 100 * correct / len(valid) if valid else 0.0,
            "wall_time_sec": invocation_wall_time,
            "wall_time_sec_this_invocation": invocation_wall_time,
            "resumed": self.resume,
            "n_preexisting": preexisting,
            "mean_time_sec": (
                sum(record.get("timings", {}).get("total", 0.0) for record in valid) / len(valid)
                if valid else 0.0
            ),
            "mean_selection_time_sec": (
                sum(record.get("timings", {}).get("selection", 0.0) for record in valid) / len(valid)
                if valid else 0.0
            ),
            "mean_answer_time_sec": (
                sum(record.get("timings", {}).get("answer", 0.0) for record in valid) / len(valid)
                if valid else 0.0
            ),
            "selection_status_counts": dict(
                Counter(record.get("selection_status") for record in valid if record.get("selection_status"))
            ),
            "selection_source_counts": dict(
                Counter(record.get("selection_source") for record in valid if record.get("selection_source"))
            ),
            "acc_by_category": {
                key: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
                for key, values in sorted(by_category.items())
                if values[1]
            },
            "acc_by_spatial_level": {
                key: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
                for key, values in sorted(by_level.items()) if values[1]
            },
        }
        (self.output_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
