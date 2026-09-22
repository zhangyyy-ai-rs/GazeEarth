from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class DatasetSample:
    sample_id: str
    image_paths: list[str]
    question: str
    answer: str = ""
    options: list[str] = field(default_factory=list)
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_question(self) -> str:
        if not self.options:
            return self.question
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = [self.question, "Options:"]
        for idx, option in enumerate(self.options):
            prefix = letters[idx] if idx < len(letters) else str(idx + 1)
            lines.append(f"{prefix}. {option}")
        return "\n".join(lines)


class BaseDatasetAdapter:
    def __iter__(self) -> Iterator[DatasetSample]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def resolve_path(root: str | Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)
