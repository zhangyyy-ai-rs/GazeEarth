from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        for key in ("data", "records", "samples", "annotations"):
            if isinstance(obj, dict) and isinstance(obj.get(key), list):
                return obj[key]
        raise ValueError(f"Unable to locate record list in {path}")
    if suffix in {".tsv", ".csv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, dialect=dialect))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to read parquet annotations") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported annotation format: {path}")


def first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def parse_options(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return parse_options([value[k] for k in sorted(value)])
    if isinstance(value, (list, tuple)):
        return [re.sub(rf"^\s*\(?{chr(65+i)}[.):]\s*", "", str(x)).strip()
                for i, x in enumerate(value)]
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return parse_options(obj)
        if isinstance(obj, dict):
            return parse_options(obj)
    except Exception:
        pass
    # Common "A: foo B: bar" / newline representations.
    parts = re.split(r"(?:^|\s)[A-Z][\.:\)]\s*", text)
    parts = [x.strip(" ;|") for x in parts if x.strip(" ;|")]
    return parts if len(parts) > 1 else [x.strip() for x in text.split("|") if x.strip()]


def parse_images(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        text = value.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        return [text]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]
