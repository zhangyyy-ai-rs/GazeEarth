from __future__ import annotations

import json
import re
from typing import Any


def _scan_object_end(text: str, start: int) -> int | None:
    """Return the matching end index for the object starting at `start`."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first complete top-level JSON object found in model output.

    If the earliest JSON-looking object is opened but never closed, return None
    instead of incorrectly accepting a nested child object from the truncated
    output.  This behavior is important for reliable fallback/truncation audits.
    """
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\{", text)]
    cursor = 0
    for start in starts:
        if start < cursor:
            continue
        end = _scan_object_end(text, start)
        if end is None:
            # Do not reinterpret nested objects inside an unclosed outer object.
            return None
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            cursor = end + 1
            continue
        if isinstance(obj, dict):
            return obj
        cursor = end + 1
    return None


def looks_like_truncated_json(text: str) -> bool:
    """Heuristically flag model outputs that begin JSON but end before closure.

    This is diagnostic only. It never repairs or accepts invalid JSON, so it cannot
    silently change the inference result.  The trace uses it to distinguish a model
    decision fallback from a likely generation-budget truncation.
    """
    if not text or extract_json_object(text) is not None:
        return False
    start = text.find("{")
    if start < 0:
        return False

    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False
    for ch in text[start:]:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1

    return in_string or brace_depth > 0 or bracket_depth > 0


def clean_short_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    obj = extract_json_object(text)
    if obj:
        for key in ("answer", "response", "prediction", "final_answer"):
            if key in obj:
                return str(obj[key]).strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[0] if lines else ""
