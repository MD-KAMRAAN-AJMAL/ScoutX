"""Keep sensitive text and local locations out of saved evaluation reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(?:token|password|secret|credential|auth|api.?key|path|directory|file.?content|query.?text|dsn|url)",
    re.IGNORECASE,
)
_LOCAL_OR_REMOTE_LOCATION = re.compile(
    r"(?:^/|^[A-Za-z]:[\\/]|^(?:https?|postgres(?:ql)?|redis)://)", re.IGNORECASE
)
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def safe_configuration(values: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in values.items():
        if not isinstance(key, str) or _SENSITIVE_KEY.search(key):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned = _safe_value(value)
            if cleaned is not None:
                output[key] = cleaned
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            cleaned = [_safe_value(item) for item in value]
            if all(item is not None for item in cleaned):
                output[key] = cleaned
    return output


def safe_label(value: Any, fallback: str = "unspecified") -> str:
    text = str(value)
    return text if _SAFE_LABEL.fullmatch(text) else fallback


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if _LOCAL_OR_REMOTE_LOCATION.search(text):
            return None
        return text[:200]
    return value
