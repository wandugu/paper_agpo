"""Helpers for environment-backed paths in local configuration files."""

from __future__ import annotations

import os
import re
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` without changing other text."""
    expanded = value
    for _ in range(10):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            if name in os.environ and (default is None or os.environ[name]):
                return os.environ[name]
            return default if default is not None else match.group(0)

        updated = _ENV_PATTERN.sub(replace, expanded)
        if updated == expanded:
            return updated
        expanded = updated
    return expanded


def expand_env_tree(value: Any) -> Any:
    """Recursively expand environment expressions in dictionaries and lists."""
    if isinstance(value, str):
        return expand_env_vars(value)
    if isinstance(value, list):
        return [expand_env_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_tree(item) for key, item in value.items()}
    return value
