"""Resolve shared model and dataset stores while preserving local fallbacks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in a string."""
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
    """Recursively expand environment expressions in config data."""
    if isinstance(value, str):
        return expand_env_vars(value)
    if isinstance(value, list):
        return [expand_env_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_tree(item) for key, item in value.items()}
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config after expanding shared-store expressions."""
    import yaml

    config = yaml.safe_load(expand_env_vars(path.read_text(encoding="utf-8"))) or {}
    return expand_env_tree(config)


def model_path(name: str, fallback: str) -> str:
    """Return a model path from MODEL_ROOT, or the exact old fallback."""
    root = os.environ.get("MODEL_ROOT")
    return str(Path(root) / name) if root else fallback


def dataset_path(name: str, fallback: str) -> str:
    """Return a dataset path from DATASET_ROOT, or the exact old fallback."""
    root = os.environ.get("DATASET_ROOT")
    return str(Path(root) / name) if root else fallback
