from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(config_path)
    data = _read_yaml(path)

    merged: dict[str, Any] = {}
    for default in data.get("defaults", []) or []:
        default_path = (path.parent / default).resolve()
        merged = _deep_update(merged, load_config(default_path))

    merged = _deep_update(merged, data)
    if overrides:
        merged = _deep_update(merged, overrides)
    return merged


def require_path(cfg: dict[str, Any], dotted_key: str) -> str:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {dotted_key}")
        cur = cur[part]
    if cur in (None, ""):
        raise ValueError(f"Please set config key '{dotted_key}' or pass the matching CLI argument.")
    return str(cur)
