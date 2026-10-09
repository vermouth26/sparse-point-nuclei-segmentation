from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .common import REPO_ROOT, make_serializable


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("PyYAML is required to read .yaml config files.") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level config must be a mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def resolve_path(value: Optional[str], base: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def resolve_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(config)
    base = REPO_ROOT

    paths = resolved.setdefault("paths", {})
    for key in list(paths.keys()):
        resolved_path = resolve_path(paths[key], base)
        paths[key] = str(resolved_path) if resolved_path is not None else None

    return resolved


def config_for_json(config: Dict[str, Any]) -> Dict[str, Any]:
    return make_serializable(config)
