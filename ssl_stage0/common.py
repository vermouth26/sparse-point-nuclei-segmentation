from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = REPO_ROOT / "detection"


def ensure_detection_imports() -> None:
    detection_path = str(DETECTION_DIR)
    if detection_path not in sys.path:
        sys.path.insert(0, detection_path)


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def set_random_seed(seed: int) -> None:
    if seed < 0:
        torch.backends.cudnn.benchmark = True
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "state_dict.", "encoder.", "resnet.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key
