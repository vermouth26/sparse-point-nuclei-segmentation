from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "CPM" / "official_28_4_32" / "split.json"
DEFAULT_TRAIN_IMAGES = REPO_ROOT / "data" / "raw" / "CPM17_official_28_4_32" / "converted" / "official_train" / "images"
DEFAULT_TRAIN_LABELS = REPO_ROOT / "data" / "raw" / "CPM17_official_28_4_32" / "converted" / "official_train" / "labels_instance"
DEFAULT_TEST_IMAGES = REPO_ROOT / "data" / "raw" / "CPM17_official_28_4_32" / "converted" / "official_test" / "images"
DEFAULT_TEST_LABELS = (
    REPO_ROOT
    / "data"
    / "raw"
    / "CPM17_official_28_4_32"
    / "converted"
    / "official_test"
    / "labels_instance_eval_only"
)


def add_hovernet_root(hovernet_root: Path) -> None:
    hovernet_root = hovernet_root.resolve()
    if not hovernet_root.exists():
        raise SystemExit(f"HoVerNet root does not exist: {hovernet_root}")
    if str(hovernet_root) not in sys.path:
        sys.path.insert(0, str(hovernet_root))
    # The upstream imgaug version still expects this NumPy alias.
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_split(path: Path = DEFAULT_SPLIT) -> dict[str, list[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    required = {"train", "val", "test"}
    if not required.issubset(split):
        raise SystemExit(f"Split file missing required keys {required}: {path}")
    return split


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_instance(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing instance label: {path}")
    inst = np.load(path, allow_pickle=False)
    return relabel_contiguous(inst)


def relabel_contiguous(inst: np.ndarray) -> np.ndarray:
    out = np.zeros(inst.shape[:2], dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        out[inst == inst_id] = next_id
        next_id += 1
    return out


def mode_shapes(mode: str) -> tuple[int, int]:
    if mode == "original":
        return 270, 80
    if mode == "fast":
        return 256, 164
    raise SystemExit(f"Unsupported HoVerNet mode: {mode}")


def center_crop_np(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return arr[top : top + crop_h, left : left + crop_w, ...]


def pad_to_shape(image: np.ndarray, min_h: int, min_w: int, *, is_label: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return image
    pad_spec = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
    if image.ndim == 3:
        pad_spec.append((0, 0))
    mode = "constant" if is_label else "reflect"
    return np.pad(image, pad_spec, mode=mode)


def random_crop_pair(
    image: np.ndarray,
    inst: np.ndarray,
    crop_size: int,
    rng: random.Random,
    *,
    min_fg_ratio: float = 0.0,
    max_tries: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    image = pad_to_shape(image, crop_size, crop_size, is_label=False)
    inst = pad_to_shape(inst, crop_size, crop_size, is_label=True)
    h, w = image.shape[:2]
    best = None
    for _ in range(max_tries):
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        img_crop = image[top : top + crop_size, left : left + crop_size]
        inst_crop = inst[top : top + crop_size, left : left + crop_size]
        fg_ratio = float((inst_crop > 0).mean())
        best = (img_crop, inst_crop)
        if fg_ratio >= min_fg_ratio:
            return img_crop, relabel_contiguous(inst_crop)
    assert best is not None
    return best[0], relabel_contiguous(best[1])


def augment_pair(
    image: np.ndarray,
    inst: np.ndarray,
    rng: random.Random,
    *,
    blur: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        inst = np.flip(inst, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        inst = np.flip(inst, axis=0)
    k = rng.randint(0, 3)
    if k:
        image = np.rot90(image, k, axes=(0, 1))
        inst = np.rot90(inst, k, axes=(0, 1))
    if blur and rng.random() < 0.25:
        pil = Image.fromarray(np.ascontiguousarray(image))
        if rng.random() < 0.5:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
        else:
            pil = pil.filter(ImageFilter.MedianFilter(size=3))
        image = np.asarray(pil, dtype=np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(inst)


def image_label_paths(image_id: str, image_dir: Path, label_dir: Path) -> tuple[Path, Path]:
    return image_dir / f"{image_id}.png", label_dir / f"{image_id}_label.npy"


def make_overlay(image: np.ndarray, inst_map: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(23)
    table = rng.integers(40, 255, size=(max(int(inst_map.max()) + 1, 2), 3), dtype=np.uint8)
    table[0] = 0
    color = table[np.clip(inst_map.astype(np.int32), 0, table.shape[0] - 1)]
    return (0.45 * image.astype(np.float32) + 0.55 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def instance_centers(inst: np.ndarray) -> dict[int, tuple[float, float]]:
    centers = {}
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        ys, xs = np.nonzero(inst == inst_id)
        if len(ys):
            centers[inst_id] = (float(ys.mean()), float(xs.mean()))
    return centers


def count_small_objects(inst: np.ndarray, min_area: int = 10) -> int:
    count = 0
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        if int((inst == inst_id).sum()) < min_area:
            count += 1
    return count


def gt_removed_small_objects(inst: np.ndarray, min_size: int) -> np.ndarray:
    out = np.zeros(inst.shape, dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        mask = inst == inst_id
        if int(mask.sum()) >= min_size:
            out[mask] = next_id
            next_id += 1
    return out


def fill_holes_per_instance(inst: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    out = np.zeros_like(inst, dtype=np.int32)
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        out[ndi.binary_fill_holes(inst == inst_id)] = inst_id
    return relabel_contiguous(out)
