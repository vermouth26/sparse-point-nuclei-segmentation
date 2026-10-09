from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_DIR = REPO_ROOT / "detection"
SEGMENTATION_DIR = REPO_ROOT / "segmentation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

import detection.prepare_data as det_prepare_data  # noqa: E402
import detection.test as det_test  # noqa: E402
from detection.model import create_model  # noqa: E402
from detection.my_transforms import get_transforms  # noqa: E402
from ssl_stage0.stage1_options import build_stage1_opt  # noqa: E402
import detection.utils as det_utils  # noqa: E402
import segmentation.metrics as seg_metrics  # noqa: E402

# The legacy module imports `utils` without a package prefix. Bind it to the
# detection utilities before calling its helpers from this package.
det_prepare_data.utils = det_utils
det_test.utils = det_utils
create_cluster_label = det_prepare_data.create_cluster_label
create_voronoi_label = det_prepare_data.create_voronoi_label
get_probmaps = det_test.get_probmaps


DEFAULT_STAGE1_CONFIG = (
    REPO_ROOT
    / "configs"
    / "stage1_ratio010.yaml"
)


def parser_common(
    description: str,
    default_split: str = "train",
    default_max_images: int = 3,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="MO")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--description", default="_alpha050_l3l4_s1_bs64")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1_CONFIG))
    parser.add_argument("--split", default=default_split, help="Comma-separated split names, e.g. train or train,val")
    parser.add_argument("--images", default="", help="Optional comma-separated image stems. Overrides --split selection.")
    parser.add_argument("--max-images", type=int, default=default_max_images)
    parser.add_argument("--out-dir", required=True)
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fail(message: str, code: int = 2) -> None:
    raise SystemExit(message)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def strip_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "state_dict.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def discover_root(kind: str, provided: str, dataset: str) -> Path:
    if provided:
        root = normalize_path(provided)
        if (root / dataset).exists():
            return root
        fail(f"{kind} root does not contain dataset '{dataset}': {root}")
    candidates = []
    if kind == "raw":
        roots = [REPO_ROOT / "data" / "raw"]
    elif kind == "processed":
        roots = [REPO_ROOT / "data" / "processed"]
    else:
        raise ValueError(kind)
    for root in roots:
        if (root / dataset).exists():
            candidates.append(root.resolve())
    if not candidates:
        fail(f"Could not discover {kind} root for dataset '{dataset}'. Please pass --{kind}-root explicitly.")
    if len(candidates) > 1:
        joined = ", ".join(str(v) for v in candidates)
        fail(f"Found multiple candidate {kind} roots for dataset '{dataset}': {joined}. Please pass --{kind}-root explicitly.")
    return candidates[0]


def verify_dataset_layout(raw_root: Path, processed_root: Path, dataset: str, ratio: float, description: str) -> dict[str, Path]:
    dataset_dir = raw_root / dataset
    processed_dataset_dir = processed_root / dataset
    ratio_dir = dataset_dir / f"ratio{ratio}{description}"
    required = {
        "split_file": dataset_dir / "train_val_test.json",
        "image_dir": dataset_dir / "images",
        "gt_instance_dir": dataset_dir / "labels_instance",
        "gt_point_dir": dataset_dir / "labels_point",
        "partial_point_dir": ratio_dir / "labels_point" / "0",
        "processed_image_dir": processed_dataset_dir / "images",
    }
    missing = [f"{key}={value}" for key, value in required.items() if not value.exists()]
    if missing:
        fail("Dataset layout is incomplete:\n" + "\n".join(missing))
    return {
        "raw_root": raw_root,
        "processed_root": processed_root,
        "dataset_dir": dataset_dir,
        "processed_dataset_dir": processed_dataset_dir,
        "ratio_dir": ratio_dir,
        **required,
    }


def resolve_stage1_config(stage1_config: str) -> Path:
    path = normalize_path(stage1_config)
    if not path.exists():
        fail(f"Stage1 config not found: {path}")
    return path


def build_stage1_runtime(stage1_config: Path, raw_root: Path | None = None, processed_root: Path | None = None):
    cfg = load_yaml(stage1_config)
    if raw_root is not None:
        cfg["dataset"]["raw_data_dir"] = str(raw_root)
    if processed_root is not None:
        cfg["dataset"]["processed_data_dir"] = str(processed_root)
    cfg["dataset"]["raw_data_dir"] = str(normalize_path(cfg["dataset"]["raw_data_dir"]))
    cfg["dataset"]["processed_data_dir"] = str(normalize_path(cfg["dataset"]["processed_data_dir"]))
    opt = build_stage1_opt(cfg)
    return cfg, opt


def load_stage1_model_from_checkpoint(checkpoint_path: Path, opt, device: torch.device):
    if not checkpoint_path.exists():
        fail(f"Stage1 checkpoint not found: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = payload.get("state_dict") or payload.get("model") or payload
    normalized = OrderedDict((strip_prefixes(str(k)), v) for k, v in state_dict.items() if torch.is_tensor(v))
    model = create_model(opt.model["name"], opt.model["out_c"], opt.model["pretrained"])
    try:
        model.load_state_dict(normalized, strict=True)
    except Exception as exc:
        fail(f"Failed to load checkpoint into stage1 detector: {exc}")
    model = model.to(device)
    model.eval()
    return model, payload


def load_split(split_file: Path) -> dict[str, list[str]]:
    with split_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_images(
    split_data: dict[str, list[str]],
    split_names: list[str],
    explicit_images: list[str],
    max_images: int,
) -> list[dict[str, str]]:
    if explicit_images:
        owner = {}
        for split_name, stems in split_data.items():
            for stem in stems:
                owner[stem] = split_name
        selected = []
        for stem in explicit_images:
            split_name = owner.get(stem)
            if split_name is None:
                fail(f"Image stem not found in split file: {stem}")
            selected.append({"image_id": stem, "split": split_name})
        return selected[:max_images]

    selected = []
    for split_name in split_names:
        for stem in split_data.get(split_name, []):
            selected.append({"image_id": stem, "split": split_name})
            if len(selected) >= max_images:
                return selected
    return selected


def parse_explicit_images(images_arg: str) -> list[str]:
    return [item.strip() for item in images_arg.split(",") if item.strip()]


def parse_split_names(split_arg: str) -> list[str]:
    split_names = [item.strip() for item in split_arg.split(",") if item.strip()]
    if not split_names:
        fail("No valid split names provided.")
    return split_names


def image_path(dataset_dir: Path, stem: str) -> Path:
    return dataset_dir / "images" / f"{stem}.png"


def gt_instance_path(dataset_dir: Path, stem: str) -> Path:
    return dataset_dir / "labels_instance" / f"{stem}_label.npy"


def gt_point_path(dataset_dir: Path, stem: str) -> Path:
    return dataset_dir / "labels_point" / f"{stem}_label_point.png"


def partial_point_path(ratio_dir: Path, stem: str) -> Path:
    return ratio_dir / "labels_point" / "0" / f"{stem}_label_point.png"


def existing_pred_point_path(ratio_dir: Path, stem: str) -> Path:
    return ratio_dir / "labels_point" / "pred" / f"{stem}_label_point.png"


def imread(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return imageio.imread(path)


def save_png(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    imageio.imwrite(path, array)


def save_instance_png(path: Path, inst_map: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(inst_map.astype(np.uint16)).save(path)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def point_count(point_map: np.ndarray) -> int:
    return int(np.sum(np.asarray(point_map) > 0))


def threshold_to_point_map(prob_map: np.ndarray, threshold: float, min_area: int, max_area: int) -> np.ndarray:
    pred = prob_map > threshold
    pred_labeled, num = measure.label(pred, return_num=True)
    if num > 1:
        import skimage.morphology as ski_morph

        bg_area = ski_morph.remove_small_objects(pred_labeled, max_area) > 0
        large_area = ski_morph.remove_small_objects(pred_labeled, min_area) > 0
        pred = pred * (bg_area == 0) * (large_area > 0)
        pred_labeled = measure.label(pred)
    point_map = np.zeros(prob_map.shape, dtype=np.uint8)
    for region in measure.regionprops(pred_labeled):
        y, x = region.centroid
        yy = int(round(y))
        xx = int(round(x))
        if 0 <= yy < point_map.shape[0] and 0 <= xx < point_map.shape[1]:
            point_map[yy, xx] = 255
    return point_map


def merge_with_partial(partial_points: np.ndarray, pseudo_points: np.ndarray, merge_radius: int) -> np.ndarray:
    import skimage.morphology as ski_morph

    merged = partial_points.copy()
    dilated = ski_morph.dilation(partial_points > 0, footprint=ski_morph.disk(merge_radius))
    pseudo_labeled = measure.label(pseudo_points > 0)
    for region in measure.regionprops(pseudo_labeled):
        y, x = region.centroid
        yy = int(round(y))
        xx = int(round(x))
        if 0 <= yy < merged.shape[0] and 0 <= xx < merged.shape[1] and not dilated[yy, xx]:
            merged[yy, xx] = 255
    return merged.astype(np.uint8)


def point_map_to_coords(point_map: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(np.asarray(point_map) > 0)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def compute_detection_counts(pred_point_map: np.ndarray, gt_point_map: np.ndarray, radius: int) -> dict[str, Any]:
    tp, fp, fn = det_utils.compute_accuracy(pred_point_map > 0, gt_point_map, radius=radius)
    precision, recall, f1 = f1_from_counts(int(tp), int(fp), int(fn))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def build_overlay_points(
    image_rgb: np.ndarray,
    partial_coords: list[tuple[int, int]],
    pseudo_coords: list[tuple[int, int]],
    final_coords: list[tuple[int, int]],
    gt_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)

    def paint(coords: list[tuple[int, int]], color: tuple[int, int, int], radius: int) -> None:
        for y, x in coords:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

    if gt_coords:
        paint(gt_coords, (0, 255, 255), 5)
    paint(partial_coords, (0, 255, 0), 4)
    paint(pseudo_coords, (255, 200, 0), 4)
    paint(final_coords, (255, 0, 0), 4)
    return np.asarray(canvas)


def binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            ys0 = max(0, -dy)
            ys1 = min(mask.shape[0], mask.shape[0] - dy)
            yd0 = max(0, dy)
            yd1 = min(mask.shape[0], mask.shape[0] + dy)
            xs0 = max(0, -dx)
            xs1 = min(mask.shape[1], mask.shape[1] - dx)
            xd0 = max(0, dx)
            xd1 = min(mask.shape[1], mask.shape[1] + dx)
            out[yd0:yd1, xd0:xd1] |= mask[ys0:ys1, xs0:xs1]
    return out


def binary_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            shifted = np.zeros_like(mask, dtype=bool)
            ys0 = max(0, -dy)
            ys1 = min(mask.shape[0], mask.shape[0] - dy)
            yd0 = max(0, dy)
            yd1 = min(mask.shape[0], mask.shape[0] + dy)
            xs0 = max(0, -dx)
            xs1 = min(mask.shape[1], mask.shape[1] - dx)
            xd0 = max(0, dx)
            xd1 = min(mask.shape[1], mask.shape[1] + dx)
            shifted[yd0:yd1, xd0:xd1] = mask[ys0:ys1, xs0:xs1]
            out &= shifted
    return out


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & (~binary_erosion(mask, 1))


def relabel_contiguous(inst_map: np.ndarray) -> np.ndarray:
    out = np.zeros(inst_map.shape, dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        out[inst_map == inst_id] = next_id
        next_id += 1
    return out


def nearest_positive_label(label_map: np.ndarray, y: int, x: int, radius: int = 6) -> int:
    if 0 <= y < label_map.shape[0] and 0 <= x < label_map.shape[1] and label_map[y, x] > 0:
        return int(label_map[y, x])
    y0 = max(0, y - radius)
    y1 = min(label_map.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(label_map.shape[1], x + radius + 1)
    best_label = 0
    best_dist = None
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if label_map[yy, xx] <= 0:
                continue
            d = (yy - y) * (yy - y) + (xx - x) * (xx - x)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_label = int(label_map[yy, xx])
    return best_label


def decode_vor(rgb: np.ndarray) -> dict[str, np.ndarray]:
    edge = rgb[:, :, 0] > 127
    point = rgb[:, :, 1] > 127
    return {
        "edge": edge,
        "point": point,
        "cell_mask": ~edge,
    }


def decode_clu(rgb: np.ndarray) -> dict[str, np.ndarray]:
    bg = rgb[:, :, 0] > 127
    fg = rgb[:, :, 1] > 127
    return {
        "bg": bg,
        "fg": fg,
        "ignore": ~(bg | fg),
    }


def point_seeds(point_map: np.ndarray) -> list[dict[str, Any]]:
    return [{"instance_id": idx, "seed_y": y, "seed_x": x, "y": y, "x": x} for idx, (y, x) in enumerate(point_map_to_coords(point_map), start=1)]


def snap_seeds_to_region(seeds: list[dict[str, Any]], region_mask: np.ndarray, snap_radius: int = 6) -> list[dict[str, Any]]:
    snapped = []
    used = np.zeros(region_mask.shape, dtype=bool)
    for item in seeds:
        y = int(item["seed_y"])
        x = int(item["seed_x"])
        if 0 <= y < region_mask.shape[0] and 0 <= x < region_mask.shape[1] and region_mask[y, x] and not used[y, x]:
            used[y, x] = True
            snapped.append(dict(item))
            continue
        best = None
        best_dist = None
        y0 = max(0, y - snap_radius)
        y1 = min(region_mask.shape[0], y + snap_radius + 1)
        x0 = max(0, x - snap_radius)
        x1 = min(region_mask.shape[1], x + snap_radius + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if not region_mask[yy, xx] or used[yy, xx]:
                    continue
                d = (yy - y) * (yy - y) + (xx - x) * (xx - x)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best = (yy, xx)
        if best is None:
            continue
        yy, xx = best
        used[yy, xx] = True
        new_item = dict(item)
        new_item["seed_y"] = int(yy)
        new_item["seed_x"] = int(xx)
        snapped.append(new_item)
    return snapped


def grow_instances_by_distance(region_mask: np.ndarray, seeds: list[dict[str, Any]]) -> np.ndarray:
    coords = np.argwhere(region_mask)
    inst_map = np.zeros(region_mask.shape, dtype=np.int32)
    if not seeds or coords.size == 0:
        return inst_map
    centers = np.asarray([(int(item["seed_y"]), int(item["seed_x"])) for item in seeds], dtype=np.int32)
    inst_ids = np.asarray([int(item["instance_id"]) for item in seeds], dtype=np.int32)
    for y, x in coords:
        d2 = np.square(centers[:, 0] - y) + np.square(centers[:, 1] - x)
        inst_map[y, x] = int(inst_ids[int(np.argmin(d2))])
    return relabel_contiguous(inst_map)


def vor_label_to_instance_map(label_vor_rgb: np.ndarray, point_map: np.ndarray) -> np.ndarray:
    decoded = decode_vor(label_vor_rgb)
    cell_labels = measure.label(decoded["cell_mask"])
    inst_map = np.zeros(cell_labels.shape, dtype=np.int32)
    per_cell: dict[int, list[tuple[int, int]]] = {}
    for y, x in point_map_to_coords(point_map):
        cell_id = nearest_positive_label(cell_labels, y, x, radius=6)
        if cell_id <= 0:
            continue
        per_cell.setdefault(cell_id, []).append((y, x))
    next_id = 1
    for cell_id, centers in per_cell.items():
        cell_mask = cell_labels == cell_id
        if len(centers) == 1:
            inst_map[cell_mask] = next_id
            next_id += 1
            continue
        yy, xx = np.where(cell_mask)
        center_arr = np.asarray(centers, dtype=np.int32)
        for y, x in zip(yy, xx):
            d2 = np.square(center_arr[:, 0] - y) + np.square(center_arr[:, 1] - x)
            inst_map[y, x] = next_id + int(np.argmin(d2))
        next_id += len(centers)
    return relabel_contiguous(inst_map)


def clu_label_to_instance_map(label_clu_rgb: np.ndarray, point_map: np.ndarray) -> np.ndarray:
    decoded = decode_clu(label_clu_rgb)
    seeds = snap_seeds_to_region(point_seeds(point_map), decoded["fg"], snap_radius=8)
    return grow_instances_by_distance(decoded["fg"], seeds)


def colorize_instances(inst_map: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    table = rng.integers(30, 255, size=(max(int(inst_map.max()) + 1, 2), 3), dtype=np.uint8)
    table[0] = 0
    return table[np.clip(inst_map.astype(np.int32), 0, table.shape[0] - 1)]


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    idx = mask > 0
    out[idx] = (1.0 - alpha) * out[idx] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def tile_images(named_arrays: list[tuple[str, np.ndarray]], out_path: Path, tile_size: tuple[int, int] = (320, 320)) -> None:
    panels = []
    for name, array in named_arrays:
        if array.ndim == 2:
            rgb = np.stack([array] * 3, axis=2).astype(np.uint8)
        else:
            rgb = array.astype(np.uint8)
        img = Image.fromarray(rgb)
        img = ImageOps.contain(img, tile_size)
        canvas = Image.new("RGB", tile_size, color=(20, 20, 20))
        x = (tile_size[0] - img.width) // 2
        y = (tile_size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, tile_size[0], 22), fill=(0, 0, 0))
        draw.text((6, 4), name, fill=(255, 255, 255))
        panels.append(canvas)
    cols = min(3, max(1, len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (tile_size[0] * cols, tile_size[1] * rows), color=(10, 10, 10))
    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        sheet.paste(panel, (col * tile_size[0], row * tile_size[1]))
    ensure_dir(out_path.parent)
    sheet.save(out_path)


def binary_dice(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    pred = pred_inst > 0
    gt = gt_inst > 0
    inter = float(np.logical_and(pred, gt).sum())
    den = float(pred.sum() + gt.sum())
    return 2.0 * inter / den if den > 0 else 1.0


def instance_centroids(inst_map: np.ndarray) -> list[tuple[float, float]]:
    centers = []
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        ys, xs = np.where(inst_map == inst_id)
        if ys.size == 0:
            continue
        centers.append((float(np.mean(ys)), float(np.mean(xs))))
    return centers


def detection_f1_from_instances(inst_map: np.ndarray, gt_point_map: np.ndarray, radius: float = 15.0) -> dict[str, Any]:
    gt_points = np.argwhere(gt_point_map > 0).astype(np.float32)
    pred_points = np.asarray(instance_centroids(inst_map), dtype=np.float32)
    if pred_points.size == 0:
        tp = 0
        fp = 0
        fn = int(gt_points.shape[0])
    else:
        used = np.zeros(pred_points.shape[0], dtype=bool)
        tp = 0
        fn = 0
        for gt_y, gt_x in gt_points:
            d = np.sqrt(np.square(pred_points[:, 0] - gt_y) + np.square(pred_points[:, 1] - gt_x))
            d[used] = np.inf
            idx = int(np.argmin(d))
            if math.isfinite(float(d[idx])) and float(d[idx]) < radius:
                used[idx] = True
                tp += 1
            else:
                fn += 1
        fp = int((~used).sum())
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def split_merge_counts(gt_inst: np.ndarray, pred_inst: np.ndarray) -> dict[str, int]:
    split_count = 0
    merge_count = 0
    for gt_id in [int(v) for v in np.unique(gt_inst) if int(v) > 0]:
        overlap = np.unique(pred_inst[gt_inst == gt_id])
        overlap = [int(v) for v in overlap if int(v) > 0]
        if len(overlap) > 1:
            split_count += 1
    for pred_id in [int(v) for v in np.unique(pred_inst) if int(v) > 0]:
        overlap = np.unique(gt_inst[pred_inst == pred_id])
        overlap = [int(v) for v in overlap if int(v) > 0]
        if len(overlap) > 1:
            merge_count += 1
    return {"split_error_count": split_count, "merge_error_count": merge_count}


def compute_instance_metrics(pred_inst: np.ndarray, gt_inst: np.ndarray, gt_point_map: np.ndarray) -> dict[str, Any]:
    pred_inst = relabel_contiguous(pred_inst)
    gt_inst = relabel_contiguous(gt_inst)
    dq, sq, pq = seg_metrics.get_pq(gt_inst, pred_inst, remap=False)
    det = detection_f1_from_instances(pred_inst, gt_point_map, radius=15.0)
    split_merge = split_merge_counts(gt_inst, pred_inst)
    pred_count = int(pred_inst.max())
    gt_count = int(gt_inst.max())
    return {
        "binary_dice": binary_dice(pred_inst, gt_inst),
        "aji": float(seg_metrics.AJI_fast(gt_inst, pred_inst)),
        "dq": float(dq),
        "sq": float(sq),
        "pq": float(pq),
        "detection_f1": float(det["f1"]),
        "detection_precision": float(det["precision"]),
        "detection_recall": float(det["recall"]),
        "pred_instance_count": pred_count,
        "gt_instance_count": gt_count,
        "pred_gt_count_ratio": safe_div(pred_count, gt_count),
        **split_merge,
    }


def mean_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    out = []
    for group, items in grouped.items():
        summary = {group_key: group, "image_id": "__mean__"}
        keys = [k for k in items[0].keys() if k not in (group_key, "image_id", "split")]
        for key in keys:
            values = [item[key] for item in items if isinstance(item.get(key), (int, float))]
            if values:
                summary[key] = float(np.mean(values))
        out.append(summary)
    return out


def write_step_report(path: Path, title: str, bullets: list[str]) -> None:
    lines = [f"# {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.append("")
    save_text(path, "\n".join(lines))
