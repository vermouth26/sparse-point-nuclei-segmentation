from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MATCHING_RADIUS = 15.0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def list_npy_ids(path: Path) -> list[str]:
    return sorted(p.stem for p in path.glob("*.npy"))


def load_np_prob(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D NP probability map or 1xHxW, got {arr.shape} from {path}")
    return arr


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(pred_inst: np.ndarray, gt_inst: np.ndarray, matching_radius: float) -> dict[str, float]:
    pred = remap_instances(pred_inst)
    gt = remap_instances(gt_inst)
    dice = dice_from_binary(pred, gt)
    aji = aji_fast(gt, pred)
    dq, sq, pq = pq_from_instances(gt, pred)
    out = {"dice": float(dice), "aji": float(aji), "dq": float(dq), "sq": float(sq), "pq": float(pq)}
    out.update(detection_f1_from_instances(pred, gt, matching_radius=matching_radius))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "dice",
        "aji",
        "dq",
        "sq",
        "pq",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "pred_gt_instance_ratio",
    ]
    out: dict[str, Any] = {"image_count": len(rows)}
    for key in metric_keys:
        out[f"mean_{key}"] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    out["sum_pred_instances"] = int(sum(row["pred_instance_count"] for row in rows))
    out["sum_gt_instances"] = int(sum(row["gt_instance_count"] for row in rows))
    out["global_pred_gt_instance_ratio"] = safe_div(out["sum_pred_instances"], out["sum_gt_instances"])
    return out


def dice_from_binary(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    pred = pred_inst > 0
    gt = gt_inst > 0
    denom = float(pred.sum() + gt.sum())
    return float(2.0 * np.logical_and(pred, gt).sum() / denom) if denom else 1.0


def aji_fast(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    gt_ids = [int(v) for v in np.unique(gt_inst) if int(v) > 0]
    pred_ids = [int(v) for v in np.unique(pred_inst) if int(v) > 0]
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0
    pred_used: set[int] = set()
    overall_inter = 0.0
    overall_union = 0.0
    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id
        overlap_pred = [int(v) for v in np.unique(pred_inst[gt_mask]) if int(v) > 0]
        best_pred = None
        best_iou = -1.0
        best_inter = 0
        best_union = int(gt_mask.sum())
        for pred_id in overlap_pred:
            if pred_id in pred_used:
                continue
            pred_mask = pred_inst == pred_id
            inter = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_pred = pred_id
                best_iou = iou
                best_inter = inter
                best_union = union
        if best_pred is not None:
            pred_used.add(best_pred)
            overall_inter += best_inter
            overall_union += best_union
        else:
            overall_union += int(gt_mask.sum())
    for pred_id in pred_ids:
        if pred_id not in pred_used:
            overall_union += int((pred_inst == pred_id).sum())
    return float(overall_inter / overall_union) if overall_union else 1.0


def pq_from_instances(gt_inst: np.ndarray, pred_inst: np.ndarray, iou_threshold: float = 0.5) -> tuple[float, float, float]:
    gt_ids = [int(v) for v in np.unique(gt_inst) if int(v) > 0]
    pred_ids = [int(v) for v in np.unique(pred_inst) if int(v) > 0]
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0
    pairs = []
    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id
        overlap_pred = [int(v) for v in np.unique(pred_inst[gt_mask]) if int(v) > 0]
        for pred_id in overlap_pred:
            pred_mask = pred_inst == pred_id
            inter = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = inter / union if union else 0.0
            if iou > iou_threshold:
                pairs.append((iou, gt_id, pred_id))
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    iou_sum = 0.0
    for iou, gt_id, pred_id in pairs:
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        iou_sum += iou
    tp = len(matched_gt)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    dq = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    sq = iou_sum / tp if tp else 0.0
    return float(dq), float(sq), float(dq * sq)


def detection_f1_from_instances(pred_inst: np.ndarray, gt_inst: np.ndarray, matching_radius: float) -> dict[str, float]:
    pred_centers = instance_centroids(pred_inst)
    gt_centers = instance_centroids(gt_inst)
    if not pred_centers and not gt_centers:
        return {"detection_precision": 1.0, "detection_recall": 1.0, "detection_f1": 1.0}
    pairs = []
    for gi, gyx in enumerate(gt_centers):
        for pi, pyx in enumerate(pred_centers):
            dist = float(np.linalg.norm(np.asarray(gyx) - np.asarray(pyx)))
            if dist <= matching_radius:
                pairs.append((dist, gi, pi))
    pairs.sort()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    tp = len(matched_gt)
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"detection_precision": float(precision), "detection_recall": float(recall), "detection_f1": float(f1)}


def instance_centroids(inst_map: np.ndarray) -> list[tuple[float, float]]:
    centers = []
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        ys, xs = np.nonzero(inst_map == inst_id)
        if len(ys):
            centers.append((float(ys.mean()), float(xs.mean())))
    return centers


def make_overlay(image: np.ndarray, inst_map: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(23)
    table = rng.integers(40, 255, size=(max(int(inst_map.max()) + 1, 2), 3), dtype=np.uint8)
    table[0] = 0
    color = table[np.clip(inst_map.astype(np.int32), 0, table.shape[0] - 1)]
    return (0.45 * image.astype(np.float32) + 0.55 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_dicts(path, rows)


def remap_instances(inst: np.ndarray) -> np.ndarray:
    out = np.zeros(inst.shape, dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        out[inst == inst_id] = next_id
        next_id += 1
    return out


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def instance_confidence(row: dict[str, Any]) -> float:
    area = max(parse_float(row.get("area")), 1.0)
    np_mean = parse_float(row.get("np_mean"))
    np_p10 = parse_float(row.get("np_p10"))
    center_flag = parse_float(row.get("center_inside_flag"))
    compactness = parse_float(row.get("compactness"))
    area_term = min(math.log1p(area) / math.log1p(500.0), 1.0)
    return 0.50 * np_mean + 0.20 * np_p10 + 0.15 * area_term + 0.10 * center_flag + 0.05 * compactness


def group_features_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_id"]), []).append(row)
    return grouped


def get_label_path(label_dir: Path, image_id: str) -> Path:
    npy = label_dir / f"{image_id}_label.npy"
    if npy.exists():
        return npy
    plain = label_dir / f"{image_id}.npy"
    if plain.exists():
        return plain
    raise FileNotFoundError(f"Missing GT label for {image_id} in {label_dir}")


def evaluate_pruned_dir(
    *,
    pred_dir: Path,
    label_dir: Path,
    image_ids: list[str],
    matching_radius: float = DEFAULT_MATCHING_RADIUS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        pred = np.load(pred_dir / f"{image_id}.npy", allow_pickle=False)
        gt = np.load(get_label_path(label_dir, image_id), allow_pickle=False)
        metrics = compute_metrics(pred, gt, matching_radius=matching_radius)
        rows.append(
            {
                "image_id": image_id,
                "pred_instance_count": int(pred.max()),
                "gt_instance_count": int(gt.max()),
                "pred_gt_instance_ratio": safe_div(float(pred.max()), float(gt.max())),
                **metrics,
            }
        )
    return rows, summarize(rows)


def load_baseline_overall(path: Path) -> dict[str, float]:
    data = load_json(path)
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
