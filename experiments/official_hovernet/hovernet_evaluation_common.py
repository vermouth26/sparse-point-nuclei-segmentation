from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common_official_hovernet import mode_shapes, relabel_contiguous, safe_div


def sliding_window_predict_official(
    model: torch.nn.Module,
    image_u8: np.ndarray,
    device: torch.device,
    mode: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_shape, output_shape = mode_shapes(mode)
    margin = (input_shape - output_shape) // 2
    height, width = image_u8.shape[:2]
    padded = np.pad(image_u8, [(margin, margin), (margin, margin), (0, 0)], mode="reflect")
    np_canvas = np.zeros((height, width), dtype=np.float32)
    hv_canvas = np.zeros((2, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    jobs = []
    for top in tile_positions(height, output_shape):
        for left in tile_positions(width, output_shape):
            bottom = min(height, top + output_shape)
            right = min(width, left + output_shape)
            patch = padded[top : top + input_shape, left : left + input_shape]
            jobs.append((top, left, bottom, right, patch))

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        batch = np.stack([job[4].transpose(2, 0, 1) for job in chunk], axis=0)
        tensor = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            prediction = model(tensor)
            np_prob = torch.softmax(prediction["np"], dim=1)[:, 1].detach().cpu().numpy()
            hv_pred = prediction["hv"].detach().cpu().numpy()
        for index, (top, left, bottom, right, _) in enumerate(chunk):
            out_h = bottom - top
            out_w = right - left
            np_canvas[top:bottom, left:right] += np_prob[index, :out_h, :out_w]
            hv_canvas[:, top:bottom, left:right] += hv_pred[index, :, :out_h, :out_w]
            weight[top:bottom, left:right] += 1.0

    weight = np.maximum(weight, 1.0e-6)
    return np_canvas / weight, hv_canvas / weight[None, ...]


def tile_positions(length: int, output_shape: int) -> list[int]:
    if length <= output_shape:
        return [0]
    positions = list(range(0, max(length - output_shape + 1, 1), output_shape))
    last = length - output_shape
    if positions[-1] != last:
        positions.append(last)
    return positions


def compute_metrics(pred_inst: np.ndarray, gt_inst: np.ndarray, matching_radius: float) -> dict[str, float]:
    pred = relabel_contiguous(pred_inst)
    gt = relabel_contiguous(gt_inst)
    dice = dice_from_binary(pred, gt)
    aji = aji_fast(gt, pred)
    dq, sq, pq = pq_from_instances(gt, pred)
    metrics = {"dice": float(dice), "aji": float(aji), "dq": float(dq), "sq": float(sq), "pq": float(pq)}
    metrics.update(detection_f1_from_instances(pred, gt, matching_radius=matching_radius))
    return metrics


def dice_from_binary(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    pred = pred_inst > 0
    gt = gt_inst > 0
    denominator = float(pred.sum() + gt.sum())
    return float(2.0 * np.logical_and(pred, gt).sum() / denominator) if denominator else 1.0


def aji_fast(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    gt_ids = [int(value) for value in np.unique(gt_inst) if int(value) > 0]
    pred_ids = [int(value) for value in np.unique(pred_inst) if int(value) > 0]
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0
    pred_used: set[int] = set()
    overall_inter = 0.0
    overall_union = 0.0
    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id
        overlap_pred = [int(value) for value in np.unique(pred_inst[gt_mask]) if int(value) > 0]
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


def pq_from_instances(
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    gt_ids = [int(value) for value in np.unique(gt_inst) if int(value) > 0]
    pred_ids = [int(value) for value in np.unique(pred_inst) if int(value) > 0]
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0
    pairs = []
    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id
        overlap_pred = [int(value) for value in np.unique(pred_inst[gt_mask]) if int(value) > 0]
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


def detection_f1_from_instances(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    matching_radius: float,
) -> dict[str, float]:
    pred_centers = instance_centroids(pred_inst)
    gt_centers = instance_centroids(gt_inst)
    if not pred_centers and not gt_centers:
        return {"detection_precision": 1.0, "detection_recall": 1.0, "detection_f1": 1.0}
    pairs = []
    for gt_index, gt_center in enumerate(gt_centers):
        for pred_index, pred_center in enumerate(pred_centers):
            distance = float(np.linalg.norm(np.asarray(gt_center) - np.asarray(pred_center)))
            if distance <= matching_radius:
                pairs.append((distance, gt_index, pred_index))
    pairs.sort()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in pairs:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    tp = len(matched_gt)
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"detection_precision": float(precision), "detection_recall": float(recall), "detection_f1": float(f1)}


def instance_centroids(inst_map: np.ndarray) -> list[tuple[float, float]]:
    centers = []
    for inst_id in [int(value) for value in np.unique(inst_map) if int(value) > 0]:
        ys, xs = np.nonzero(inst_map == inst_id)
        if len(ys):
            centers.append((float(ys.mean()), float(xs.mean())))
    return centers


def summarize(rows: list[dict]) -> dict:
    metric_keys = [
        "dice", "aji", "dq", "sq", "pq", "detection_precision",
        "detection_recall", "detection_f1", "pred_gt_instance_ratio",
    ]
    out = {"image_count": len(rows)}
    for key in metric_keys:
        out[f"mean_{key}"] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    out["sum_pred_instances"] = int(sum(row["pred_instance_count"] for row in rows))
    out["sum_gt_instances"] = int(sum(row["gt_instance_count"] for row in rows))
    out["global_pred_gt_instance_ratio"] = safe_div(out["sum_pred_instances"], out["sum_gt_instances"])
    return out


def vars_for_json(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
