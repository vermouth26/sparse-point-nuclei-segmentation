from __future__ import annotations

import csv
import json
import math
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def save_png(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(array).save(path)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stem_to_paths(prepared_root: Path, raw_root: Path, stem: str) -> dict[str, Path]:
    return {
        "image": raw_root / "images" / f"{stem}.png",
        "gt_instance": raw_root / "labels_instance" / f"{stem}_label.npy",
        "gt_point": raw_root / "labels_point" / f"{stem}_label_point.png",
        "labels_clu": prepared_root / "labels_clu" / f"{stem}_label_clu.png",
        "labels_vor": prepared_root / "labels_vor" / f"{stem}_label_vor.png",
        "merged_point": prepared_root / "labels_point" / "pred" / f"{stem}_label_point.png",
        "pseudo_only_point": prepared_root / "labels_point_pseudo_only" / f"{stem}_label_point.png",
        "prob_map": prepared_root / "prob_maps" / f"{stem}_prob.npy",
    }


def unique_rgb_counts(rgb: np.ndarray) -> list[dict]:
    flat = rgb.reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(cnt)[::-1]
    return [{"rgb": uniq[i].tolist(), "count": int(cnt[i])} for i in order]


def decode_clu(rgb: np.ndarray) -> dict[str, np.ndarray]:
    bg = rgb[:, :, 0] > 127
    fg = rgb[:, :, 1] > 127
    ignore = ~(bg | fg)
    return {"fg": fg, "bg": bg, "ignore": ignore}


def decode_vor(rgb: np.ndarray) -> dict[str, np.ndarray]:
    edge = rgb[:, :, 0] > 127
    point = rgb[:, :, 1] > 127
    ignore = ~(edge | point)
    return {"edge": edge, "point": point, "ignore": ignore}


def point_map_to_centers(point_map: np.ndarray, score_map: np.ndarray | None = None, source: str = "merged_point") -> list[dict]:
    coords = np.argwhere(point_map > 0)
    centers: list[dict] = []
    for y, x in coords:
        centers.append(
            {
                "x": int(x),
                "y": int(y),
                "score": float(score_map[y, x]) if score_map is not None else 1.0,
                "source": source,
            }
        )
    return centers


def offsets_for_radius(radius: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out.append((dy, dx))
    return out or [(0, 0)]


def binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    for dy, dx in offsets_for_radius(radius):
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
    for dy, dx in offsets_for_radius(radius):
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


def label_boundary(inst_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros(inst_map.shape, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.zeros_like(inst_map)
        ys0 = max(0, -dy)
        ys1 = min(inst_map.shape[0], inst_map.shape[0] - dy)
        yd0 = max(0, dy)
        yd1 = min(inst_map.shape[0], inst_map.shape[0] + dy)
        xs0 = max(0, -dx)
        xs1 = min(inst_map.shape[1], inst_map.shape[1] - dx)
        xd0 = max(0, dx)
        xd1 = min(inst_map.shape[1], inst_map.shape[1] + dx)
        shifted[yd0:yd1, xd0:xd1] = inst_map[ys0:ys1, xs0:xs1]
        boundary |= (inst_map > 0) & (shifted > 0) & (shifted != inst_map)
    return boundary


def nearest_region_pixel(mask: np.ndarray, y: int, x: int, snap_radius: int) -> tuple[int, int] | None:
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return int(y), int(x)
    best: tuple[int, int] | None = None
    best_d = None
    y0 = max(0, y - snap_radius)
    y1 = min(mask.shape[0], y + snap_radius + 1)
    x0 = max(0, x - snap_radius)
    x1 = min(mask.shape[1], x + snap_radius + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if not mask[yy, xx]:
                continue
            d = (yy - y) * (yy - y) + (xx - x) * (xx - x)
            if best_d is None or d < best_d:
                best = (yy, xx)
                best_d = d
    return best


def snap_centers_to_region(centers: list[dict], region_mask: np.ndarray, snap_radius: int) -> tuple[list[dict], int]:
    used = np.zeros(region_mask.shape, dtype=bool)
    assigned: list[dict] = []
    dropped = 0
    for idx, item in enumerate(centers, start=1):
        snapped = nearest_region_pixel(region_mask, int(item["y"]), int(item["x"]), snap_radius)
        if snapped is None:
            dropped += 1
            continue
        yy, xx = snapped
        if used[yy, xx]:
            dropped += 1
            continue
        used[yy, xx] = True
        assigned.append(
            {
                "instance_id": idx,
                "source_x": int(item["x"]),
                "source_y": int(item["y"]),
                "seed_x": int(xx),
                "seed_y": int(yy),
                "score": float(item.get("score", 1.0)),
                "source": str(item.get("source", "merged_point")),
            }
        )
    return assigned, dropped


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    dist = np.zeros(mask.shape, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    boundary = mask_boundary(mask)
    for y, x in np.argwhere(boundary):
        dist[y, x] = 1
        q.append((int(y), int(x)))
    for y, x in q:
        pass
    while q:
        y, x = q.popleft()
        base = dist[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= mask.shape[0] or xx < 0 or xx >= mask.shape[1]:
                continue
            if not mask[yy, xx] or dist[yy, xx] > 0:
                continue
            dist[yy, xx] = base + 1
            q.append((yy, xx))
    return dist


def grow_instances_watershed_like(region_mask: np.ndarray, seeds: list[dict]) -> np.ndarray:
    dist = boundary_distance(region_mask)
    inst_map = np.zeros(region_mask.shape, dtype=np.int32)
    heap: list[tuple[int, int, int, int]] = []
    for order, item in enumerate(seeds):
        y = int(item["seed_y"])
        x = int(item["seed_x"])
        inst_id = int(item["instance_id"])
        inst_map[y, x] = inst_id
        heappush(heap, (-int(dist[y, x]), order, y, x))
    while heap:
        _, order, y, x = heappop(heap)
        inst_id = int(inst_map[y, x])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= region_mask.shape[0] or xx < 0 or xx >= region_mask.shape[1]:
                continue
            if not region_mask[yy, xx] or inst_map[yy, xx] != 0:
                continue
            inst_map[yy, xx] = inst_id
            heappush(heap, (-int(dist[yy, xx]), order, yy, xx))
    return inst_map


def grow_instances_voronoi(region_mask: np.ndarray, seeds: list[dict]) -> np.ndarray:
    inst_map = np.zeros(region_mask.shape, dtype=np.int32)
    q: deque[tuple[int, int, int]] = deque()
    for item in seeds:
        y = int(item["seed_y"])
        x = int(item["seed_x"])
        inst_id = int(item["instance_id"])
        inst_map[y, x] = inst_id
        q.append((y, x, inst_id))
    while q:
        y, x, inst_id = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= region_mask.shape[0] or xx < 0 or xx >= region_mask.shape[1]:
                continue
            if not region_mask[yy, xx] or inst_map[yy, xx] != 0:
                continue
            inst_map[yy, xx] = inst_id
            q.append((yy, xx, inst_id))
    return inst_map


def remove_small_instances(inst_map: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    out = inst_map.copy()
    removed = 0
    for inst_id in [int(v) for v in np.unique(out) if int(v) > 0]:
        area = int(np.sum(out == inst_id))
        if area < min_area:
            out[out == inst_id] = 0
            removed += 1
    return relabel_contiguous(out), removed


def relabel_contiguous(inst_map: np.ndarray) -> np.ndarray:
    out = np.zeros(inst_map.shape, dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        out[inst_map == inst_id] = next_id
        next_id += 1
    return out


def make_np_targets(region_fg: np.ndarray, clu_bg: np.ndarray, uncertain: np.ndarray, vor_edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundary = mask_boundary(region_fg)
    np_map = region_fg.astype(np.uint8)
    np_ignore = (uncertain | boundary | binary_dilation(vor_edge, 1)) & (~clu_bg)
    return np_map, np_ignore.astype(np.uint8)


def make_hv_targets(inst_map: np.ndarray, region_fg: np.ndarray, uncertain: np.ndarray, vor_edge: np.ndarray, erosion_radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    hv_map = np.zeros(inst_map.shape + (2,), dtype=np.float32)
    reliable_region = binary_erosion(region_fg, erosion_radius)
    inst_boundary = label_boundary(inst_map)
    competition = binary_dilation(vor_edge, 1)
    hv_valid = reliable_region & (inst_map > 0) & (~inst_boundary) & (~competition) & (~uncertain)
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        mask = inst_map == inst_id
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        max_abs_x = max(float(np.max(np.abs(xs - cx))), 1.0)
        max_abs_y = max(float(np.max(np.abs(ys - cy))), 1.0)
        valid = hv_valid & mask
        vy, vx = np.where(valid)
        hv_map[vy, vx, 0] = ((vx.astype(np.float32) - np.float32(cx)) / np.float32(max_abs_x)).astype(np.float32)
        hv_map[vy, vx, 1] = ((vy.astype(np.float32) - np.float32(cy)) / np.float32(max_abs_y)).astype(np.float32)
    hv_ignore = (~hv_valid).astype(np.uint8)
    return hv_map, hv_ignore


def center_assignment_success(inst_map: np.ndarray, seeds: list[dict]) -> tuple[int, int]:
    total = len(seeds)
    ok = 0
    for item in seeds:
        ok += int(inst_map[int(item["seed_y"]), int(item["seed_x"])] > 0)
    return ok, total


def colorize_instances(inst_map: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(17)
    table = rng.integers(40, 255, size=(max(int(inst_map.max()) + 1, 2), 3), dtype=np.uint8)
    table[0] = 0
    return table[np.clip(inst_map, 0, table.shape[0] - 1)]


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    idx = mask > 0
    out[idx] = (1.0 - alpha) * out[idx] + alpha * np.asarray(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def draw_centers(image: np.ndarray, centers: Iterable[dict], radius: int = 4, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    canvas = Image.fromarray(image.copy())
    painter = ImageDraw.Draw(canvas)
    for item in centers:
        x = int(item["x"]) if "x" in item else int(item["seed_x"])
        y = int(item["y"]) if "y" in item else int(item["seed_y"])
        painter.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    return np.asarray(canvas)


def render_gray(mask: np.ndarray) -> np.ndarray:
    arr = (mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=2)


def render_hv_magnitude(hv_map: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.square(hv_map[:, :, 0]) + np.square(hv_map[:, :, 1]))
    if float(mag.max()) > 0:
        mag = mag / float(mag.max())
    arr = (mag * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=2)


def split_instances(inst_map: np.ndarray) -> list[np.ndarray]:
    return [(inst_map == inst_id) for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]]


def dice_from_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred > 0
    gt_bin = gt > 0
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    den = float(pred_bin.sum() + gt_bin.sum())
    return 2.0 * inter / den if den > 0 else 1.0


def aji_fast(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = relabel_contiguous(gt)
    pred = relabel_contiguous(pred)
    gt_ids = [int(v) for v in np.unique(gt) if int(v) > 0]
    pred_ids = [int(v) for v in np.unique(pred) if int(v) > 0]
    if not gt_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    gt_area = np.bincount(gt.reshape(-1))
    pred_area = np.bincount(pred.reshape(-1))
    paired = np.stack([gt.reshape(-1), pred.reshape(-1)], axis=1)
    uniq, cnt = np.unique(paired, axis=0, return_counts=True)
    overlap = {(int(gid), int(pid)): int(cc) for (gid, pid), cc in zip(uniq, cnt) if int(gid) > 0 and int(pid) > 0}
    best_for_gt: dict[int, tuple[int, int, int, float]] = {}
    for (gid, pid), inter in overlap.items():
        union = int(gt_area[gid] + pred_area[pid] - inter)
        iou = inter / union if union > 0 else 0.0
        prev = best_for_gt.get(gid)
        if prev is None or iou > prev[3]:
            best_for_gt[gid] = (pid, inter, union, iou)
    used_pred: set[int] = set()
    inter_sum = 0
    union_sum = 0
    for gid in gt_ids:
        best = best_for_gt.get(gid)
        if best is None:
            union_sum += int(gt_area[gid])
            continue
        pid, inter, union, _ = best
        used_pred.add(pid)
        inter_sum += inter
        union_sum += union
    for pid in pred_ids:
        if pid not in used_pred:
            union_sum += int(pred_area[pid])
    return float(inter_sum) / float(union_sum) if union_sum > 0 else 0.0


def pq_from_instances(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5) -> tuple[float, float, float]:
    gt = relabel_contiguous(gt)
    pred = relabel_contiguous(pred)
    gt_ids = [int(v) for v in np.unique(gt) if int(v) > 0]
    pred_ids = [int(v) for v in np.unique(pred) if int(v) > 0]
    gt_area = np.bincount(gt.reshape(-1))
    pred_area = np.bincount(pred.reshape(-1))
    paired = np.stack([gt.reshape(-1), pred.reshape(-1)], axis=1)
    uniq, cnt = np.unique(paired, axis=0, return_counts=True)
    pairs: list[tuple[float, int, int]] = []
    gt_set = set(gt_ids)
    pred_set = set(pred_ids)
    for (gid, pid), inter in zip(uniq, cnt):
        gid = int(gid)
        pid = int(pid)
        if gid not in gt_set or pid not in pred_set:
            continue
        union = int(gt_area[gid] + pred_area[pid] - int(inter))
        iou = int(inter) / union if union > 0 else 0.0
        if iou > iou_thresh:
            pairs.append((iou, gid, pid))
    pairs.sort(reverse=True)
    matched_g: set[int] = set()
    matched_p: set[int] = set()
    matched_ious: list[float] = []
    for iou, gid, pid in pairs:
        if gid in matched_g or pid in matched_p:
            continue
        matched_g.add(gid)
        matched_p.add(pid)
        matched_ious.append(float(iou))
    tp = len(matched_ious)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 1.0
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    return dq, sq, dq * sq


def detection_f1_from_centers(pred_centers: list[tuple[float, float]], gt_point_map: np.ndarray, radius: float = 15.0) -> tuple[float, float, float, int, int, int]:
    gt_centers = np.argwhere(gt_point_map > 0).astype(np.float32)
    preds = np.asarray(pred_centers, dtype=np.float32)
    if preds.size == 0:
        tp = 0
        fp = 0
        fn = int(gt_centers.shape[0])
    else:
        used = np.zeros(preds.shape[0], dtype=bool)
        tp = 0
        fn = 0
        for gt_y, gt_x in gt_centers:
            if preds.shape[0] == 0:
                fn += 1
                continue
            dist = np.sqrt(np.square(preds[:, 0] - gt_y) + np.square(preds[:, 1] - gt_x))
            dist[used] = np.inf
            idx = int(np.argmin(dist))
            if math.isfinite(float(dist[idx])) and float(dist[idx]) < radius:
                used[idx] = True
                tp += 1
            else:
                fn += 1
        fp = int((~used).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def instance_centroids(inst_map: np.ndarray) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for inst_id in [int(v) for v in np.unique(inst_map) if int(v) > 0]:
        ys, xs = np.where(inst_map == inst_id)
        if ys.size == 0:
            continue
        out.append((float(np.mean(ys)), float(np.mean(xs))))
    return out
