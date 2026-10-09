from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common_official_hovernet import (
    DEFAULT_TEST_LABELS,
    DEFAULT_TRAIN_LABELS,
    ensure_dir,
    read_instance,
    relabel_contiguous,
    safe_div,
    write_csv,
)
from hovernet_evaluation_common import compute_metrics, summarize


VAL_IDS = ["train_image_07", "train_image_15", "train_image_23", "train_image_31"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Val4-only configurable official HoVerNet postprocess sweep for CPM17 fullsup raw outputs.")
    parser.add_argument("--val-raw-dir", type=Path, required=True, help="Directory containing val4 np_prob/ and hv_pred/.")
    parser.add_argument("--test-raw-dir", type=Path, required=True, help="Directory containing test32 np_prob/ and hv_pred/.")
    parser.add_argument("--val-label-dir", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--test-label-dir", type=Path, default=DEFAULT_TEST_LABELS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    validate_raw_dir(args.val_raw_dir, VAL_IDS)
    test_ids = sorted(path.name[: -len("_label.npy")] for path in args.test_label_dir.glob("test_image_*_label.npy"))
    validate_raw_dir(args.test_raw_dir, test_ids)

    default_cfg = cfg(0.50, 10, 21, 0.40, 5, 10)
    baseline_val_rows, baseline_val = evaluate_config(
        raw_dir=args.val_raw_dir,
        label_dir=args.val_label_dir,
        image_ids=VAL_IDS,
        config=default_cfg,
        matching_radius=args.matching_radius,
        pred_out_dir=out_dir / "official_default_val_instance_maps",
    )
    write_csv(out_dir / "official_default_val_metrics.csv", baseline_val_rows)
    (out_dir / "official_default_val_overall.json").write_text(json.dumps(baseline_val, indent=2), encoding="utf-8")

    coarse_rows = evaluate_config_set(
        phase="coarse",
        configs=dedupe_configs(build_coarse_configs()),
        raw_dir=args.val_raw_dir,
        label_dir=args.val_label_dir,
        image_ids=VAL_IDS,
        matching_radius=args.matching_radius,
        baseline=baseline_val,
    )
    coarse_best = select_best(coarse_rows)
    refine_rows = evaluate_config_set(
        phase="refine",
        configs=dedupe_configs(build_refine_configs(coarse_best)),
        raw_dir=args.val_raw_dir,
        label_dir=args.val_label_dir,
        image_ids=VAL_IDS,
        matching_radius=args.matching_radius,
        baseline=baseline_val,
    )
    rows = coarse_rows + refine_rows
    write_csv(out_dir / "official_postprocess_val_sweep.csv", rows)
    selected = select_best(rows)
    (out_dir / "selected_postprocess_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    val_map_dir = out_dir / "official_postprocess_best_val_instance_maps"
    val_rows, val_overall = evaluate_config(
        raw_dir=args.val_raw_dir,
        label_dir=args.val_label_dir,
        image_ids=VAL_IDS,
        config=config_only(selected),
        matching_radius=args.matching_radius,
        pred_out_dir=val_map_dir,
    )
    write_csv(out_dir / "official_postprocess_best_val_metrics.csv", val_rows)
    (out_dir / "official_postprocess_best_val_overall.json").write_text(json.dumps(val_overall, indent=2), encoding="utf-8")

    test_map_dir = out_dir / "official_postprocess_best_test_instance_maps"
    test_rows, test_overall = evaluate_config(
        raw_dir=args.test_raw_dir,
        label_dir=args.test_label_dir,
        image_ids=test_ids,
        config=config_only(selected),
        matching_radius=args.matching_radius,
        pred_out_dir=test_map_dir,
    )
    write_csv(out_dir / "official_postprocess_best_test_metrics.csv", test_rows)
    (out_dir / "official_postprocess_best_test_overall.json").write_text(json.dumps(test_overall, indent=2), encoding="utf-8")
    write_report(out_dir / "OFFICIAL_HOVERNET_POSTPROCESS_SWEEP_REPORT.md", baseline_val, selected, val_overall, test_overall)
    print(json.dumps({"selected_config": config_only(selected), "val": val_overall, "test": test_overall}, indent=2))


def build_coarse_configs() -> list[dict[str, Any]]:
    configs = [cfg(0.50, 10, 21, 0.40, 5, 10)]
    for np_thr in [0.50, 0.55]:
        for blob_min in [10, 20, 30]:
            for overall_thr in [0.40, 0.45, 0.50]:
                for marker_kernel in [5, 7]:
                    for marker_min in [10, 20]:
                        configs.append(cfg(np_thr, blob_min, 21, overall_thr, marker_kernel, marker_min))
    return configs


def build_refine_configs(best: dict[str, Any]) -> list[dict[str, Any]]:
    best_np = float(best["np_threshold"])
    best_blob = int(best["blob_min_size"])
    best_sobel = int(best["sobel_ksize"])
    best_overall = float(best["overall_threshold"])
    best_kernel = int(best["marker_kernel_size"])
    best_marker = int(best["marker_min_size"])
    configs = []
    sobel_alt = 21 if best_sobel != 21 else 15
    kernel_alt = 5 if best_kernel != 5 else 7
    for np_thr in sorted({round(best_np + d, 2) for d in [-0.03, 0.0, 0.03]}):
        if not 0.45 <= np_thr <= 0.60:
            continue
        for blob_min in sorted({max(5, best_blob - 10), best_blob, best_blob + 10}):
            for sobel in sorted({best_sobel, sobel_alt}):
                if sobel % 2 == 0:
                    continue
                for overall_thr in sorted({round(best_overall + d, 2) for d in [-0.05, 0.0, 0.05]}):
                    if not 0.30 <= overall_thr <= 0.60:
                        continue
                    for marker_kernel in sorted({best_kernel, kernel_alt}):
                        if marker_kernel % 2 == 0:
                            continue
                        for marker_min in sorted({max(5, best_marker - 10), best_marker, best_marker + 10}):
                            configs.append(cfg(np_thr, blob_min, sobel, overall_thr, marker_kernel, marker_min))
    return configs


def cfg(np_thr: float, blob_min: int, sobel: int, overall_thr: float, marker_kernel: int, marker_min: int) -> dict[str, Any]:
    return {
        "np_threshold": float(np_thr),
        "blob_min_size": int(blob_min),
        "sobel_ksize": int(sobel),
        "overall_threshold": float(overall_thr),
        "marker_kernel_size": int(marker_kernel),
        "marker_min_size": int(marker_min),
    }


def dedupe_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in configs:
        key = tuple(item[k] for k in ["np_threshold", "blob_min_size", "sobel_ksize", "overall_threshold", "marker_kernel_size", "marker_min_size"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def evaluate_config_set(
    *,
    phase: str,
    configs: list[dict[str, Any]],
    raw_dir: Path,
    label_dir: Path,
    image_ids: list[str],
    matching_radius: float,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for idx, config in enumerate(configs):
        _, overall = evaluate_config(
            raw_dir=raw_dir,
            label_dir=label_dir,
            image_ids=image_ids,
            config=config,
            matching_radius=matching_radius,
            pred_out_dir=None,
        )
        row = {
            "phase": phase,
            "phase_index": idx,
            **config,
            **overall,
            "delta_dice": overall["mean_dice"] - baseline["mean_dice"],
            "delta_aji": overall["mean_aji"] - baseline["mean_aji"],
            "delta_pq": overall["mean_pq"] - baseline["mean_pq"],
            "ratio_abs_error_to_105": abs(overall["global_pred_gt_instance_ratio"] - 1.05),
        }
        row["selection_score"] = selection_score(row)
        rows.append(row)
        print(
            f"[official_hovernet_postprocess:{phase}] {idx:03d} pq={row['mean_pq']:.6f} "
            f"aji={row['mean_aji']:.6f} dq={row['mean_dq']:.6f} ratio={row['global_pred_gt_instance_ratio']:.4f}"
        )
    return rows


def evaluate_config(
    *,
    raw_dir: Path,
    label_dir: Path,
    image_ids: list[str],
    config: dict[str, Any],
    matching_radius: float,
    pred_out_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pred_out_dir is not None:
        ensure_dir(pred_out_dir)
    rows = []
    for image_id in image_ids:
        np_prob = load_np_prob(raw_dir / "np_prob" / f"{image_id}.npy")
        hv_pred = np.load(raw_dir / "hv_pred" / f"{image_id}.npy", allow_pickle=False).astype(np.float32)
        inst = postprocess_official_configurable(np_prob=np_prob, hv_pred=hv_pred, config=config)
        gt = read_instance(label_dir / f"{image_id}_label.npy")
        metrics = compute_metrics(inst, gt, matching_radius=matching_radius)
        rows.append(
            {
                "image_id": image_id,
                "pred_instance_count": int(inst.max()),
                "gt_instance_count": int(gt.max()),
                "pred_gt_instance_ratio": safe_div(float(inst.max()), float(gt.max())),
                **metrics,
            }
        )
        if pred_out_dir is not None:
            np.save(pred_out_dir / f"{image_id}.npy", inst.astype(np.int32))
    return rows, summarize(rows)


def postprocess_official_configurable(np_prob: np.ndarray, hv_pred: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import cv2
    from scipy.ndimage import binary_fill_holes, measurements
    from skimage.segmentation import watershed

    if hv_pred.ndim != 3 or hv_pred.shape[0] != 2:
        raise SystemExit(f"Expected hv_pred shape 2xHxW, got {hv_pred.shape}")
    h_dir_raw = hv_pred[0]
    v_dir_raw = hv_pred[1]
    blb = np.array(np_prob >= float(config["np_threshold"]), dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_labeled(blb, min_size=int(config["blob_min_size"]))
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobel_ksize = int(config["sobel_ksize"])
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall_bin = np.array(overall >= float(config["overall_threshold"]), dtype=np.int32)
    marker = blb - overall_bin
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel_size = int(config["marker_kernel_size"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_labeled(marker, min_size=int(config["marker_min_size"]))
    inst = watershed(dist, markers=marker, mask=blb)
    return relabel_contiguous(inst)


def remove_small_labeled(inst: np.ndarray, min_size: int) -> np.ndarray:
    out = np.zeros(inst.shape, dtype=np.int32)
    next_id = 1
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        mask = inst == inst_id
        if int(mask.sum()) >= min_size:
            out[mask] = next_id
            next_id += 1
    return out


def selection_score(row: dict[str, Any]) -> float:
    score = float(row["mean_pq"])
    score += 0.08 * float(row["mean_aji"])
    score += 0.03 * float(row["mean_dq"])
    score += 0.01 * float(row["mean_detection_precision"])
    score -= 0.02 * float(row["ratio_abs_error_to_105"])
    if float(row["delta_dice"]) < -0.005:
        score -= 1.0
    return score


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if float(row.get("delta_dice", 0.0)) >= -0.005]
    pool = valid if valid else rows
    return max(
        pool,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["mean_pq"]),
            float(row["mean_aji"]),
            float(row["mean_dq"]),
            -float(row["ratio_abs_error_to_105"]),
        ),
    )


def config_only(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ["np_threshold", "blob_min_size", "sobel_ksize", "overall_threshold", "marker_kernel_size", "marker_min_size"]}


def load_np_prob(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise SystemExit(f"Expected 2D NP probability map, got {arr.shape} from {path}")
    return arr


def validate_raw_dir(raw_dir: Path, image_ids: list[str]) -> None:
    missing = []
    for image_id in image_ids:
        for sub in ["np_prob", "hv_pred"]:
            path = raw_dir / sub / f"{image_id}.npy"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing raw outputs:\n" + "\n".join(missing[:20]))


def write_report(path: Path, baseline_val: dict[str, Any], selected: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> None:
    report = f"""# Official HoVerNet Fullsup Postprocess Sweep

## No-Leakage Statement

- Postprocess parameters were selected on CPM17 official val4 only.
- Official test32 GT is used only for the single fixed evaluation with the val-selected config.
- No training is run in this sweep.

## Baseline Val Default

- Dice: `{baseline_val['mean_dice']:.6f}`
- AJI: `{baseline_val['mean_aji']:.6f}`
- PQ: `{baseline_val['mean_pq']:.6f}`
- DQ: `{baseline_val['mean_dq']:.6f}`
- SQ: `{baseline_val['mean_sq']:.6f}`
- global pred/GT: `{baseline_val['global_pred_gt_instance_ratio']:.6f}`

## Selected Config

```json
{json.dumps(config_only(selected), indent=2)}
```

## Selected Val Metrics

- Dice: `{val['mean_dice']:.6f}`
- AJI: `{val['mean_aji']:.6f}`
- PQ: `{val['mean_pq']:.6f}`
- DQ: `{val['mean_dq']:.6f}`
- SQ: `{val['mean_sq']:.6f}`
- global pred/GT: `{val['global_pred_gt_instance_ratio']:.6f}`

## Fixed Test32 Metrics

- Dice: `{test['mean_dice']:.6f}`
- AJI: `{test['mean_aji']:.6f}`
- PQ: `{test['mean_pq']:.6f}`
- DQ: `{test['mean_dq']:.6f}`
- SQ: `{test['mean_sq']:.6f}`
- precision: `{test['mean_detection_precision']:.6f}`
- recall: `{test['mean_detection_recall']:.6f}`
- F1: `{test['mean_detection_f1']:.6f}`
- global pred/GT: `{test['global_pred_gt_instance_ratio']:.6f}`
"""
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
