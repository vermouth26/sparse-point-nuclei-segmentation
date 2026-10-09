from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
VIT_DIR = REPO_ROOT / "experiments" / "vit_region_prior_sam"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from common import (  # noqa: E402
    build_stage1_runtime,
    compute_detection_counts,
    get_probmaps,
    get_transforms,
    gt_point_path,
    image_path,
    load_split,
    load_stage1_model_from_checkpoint,
    merge_with_partial,
    partial_point_path,
    read_rgb,
    threshold_to_point_map,
    verify_dataset_layout,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Stage1 epoch and BN-recalibrated operating point using MoNuSeg val only."
    )
    parser.add_argument("--phase", choices=["epoch", "operating"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Optional external raw-data root. Defaults to <project-root>/data/raw.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Optional external processed-data root. Defaults to <project-root>/data/processed.",
    )
    parser.add_argument("--dataset", default="MO")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--stage1-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision-floor", type=float, default=0.84)
    parser.add_argument(
        "--no-precision-floor",
        action="store_true",
        help="For epoch selection only, select maximum macro F1 across all epochs; exact ties use the earlier epoch.",
    )
    parser.add_argument("--epoch-threshold", type=float, default=0.65)
    parser.add_argument("--epoch-min-area", type=int, default=12)
    parser.add_argument("--epoch-max-area", type=int, default=150)
    parser.add_argument("--thresholds", default="0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75")
    parser.add_argument("--min-areas", default="8,12,16")
    parser.add_argument("--max-areas", default="100,150,200")
    parser.add_argument("--merge-radius", type=int, default=15)
    parser.add_argument("--matching-radius", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"checkpoint_(\d+)\.pth\.tar", path.name)
    if match:
        return int(match.group(1))
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and payload.get("epoch") is not None:
        return int(payload["epoch"])
    return -1


def discover_epoch_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_*.pth.tar"):
        if path.name == "checkpoint_best.pth.tar" or "precision_guard" in path.name:
            continue
        epoch = checkpoint_epoch(path)
        if epoch >= 0:
            found.append((epoch, path.resolve()))
    found.sort(key=lambda item: (item[0], str(item[1])))
    if not found:
        raise SystemExit(f"No per-epoch checkpoints found under: {checkpoint_dir}")
    return found


def prepare_val_data(layout: dict[str, Path], val_ids: list[str], transform) -> list[dict[str, object]]:
    rows = []
    for image_id in val_ids:
        image = Image.fromarray(read_rgb(image_path(layout["dataset_dir"], image_id)))
        tensor = transform((image,))[0].unsqueeze(0)
        rows.append(
            {
                "image_id": image_id,
                "tensor": tensor,
                "partial": imageio.imread(partial_point_path(layout["ratio_dir"], image_id)).astype(np.uint8),
                "gt": imageio.imread(gt_point_path(layout["dataset_dir"], image_id)).astype(np.uint8),
            }
        )
    return rows


@torch.no_grad()
def infer_prob_maps(checkpoint: Path, opt, device: torch.device, val_data: list[dict[str, object]]) -> list[np.ndarray]:
    model, _ = load_stage1_model_from_checkpoint(checkpoint, opt, device)
    outputs = []
    for item in val_data:
        # split_forward pads on CPU and transfers each patch internally. Keep the
        # full tensor on CPU so its padding tensors stay on the same device.
        tensor = item["tensor"]
        outputs.append(get_probmaps(tensor, model, opt).astype(np.float32))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def evaluate_config(
    prob_maps: list[np.ndarray],
    val_data: list[dict[str, object]],
    threshold: float,
    min_area: int,
    max_area: int,
    merge_radius: int,
    matching_radius: int,
) -> dict[str, float | int]:
    per_image = []
    total_final = 0
    total_gt = 0
    for prob_map, item in zip(prob_maps, val_data):
        pseudo = threshold_to_point_map(prob_map, threshold, min_area, max_area)
        merged = merge_with_partial(item["partial"], pseudo, merge_radius)
        metrics = compute_detection_counts(merged, item["gt"], radius=matching_radius)
        pred_n = int(np.count_nonzero(merged))
        gt_n = int(np.count_nonzero(item["gt"]))
        total_final += pred_n
        total_gt += gt_n
        per_image.append(
            {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "pred_gt": pred_n / max(gt_n, 1),
            }
        )
    mean = lambda key: float(np.mean([row[key] for row in per_image]))
    return {
        "image_count": len(per_image),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "pred_gt_ratio": mean("pred_gt"),
        "global_pred_gt_ratio": total_final / max(total_gt, 1),
        "pred_points": total_final,
        "gt_points": total_gt,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    raw_root = args.raw_root.resolve() if args.raw_root is not None else root / "data" / "raw"
    processed_root = (
        args.processed_root.resolve() if args.processed_root is not None else root / "data" / "processed"
    )
    layout = verify_dataset_layout(raw_root, processed_root, args.dataset, args.ratio, args.description)
    split = load_split(layout["split_file"])
    val_ids = list(split.get("val", []))
    if len(val_ids) != 6:
        raise SystemExit(f"Expected MoNuSeg val6, found {len(val_ids)} images")

    config_path = args.stage1_config.resolve()
    _, opt = build_stage1_runtime(config_path, raw_root=raw_root, processed_root=processed_root)
    if not opt.transform.get("test"):
        raise SystemExit("Stage1 test transform is unavailable; check processed mean/std.")
    transform = get_transforms(opt.transform["test"])
    val_data = prepare_val_data(layout, val_ids, transform)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if args.phase == "epoch":
        if args.checkpoint_dir is None:
            raise SystemExit("--checkpoint-dir is required for --phase epoch")
        for epoch, checkpoint in discover_epoch_checkpoints(args.checkpoint_dir.resolve()):
            prob_maps = infer_prob_maps(checkpoint, opt, device, val_data)
            metrics = evaluate_config(
                prob_maps,
                val_data,
                args.epoch_threshold,
                args.epoch_min_area,
                args.epoch_max_area,
                args.merge_radius,
                args.matching_radius,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "threshold": args.epoch_threshold,
                    "min_area": args.epoch_min_area,
                    "max_area": args.epoch_max_area,
                    **metrics,
                    "eligible": bool(float(metrics["precision"]) > args.precision_floor),
                }
            )
            print(f"[epoch {epoch:03d}] P={metrics['precision']:.6f} R={metrics['recall']:.6f} F1={metrics['f1']:.6f}")
        if args.no_precision_floor:
            candidates = rows
            # Current rule: maximize macro F1 and use the earlier exact tie.
            best = min(candidates, key=lambda row: (-float(row["f1"]), int(row["epoch"])))
        else:
            candidates = [row for row in rows if row["eligible"]]
            if not candidates:
                raise SystemExit(f"No epoch satisfies validation precision > {args.precision_floor}")
            # Legacy P084 rule: F1, precision, then the earlier epoch.
            best = max(candidates, key=lambda row: (row["f1"], row["precision"], -row["epoch"]))
        csv_name = "epoch_val_sweep.csv"
        json_name = "selected_epoch.json"
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --phase operating")
        checkpoint = args.checkpoint.resolve()
        prob_maps = infer_prob_maps(checkpoint, opt, device, val_data)
        for threshold in parse_float_list(args.thresholds):
            for min_area in parse_int_list(args.min_areas):
                for max_area in parse_int_list(args.max_areas):
                    if max_area <= min_area:
                        continue
                    metrics = evaluate_config(
                        prob_maps,
                        val_data,
                        threshold,
                        min_area,
                        max_area,
                        args.merge_radius,
                        args.matching_radius,
                    )
                    rows.append(
                        {
                            "checkpoint": str(checkpoint),
                            "threshold": threshold,
                            "min_area": min_area,
                            "max_area": max_area,
                            **metrics,
                            "eligible": bool(float(metrics["precision"]) >= args.precision_floor),
                        }
                    )
        candidates = [row for row in rows if row["eligible"]]
        if not candidates:
            raise SystemExit(f"No operating point satisfies validation precision >= {args.precision_floor}")
        best = max(
            candidates,
            key=lambda row: (
                row["f1"],
                row["precision"],
                -abs(row["pred_gt_ratio"] - 1.0),
                -row["threshold"],
            ),
        )
        csv_name = "operating_point_val_sweep.csv"
        json_name = "selected_operating_point.json"

    write_csv(out_dir / csv_name, rows)
    payload = {
        "protocol": "MoNuSeg val-only uniform P084 Stage1 selection",
        "phase": args.phase,
        "ratio": args.ratio,
        "description": args.description,
        "precision_floor": None if args.no_precision_floor and args.phase == "epoch" else args.precision_floor,
        "checkpoint_selection_rule": (
            "maximum validation macro detection F1; exact tie -> earlier epoch"
            if args.no_precision_floor and args.phase == "epoch"
            else "legacy precision-floor selection"
        ),
        "validation_images": val_ids,
        "test_gt_used": False,
        "candidate_count": len(rows),
        "eligible_count": len(candidates),
        "best": best,
    }
    (out_dir / json_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
