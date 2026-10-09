from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from common import (
    build_overlay_points,
    build_stage1_runtime,
    choose_images,
    compute_detection_counts,
    discover_root,
    ensure_dir,
    get_probmaps,
    get_transforms,
    gt_point_path,
    image_path,
    load_split,
    load_stage1_model_from_checkpoint,
    merge_with_partial,
    parse_explicit_images,
    parse_split_names,
    parser_common,
    partial_point_path,
    point_count,
    point_map_to_coords,
    read_rgb,
    resolve_stage1_config,
    save_csv,
    save_json,
    save_png,
    threshold_to_point_map,
    verify_dataset_layout,
    write_step_report,
)


def parse_args() -> argparse.Namespace:
    parser = parser_common(
        "Generate or reuse merged Stage-I center maps for pseudo-instance construction.",
        default_split="train",
        default_max_images=3,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-area", type=int, default=150)
    parser.add_argument("--merge-radius", type=int, default=15)
    parser.add_argument("--reuse-existing-pred", action="store_true", help="Reuse source ratio labels_point/pred/*.png when available.")
    parser.add_argument("--reuse-pred-dir", default="", help="Explicit directory containing *_label_point.png merged point maps.")
    parser.add_argument("--reuse-candidate-dir", default="", help="Optional detection binary directory with *_pred.png for raw pseudo candidate counting.")
    parser.add_argument(
        "--no-gt-diagnostics",
        action="store_true",
        help="Do not read full point GT or write GT-derived metrics/overlays. Required for strict TRAIN export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = discover_root("raw", args.raw_root, args.dataset)
    processed_root = discover_root("processed", args.processed_root, args.dataset)
    layout = verify_dataset_layout(raw_root, processed_root, args.dataset, args.ratio, args.description)
    stage1_config = resolve_stage1_config(args.stage1_config)
    split_data = load_split(layout["split_file"])
    selected = choose_images(
        split_data,
        parse_split_names(args.split),
        parse_explicit_images(args.images),
        args.max_images,
    )
    if not selected:
        raise SystemExit("No images were selected for center-map preparation.")

    out_root = Path(args.out_dir).resolve()
    step_root = ensure_dir(out_root / "01_points")
    partial_dir = ensure_dir(step_root / "partial_points")
    pseudo_dir = ensure_dir(step_root / "pseudo_points")
    merged_dir = ensure_dir(step_root / "merged_points")
    prob_dir = ensure_dir(step_root / "prob_maps")
    overlay_dir = ensure_dir(step_root / "overlays")

    explicit_reuse_dir = Path(args.reuse_pred_dir).resolve() if args.reuse_pred_dir else None
    candidate_dir = Path(args.reuse_candidate_dir).resolve() if args.reuse_candidate_dir else None
    source_reuse_dir = explicit_reuse_dir or (layout["ratio_dir"] / "labels_point" / "pred" if args.reuse_existing_pred else None)

    cfg, opt = build_stage1_runtime(stage1_config, raw_root=raw_root, processed_root=processed_root)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = None
    checkpoint_payload = None
    if source_reuse_dir is None:
        model, checkpoint_payload = load_stage1_model_from_checkpoint(Path(args.checkpoint).resolve(), opt, device)
        transform = get_transform(opt)
    else:
        transform = None

    records = []
    for item in selected:
        stem = item["image_id"]
        split_name = item["split"]
        image_rgb = read_rgb(image_path(layout["dataset_dir"], stem))
        partial_map = imageio.imread(partial_point_path(layout["ratio_dir"], stem)).astype(np.uint8)
        gt_point_map = None
        if not args.no_gt_diagnostics:
            gt_point_map = imageio.imread(gt_point_path(layout["dataset_dir"], stem)).astype(np.uint8)
        save_png(partial_dir / f"{stem}_label_point.png", partial_map)

        if source_reuse_dir is not None:
            source_point_path = source_reuse_dir / f"{stem}_label_point.png"
            if not source_point_path.exists():
                raise SystemExit(f"Reuse point map not found: {source_point_path}")
            merged_map = imageio.imread(source_point_path).astype(np.uint8)
            added_mask = (merged_map > 0) & (partial_map == 0)
            pseudo_map = np.zeros_like(merged_map, dtype=np.uint8)
            pseudo_map[added_mask] = 255
            raw_candidates = point_count(pseudo_map)
            if candidate_dir is not None:
                cand_path = candidate_dir / f"{stem}_pred.png"
                if cand_path.exists():
                    raw_candidates = int(measure_candidate_count(cand_path))
            prob_map = None
            source_mode = "reuse_existing_pred"
        else:
            image = Image.fromarray(image_rgb)
            tensor = transform((image,))[0].unsqueeze(0)
            with torch.no_grad():
                prob_map = get_probmaps(tensor, model, opt)
            pseudo_map = threshold_to_point_map(prob_map, args.threshold, args.min_area, args.max_area)
            merged_map = merge_with_partial(partial_map, pseudo_map, args.merge_radius)
            raw_candidates = point_count(pseudo_map)
            np.save(prob_dir / f"{stem}_prob.npy", prob_map.astype(np.float32))
            source_mode = "checkpoint_inference"

        save_png(pseudo_dir / f"{stem}_label_point.png", pseudo_map.astype(np.uint8))
        save_png(merged_dir / f"{stem}_label_point.png", merged_map.astype(np.uint8))

        partial_n = point_count(partial_map)
        final_n = point_count(merged_map)
        added_n = max(0, final_n - partial_n)
        gt_n = point_count(gt_point_map) if gt_point_map is not None else None
        det_metrics = compute_detection_counts(merged_map, gt_point_map, radius=11) if gt_point_map is not None else None
        overlay = build_overlay_points(
            image_rgb,
            point_map_to_coords(partial_map),
            point_map_to_coords(pseudo_map),
            point_map_to_coords(merged_map),
            point_map_to_coords(gt_point_map) if gt_point_map is not None else None,
        )
        save_png(overlay_dir / f"{stem}_points_overlay.png", overlay)
        records.append(
            {
                "image_id": stem,
                "split": split_name,
                "source_mode": source_mode,
                "original_partial_points": partial_n,
                "raw_pseudo_candidates": raw_candidates,
                "added_pseudo_points": added_n,
                "final_points": final_n,
                "gt_instances": gt_n,
                "final_points_over_gt_instances": round(final_n / max(gt_n, 1), 6) if gt_n is not None else None,
                "threshold": float(args.threshold),
                "min_area": int(args.min_area),
                "max_area": int(args.max_area),
                "merge_radius": int(args.merge_radius),
                "final_detection_precision_r11": round(det_metrics["precision"], 6) if det_metrics is not None else None,
                "final_detection_recall_r11": round(det_metrics["recall"], 6) if det_metrics is not None else None,
                "final_detection_f1_r11": round(det_metrics["f1"], 6) if det_metrics is not None else None,
            }
        )

    summary = {
        "dataset": args.dataset,
        "ratio": args.ratio,
        "description": args.description,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "stage1_config": str(stage1_config),
        "checkpoint": str(Path(args.checkpoint).resolve()) if source_reuse_dir is None else None,
        "checkpoint_epoch": checkpoint_payload.get("epoch") if isinstance(checkpoint_payload, dict) else None,
        "gt_diagnostics_enabled": not args.no_gt_diagnostics,
        "train_hidden_gt_read": False if args.no_gt_diagnostics else None,
        "selection": selected,
        "records": records,
        "paths": {
            "partial_dir": str(partial_dir),
            "pseudo_dir": str(pseudo_dir),
            "merged_dir": str(merged_dir),
            "prob_dir": str(prob_dir),
            "overlay_dir": str(overlay_dir),
        },
    }
    save_json(step_root / "prepare_stage1_points_manifest.json", summary)
    save_csv(step_root / "prepare_stage1_points_stats.csv", records)
    write_step_report(
        step_root / "prepare_stage1_points_report.md",
        "Stage1 Point Preparation",
        [
            f"dataset={args.dataset}, ratio={args.ratio}, description={args.description}",
            f"selected_images={', '.join(item['image_id'] for item in selected)}",
            f"source_mode={'reuse_existing_pred' if source_reuse_dir is not None else 'checkpoint_inference'}",
            f"checkpoint={Path(args.checkpoint).resolve() if source_reuse_dir is None else 'not used'}",
            f"threshold={args.threshold}, min_area={args.min_area}, max_area={args.max_area}, merge_radius={args.merge_radius}",
            f"gt_diagnostics_enabled={not args.no_gt_diagnostics}",
        ],
    )
    print(f"Saved Stage-I center preparation outputs to: {step_root}")


def get_transform(opt):
    if not opt.transform.get("test"):
        raise SystemExit(
            "Stage1 test transform is unavailable because mean/std was not found under the processed dataset root. "
            "Please pass a valid --processed-root."
        )
    return get_transforms(opt.transform["test"])


def measure_candidate_count(path: Path) -> int:
    from skimage import measure

    arr = imageio.imread(path)
    return int(len(measure.regionprops(measure.label(arr > 0))))


if __name__ == "__main__":
    main()
