from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from common import (
    clu_label_to_instance_map,
    create_cluster_label,
    create_voronoi_label,
    ensure_dir,
    gt_instance_path,
    image_path,
    imread,
    read_rgb,
    relabel_contiguous,
    save_csv,
    save_instance_png,
    save_json,
    save_png,
    tile_images,
    verify_dataset_layout,
    vor_label_to_instance_map,
    write_step_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BoNuS baseline labels_vor / labels_clu in an isolated experiment directory.")
    parser.add_argument("--points-manifest", required=True, help="Manifest from 01_prepare_stage1_points.py")
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--no-gt-diagnostics",
        action="store_true",
        help="Do not read full instance GT. Required when building train pseudo-label inputs under a strict no-exposure protocol.",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["original", "copy-voronoi"],
        default="original",
        help=(
            "Use BoNuS original KMeans-based cluster labels, or copy Voronoi labels as "
            "a safe local fallback when sklearn/KMeans native code is unstable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points_manifest = Path(args.points_manifest).resolve()
    if not points_manifest.exists():
        raise SystemExit(f"Points manifest not found: {points_manifest}")
    payload = json.loads(points_manifest.read_text(encoding="utf-8"))
    out_root = Path(args.out_dir).resolve() if args.out_dir else points_manifest.parents[1]
    step_root = ensure_dir(out_root / "02_bonus_region_baseline")
    vor_dir = ensure_dir(step_root / "labels_vor")
    clu_dir = ensure_dir(step_root / "labels_clu")
    vor_inst_dir = ensure_dir(step_root / "bonus_vor_instance")
    clu_inst_dir = ensure_dir(step_root / "bonus_clu_instance")
    overlay_dir = ensure_dir(step_root / "overlays")

    raw_root = Path(payload["raw_root"]).resolve()
    processed_root = Path(payload["processed_root"]).resolve()
    dataset = payload["dataset"]
    ratio = float(payload["ratio"])
    description = str(payload["description"])
    layout = verify_dataset_layout(raw_root, processed_root, dataset, ratio, description)
    merged_dir = Path(payload["paths"]["merged_dir"]).resolve()
    image_ids = [row["image_id"] for row in payload["records"]]

    create_voronoi_label(str(merged_dir), str(vor_dir), image_ids)
    if args.cluster_mode == "original":
        create_cluster_label(str(layout["image_dir"]), str(merged_dir), str(vor_dir), str(clu_dir), image_ids)
    else:
        for stem in image_ids:
            label_vor = imread(vor_dir / f"{stem}_label_vor.png")
            save_png(clu_dir / f"{stem}_label_clu.png", label_vor)

    rows = []
    for row in payload["records"]:
        stem = row["image_id"]
        image_rgb = read_rgb(image_path(layout["dataset_dir"], stem))
        merged_points = imread(merged_dir / f"{stem}_label_point.png").astype(np.uint8)
        label_vor = read_rgb(vor_dir / f"{stem}_label_vor.png")
        label_clu = read_rgb(clu_dir / f"{stem}_label_clu.png")
        gt_inst = None if args.no_gt_diagnostics else np.load(gt_instance_path(layout["dataset_dir"], stem))
        vor_inst = relabel_contiguous(vor_label_to_instance_map(label_vor, merged_points))
        clu_inst = relabel_contiguous(clu_label_to_instance_map(label_clu, merged_points))
        save_instance_png(vor_inst_dir / f"{stem}.png", vor_inst)
        save_instance_png(clu_inst_dir / f"{stem}.png", clu_inst)
        panels = [
                ("image", image_rgb),
                ("points", np.stack([merged_points] * 3, axis=2)),
                ("labels_vor", label_vor),
                ("labels_clu", label_clu),
                ("vor_instance", colorize(vor_inst)),
                ("clu_instance", colorize(clu_inst)),
            ]
        if gt_inst is not None:
            panels.append(("gt_instance", colorize(gt_inst)))
        tile_images(panels, overlay_dir / f"{stem}_baseline_panel.png")
        rows.append(
            {
                "image_id": stem,
                "split": row["split"],
                "merged_point_count": int(np.sum(merged_points > 0)),
                "vor_instance_count": int(vor_inst.max()),
                "clu_instance_count": int(clu_inst.max()),
                "gt_instance_count": int(np.max(gt_inst)) if gt_inst is not None else None,
            }
        )

    manifest = {
        "dataset": dataset,
        "ratio": ratio,
        "description": description,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "points_manifest": str(points_manifest),
        "records": rows,
        "paths": {
            "merged_dir": str(merged_dir),
            "labels_vor_dir": str(vor_dir),
            "labels_clu_dir": str(clu_dir),
            "bonus_vor_instance_dir": str(vor_inst_dir),
            "bonus_clu_instance_dir": str(clu_inst_dir),
            "overlay_dir": str(overlay_dir),
        },
        "cluster_mode": args.cluster_mode,
        "gt_diagnostics_enabled": not args.no_gt_diagnostics,
    }
    save_json(step_root / "bonus_region_baseline_manifest.json", manifest)
    save_csv(step_root / "bonus_region_baseline_summary.csv", rows)
    write_step_report(
        step_root / "bonus_region_baseline_report.md",
        "BoNuS Baseline Region Labels",
        [
            f"generated isolated labels_vor and labels_clu under {step_root}",
            f"cluster_mode={args.cluster_mode}",
            f"image_ids={', '.join(image_ids)}",
            "saved instance-like conversions for both Voronoi and cluster labels",
        ],
    )
    print(f"Saved BoNuS baseline region outputs to: {step_root}")


def colorize(inst_map: np.ndarray) -> np.ndarray:
    from common import colorize_instances

    return colorize_instances(inst_map)


if __name__ == "__main__":
    main()
