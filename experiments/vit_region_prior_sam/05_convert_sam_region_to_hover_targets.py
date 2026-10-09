from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    compute_instance_metrics,
    ensure_dir,
    gt_instance_path,
    gt_point_path,
    image_path,
    imread,
    point_map_to_coords,
    read_rgb,
    save_csv,
    save_json,
    save_instance_png,
    save_png,
    save_text,
    tile_images,
    verify_dataset_layout,
)

from experiments.hover_stage23_alpha050_bn_region_prior.stage2_bridge.common import (
    colorize_instances,
    draw_centers,
    make_hv_targets,
    make_np_targets,
    render_gray,
    render_hv_magnitude,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pseudo-instance regions into HoVerNet supervision targets.")
    parser.add_argument("--points-manifest", required=True)
    parser.add_argument("--sam-manifest", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--hv-erosion-radius", type=int, default=2)
    parser.add_argument("--use-sam-ignore", action="store_true", help="Keep SAM ignore mask as additional uncertain area. Recommended.")
    parser.add_argument(
        "--no-gt-diagnostics",
        action="store_true",
        help="Do not read full instance/point GT or compute target-quality metrics. Required for strict train target construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points_manifest = load_manifest(Path(args.points_manifest))
    sam_manifest = load_manifest(Path(args.sam_manifest))
    out_root = Path(args.out_dir).resolve() if args.out_dir else Path(args.points_manifest).resolve().parents[1]
    step_root = ensure_dir(out_root / "05_hover_targets")

    raw_root = Path(points_manifest["raw_root"]).resolve()
    processed_root = Path(points_manifest["processed_root"]).resolve()
    dataset = points_manifest["dataset"]
    ratio = float(points_manifest["ratio"])
    description = str(points_manifest["description"])
    layout = verify_dataset_layout(raw_root, processed_root, dataset, ratio, description)

    merged_dir = Path(points_manifest["paths"]["merged_dir"]).resolve()
    sam_inst_dir = Path(sam_manifest["paths"]["sam_instance_dir"]).resolve()
    sam_ignore_dir = Path(sam_manifest["paths"]["sam_ignore_dir"]).resolve()
    sam_boundary_dir = Path(sam_manifest["paths"]["sam_boundary_dir"]).resolve()

    inst_dir = ensure_dir(step_root / "inst_map")
    np_dir = ensure_dir(step_root / "np_map")
    hv_dir = ensure_dir(step_root / "hv_map")
    np_ignore_dir = ensure_dir(step_root / "np_ignore_mask")
    hv_ignore_dir = ensure_dir(step_root / "hv_ignore_mask")
    boundary_dir = ensure_dir(step_root / "boundary_map")
    overlay_dir = ensure_dir(step_root / "overlays")

    rows = []
    for row in points_manifest["records"]:
        stem = row["image_id"]
        image_rgb = read_rgb(image_path(layout["dataset_dir"], stem))
        gt_inst = None if args.no_gt_diagnostics else np.load(gt_instance_path(layout["dataset_dir"], stem))
        gt_point = None if args.no_gt_diagnostics else imread(gt_point_path(layout["dataset_dir"], stem)).astype(np.uint8)
        merged_points = imread(merged_dir / f"{stem}_label_point.png").astype(np.uint8)
        inst_map = imread(sam_inst_dir / f"{stem}.png").astype(np.int32)
        sam_ignore = imread(sam_ignore_dir / f"{stem}.png").astype(np.uint8) > 0
        sam_boundary = imread(sam_boundary_dir / f"{stem}.png").astype(np.uint8) > 0

        region_fg = inst_map > 0
        bg_mask = ~region_fg
        uncertain = sam_boundary.copy()
        if args.use_sam_ignore:
            uncertain |= sam_ignore

        np_map, np_ignore = make_np_targets(region_fg, bg_mask, uncertain, sam_boundary)
        hv_map, hv_ignore = make_hv_targets(inst_map, region_fg, uncertain, sam_boundary, erosion_radius=args.hv_erosion_radius)
        boundary_map = sam_boundary.astype(np.uint8)

        np.save(inst_dir / f"{stem}.npy", inst_map.astype(np.int32))
        np.save(np_dir / f"{stem}.npy", np_map.astype(np.uint8))
        np.save(hv_dir / f"{stem}.npy", hv_map.astype(np.float32))
        np.save(np_ignore_dir / f"{stem}.npy", np_ignore.astype(np.uint8))
        np.save(hv_ignore_dir / f"{stem}.npy", hv_ignore.astype(np.uint8))
        np.save(boundary_dir / f"{stem}.npy", boundary_map.astype(np.uint8))
        save_instance_png(inst_dir / f"{stem}.png", inst_map.astype(np.uint16))
        save_png(np_dir / f"{stem}.png", np_map.astype(np.uint8) * 255)
        save_png(np_ignore_dir / f"{stem}.png", np_ignore.astype(np.uint8) * 255)
        save_png(hv_ignore_dir / f"{stem}.png", hv_ignore.astype(np.uint8) * 255)
        save_png(boundary_dir / f"{stem}.png", boundary_map.astype(np.uint8) * 255)

        panels = [
                ("image", image_rgb),
                ("merged_points", draw_center_overlay(image_rgb, merged_points)),
                ("sam_instance", colorize_instances(inst_map)),
                ("np_map", render_gray(np_map)),
                ("hv_mag", render_hv_magnitude(hv_map)),
                ("np_ignore", render_gray(np_ignore)),
                ("hv_ignore", render_gray(hv_ignore)),
            ]
        if gt_inst is not None:
            panels.append(("gt_instance", colorize_instances(gt_inst)))
        tile_images(panels, overlay_dir / f"{stem}_hover_target_panel.png")

        inst_metrics = compute_instance_metrics(inst_map, gt_inst, gt_point) if gt_inst is not None else {}
        hv_valid = hv_map[hv_ignore == 0]
        rows.append(
            {
                "image_id": stem,
                "split": row["split"],
                "pseudo_instance_count": int(inst_map.max()),
                "np_positive_ratio": float(np.mean(np_map > 0)),
                "np_ignore_ratio": float(np.mean(np_ignore > 0)),
                "hv_ignore_ratio": float(np.mean(hv_ignore > 0)),
                "hv_min": float(hv_valid.min()) if hv_valid.size else 0.0,
                "hv_max": float(hv_valid.max()) if hv_valid.size else 0.0,
                "hv_has_nan": bool(np.isnan(hv_map).any()),
                "hv_has_inf": bool(np.isinf(hv_map).any()),
                **inst_metrics,
            }
        )

    manifest = {
        "dataset": dataset,
        "ratio": ratio,
        "description": description,
        "points_manifest": str(Path(args.points_manifest).resolve()),
        "sam_manifest": str(Path(args.sam_manifest).resolve()),
        "records": rows,
        "paths": {
            "inst_map": str(inst_dir.resolve()),
            "np_map": str(np_dir.resolve()),
            "hv_map": str(hv_dir.resolve()),
            "np_ignore_mask": str(np_ignore_dir.resolve()),
            "hv_ignore_mask": str(hv_ignore_dir.resolve()),
            "boundary_map": str(boundary_dir.resolve()),
            "overlays": str(overlay_dir.resolve()),
        },
        "hv_layout": "H x W x 2, channel 0 = normalized dx, channel 1 = normalized dy.",
        "target_mode": "hard",
        "gt_diagnostics_enabled": not args.no_gt_diagnostics,
    }
    save_csv(step_root / "hover_target_summary.csv", rows)
    save_json(step_root / "hover_target_manifest.json", manifest)
    save_text(
        step_root / "hover_target_report.md",
        "\n".join(
            [
                "# HoVer Target Conversion",
                "",
                f"- source SAM manifest: `{Path(args.sam_manifest).resolve()}`",
                f"- hv erosion radius: {args.hv_erosion_radius}",
                f"- use_sam_ignore: {args.use_sam_ignore}",
                f"- images: {', '.join(row['image_id'] for row in rows)}",
                "",
                "These supervision targets are derived from pseudo instances rather than ground-truth instances.",
                "",
            ]
        ),
    )
    print(f"Saved HoVer target conversion outputs to: {step_root}")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def draw_center_overlay(image_rgb: np.ndarray, point_map: np.ndarray) -> np.ndarray:
    centers = [{"x": x, "y": y} for y, x in point_map_to_coords(point_map)]
    return draw_centers(image_rgb, centers)


if __name__ == "__main__":
    main()
