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
    mean_rows,
    read_rgb,
    save_csv,
    save_json,
    save_text,
    tile_images,
    verify_dataset_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pseudo region quality for BoNuS baseline labels and SAM point-prompt regions.")
    parser.add_argument("--points-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--sam-manifest", required=True)
    parser.add_argument("--sam-label", default="sam_region")
    parser.add_argument(
        "--extra-sam-manifest",
        action="append",
        default=[],
        help="Optional extra SAM manifest in the form label=path, for side-by-side comparison.",
    )
    parser.add_argument("--coarse-pred-dir", default="", help="Optional Stage2 coarse prediction directory containing *.npy instance maps.")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points_manifest = load_manifest(Path(args.points_manifest))
    baseline_manifest = load_manifest(Path(args.baseline_manifest))
    sam_manifest = load_manifest(Path(args.sam_manifest))
    out_root = Path(args.out_dir).resolve() if args.out_dir else Path(args.points_manifest).resolve().parents[1]
    step_root = ensure_dir(out_root / "04_eval")
    overlay_dir = ensure_dir(step_root / "overlays")

    raw_root = Path(points_manifest["raw_root"]).resolve()
    processed_root = Path(points_manifest["processed_root"]).resolve()
    dataset = points_manifest["dataset"]
    ratio = float(points_manifest["ratio"])
    description = str(points_manifest["description"])
    layout = verify_dataset_layout(raw_root, processed_root, dataset, ratio, description)

    method_roots = {
        "bonus_vor": Path(baseline_manifest["paths"]["bonus_vor_instance_dir"]).resolve(),
        "bonus_clu": Path(baseline_manifest["paths"]["bonus_clu_instance_dir"]).resolve(),
        args.sam_label: Path(sam_manifest["paths"]["sam_instance_dir"]).resolve(),
    }
    for entry in args.extra_sam_manifest:
        label, manifest_path = parse_extra_manifest(entry)
        extra_manifest = load_manifest(Path(manifest_path))
        method_roots[label] = Path(extra_manifest["paths"]["sam_instance_dir"]).resolve()
    if args.coarse_pred_dir:
        method_roots["bonus_coarse"] = Path(args.coarse_pred_dir).resolve()

    per_image_rows = []
    first_pass = {}
    for row in points_manifest["records"]:
        stem = row["image_id"]
        gt_inst = np.load(gt_instance_path(layout["dataset_dir"], stem))
        gt_point_map = imread(gt_point_path(layout["dataset_dir"], stem)).astype(np.uint8)
        image_rgb = read_rgb(image_path(layout["dataset_dir"], stem))
        first_pass[stem] = {"image": image_rgb, "gt": gt_inst}
        for method, root in method_roots.items():
            pred_path = resolve_pred_path(root, stem)
            if pred_path is None:
                continue
            pred_inst = imread(pred_path).astype(np.int32)
            metrics = compute_instance_metrics(pred_inst, gt_inst, gt_point_map)
            per_image_rows.append(
                {
                    "method": method,
                    "image_id": stem,
                    "split": row["split"],
                    **metrics,
                }
            )
            first_pass[stem][method] = pred_inst

    if not per_image_rows:
        raise SystemExit("No prediction files were found for evaluation.")
    summary_rows = mean_rows(per_image_rows, "method")
    save_csv(step_root / "pseudo_region_quality_per_image.csv", per_image_rows)
    save_csv(step_root / "pseudo_region_quality_summary.csv", summary_rows)

    for stem, payload in first_pass.items():
        panels = [("image", payload["image"]), ("gt_instance", colorize(payload["gt"]))]
        for method in method_roots.keys():
            if method in payload:
                panels.append((method, colorize(payload[method])))
        if "bonus_coarse" in payload:
            panels.append(("bonus_coarse", colorize(payload["bonus_coarse"])))
        tile_images(panels, overlay_dir / f"{stem}_quality_panel.png")

    report = build_report(summary_rows, args.sam_label)
    save_text(step_root / "pseudo_region_quality_report.md", report)
    save_json(
        step_root / "pseudo_region_quality_report.json",
        {
            "summary": summary_rows,
            "report_markdown": report,
        },
    )
    print(f"Saved pseudo-region quality evaluation to: {step_root}")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_pred_path(root: Path, stem: str) -> Path | None:
    candidates = [
        root / f"{stem}.png",
        root / f"{stem}.npy",
        root / f"{stem}_label.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def colorize(inst_map: np.ndarray) -> np.ndarray:
    from common import colorize_instances

    return colorize_instances(inst_map)


def build_report(summary_rows: list[dict], primary_sam_label: str) -> str:
    summary = {row["method"]: row for row in summary_rows}

    def metric(method: str, key: str) -> float:
        row = summary.get(method, {})
        value = row.get(key, float("nan"))
        return float(value) if value == value else float("nan")

    def compare_text(target: str, base: str) -> str:
        if target not in summary or base not in summary:
            return f"{target} vs {base}: unavailable"
        deltas = {
            "binary_dice": metric(target, "binary_dice") - metric(base, "binary_dice"),
            "aji": metric(target, "aji") - metric(base, "aji"),
            "pq": metric(target, "pq") - metric(base, "pq"),
        }
        better = sum(delta > 0.01 for delta in deltas.values())
        worse = sum(delta < -0.01 for delta in deltas.values())
        verdict = "better" if better >= 2 and worse == 0 else "not better" if worse >= 2 else "mixed"
        return (
            f"{target} vs {base}: {verdict} "
            f"(Dice {deltas['binary_dice']:+.4f}, AJI {deltas['aji']:+.4f}, PQ {deltas['pq']:+.4f})"
        )

    lines = [
        "# Pseudo Region Quality Report",
        "",
        "## Aggregate Means",
        "",
        "| method | binary_dice | aji | dq | sq | pq | detection_f1 | pred_instance_count | pred_gt_count_ratio | split_error_count | merge_error_count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row.get('binary_dice', float('nan')):.4f} | {row.get('aji', float('nan')):.4f} | "
            f"{row.get('dq', float('nan')):.4f} | {row.get('sq', float('nan')):.4f} | {row.get('pq', float('nan')):.4f} | "
            f"{row.get('detection_f1', float('nan')):.4f} | {row.get('pred_instance_count', float('nan')):.2f} | "
            f"{row.get('pred_gt_count_ratio', float('nan')):.4f} | {row.get('split_error_count', float('nan')):.2f} | "
            f"{row.get('merge_error_count', float('nan')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Required Conclusions",
            "",
            f"1. SAM region vs labels_vor: {compare_text(primary_sam_label, 'bonus_vor')}",
            f"2. SAM region vs labels_clu: {compare_text(primary_sam_label, 'bonus_clu')}",
        ]
    )
    if "bonus_coarse" in summary:
        lines.append(f"3. SAM region vs BoNuS coarse prediction: {compare_text(primary_sam_label, 'bonus_coarse')}")
    else:
        lines.append("3. SAM region vs BoNuS coarse prediction: unavailable because --coarse-pred-dir was not provided.")

    proceed = should_proceed(summary, primary_sam_label)
    lines.extend(
        [
            f"4. Worth proceeding to Stage2/Stage3 training: {proceed}",
            "",
            "Only move to Stage2/Stage3 if the untrained SAM region prior is clearly stronger than labels_vor / labels_clu, or at least close to an available coarse prediction baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def should_proceed(summary: dict[str, dict], primary_sam_label: str) -> str:
    sam = summary.get(primary_sam_label)
    vor = summary.get("bonus_vor")
    clu = summary.get("bonus_clu")
    if not sam or not vor or not clu:
        return "insufficient evidence"
    better_than_vor = (
        sam.get("pq", 0.0) > vor.get("pq", 0.0) + 0.01
        and sam.get("aji", 0.0) > vor.get("aji", 0.0) + 0.01
    )
    better_than_clu = (
        sam.get("pq", 0.0) > clu.get("pq", 0.0) + 0.01
        and sam.get("aji", 0.0) > clu.get("aji", 0.0) + 0.01
    )
    if better_than_vor and better_than_clu:
        return "yes, the untrained SAM prior looks strong enough to justify the next training stage"
    if better_than_vor or better_than_clu:
        return "maybe, but the gain is partial and should be checked on a slightly larger validation subset first"
    return "no, stay at the pseudo-label quality stage before spending training budget"


def parse_extra_manifest(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise SystemExit(f"Invalid --extra-sam-manifest entry: {entry}. Expected label=path.")
    label, path = entry.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit(f"Invalid --extra-sam-manifest entry: {entry}. Expected label=path.")
    return label, path


if __name__ == "__main__":
    main()
