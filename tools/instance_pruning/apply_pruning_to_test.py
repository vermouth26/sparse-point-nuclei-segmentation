from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.instance_pruning.common import (
    ensure_dir,
    evaluate_pruned_dir,
    load_baseline_overall,
    load_json,
    read_csv_dicts,
    write_csv_dicts,
    write_json,
)
from tools.instance_pruning.prune_instances import apply_pruning_to_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply val-selected instance pruning config to official test32 once.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--np-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--baseline-overall", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    selected = load_json(args.selected_config)
    config = {k: selected[k] for k in ["mode", "min_area", "np_mean_threshold", "np_p10_threshold", "count_upper_factor"]}
    pruned_dir = ensure_dir(out_dir / "inst_pred_pruned")
    pruning_summary = apply_pruning_to_dir(
        pred_dir=args.pred_dir,
        features_csv=args.features_csv,
        out_dir=pruned_dir,
        config=config,
    )
    image_ids = sorted(p.stem for p in args.pred_dir.glob("*.npy"))
    rows, overall = evaluate_pruned_dir(
        pred_dir=pruned_dir,
        label_dir=args.label_dir,
        image_ids=image_ids,
        matching_radius=args.matching_radius,
    )
    write_csv_dicts(out_dir / "per_image_metrics_pruned.csv", rows)
    write_json(out_dir / "overall_metrics_pruned.json", overall)
    write_json(out_dir / "pruning_summary.json", pruning_summary)
    write_summary_csv(out_dir / "summary_test_pruning.csv", args.baseline_overall, overall)
    write_final_report(
        path=out_dir / "FINAL_PRUNING_REPORT.md",
        baseline=load_baseline_overall(args.baseline_overall),
        pruned=overall,
        selected=selected,
        pruning_summary=pruning_summary,
    )
    make_visualizations(
        out_dir=ensure_dir(out_dir / "vis_pruning"),
        image_dir=args.image_dir,
        pred_dir=args.pred_dir,
        pruned_dir=pruned_dir,
        np_dir=args.np_dir,
        features_csv=args.features_csv,
        baseline_per_image=args.baseline_per_image,
        pruned_rows=rows,
        pruning_summary=pruning_summary,
    )
    print(json.dumps({"baseline": load_json(args.baseline_overall), "pruned": overall}, indent=2))
    print(f"Saved fixed-config test pruning outputs to: {out_dir.resolve()}")


def write_summary_csv(path: Path, baseline_path: Path, pruned: dict[str, Any]) -> None:
    baseline = load_baseline_overall(baseline_path)
    metrics = [
        "mean_dice",
        "mean_aji",
        "mean_dq",
        "mean_sq",
        "mean_pq",
        "mean_detection_precision",
        "mean_detection_recall",
        "mean_detection_f1",
        "global_pred_gt_instance_ratio",
    ]
    rows = []
    for metric in metrics:
        old = float(baseline.get(metric, 0.0))
        new = float(pruned.get(metric, 0.0))
        rows.append(
            {
                "metric": metric,
                "baseline": old,
                "pruned": new,
                "delta": new - old,
                "relative_delta": (new - old) / old if old else 0.0,
            }
        )
    write_csv_dicts(path, rows)


def write_final_report(
    *,
    path: Path,
    baseline: dict[str, float],
    pruned: dict[str, Any],
    selected: dict[str, Any],
    pruning_summary: dict[str, Any],
) -> None:
    delta = {k: float(pruned.get(k, 0.0)) - baseline.get(k, 0.0) for k in pruned if isinstance(pruned.get(k), (int, float))}
    removed_total = int(pruning_summary.get("total_removed_count", 0))
    before_total = int(pruning_summary.get("total_before_count", 0))
    keep_verdict = (
        delta.get("mean_pq", 0.0) >= 0.005
        or delta.get("mean_aji", 0.0) >= 0.005
        or (
            delta.get("mean_detection_precision", 0.0) > 0
            and abs(float(pruned.get("global_pred_gt_instance_ratio", 0.0)) - 1.0)
            < abs(baseline.get("global_pred_gt_instance_ratio", 0.0) - 1.0)
            and delta.get("mean_pq", 0.0) >= 0
        )
    )
    lines = [
        "# CPM17 r015 Fixed Test Instance Pruning Report",
        "",
        "## No-Leakage Statement",
        "",
        "- Pruning parameters were selected on official val4 only.",
        "- Official test32 GT is used here only for the single final evaluation with fixed parameters.",
        "- Stage1, PathoSAM-D, HoVer targets, HoVer network, loss, and postprocess are unchanged.",
        "",
        "## Fixed Config",
        "",
        f"- mode: `{selected['mode']}`",
        f"- min_area: `{selected['min_area']}`",
        f"- np_mean_threshold: `{selected['np_mean_threshold']}`",
        f"- np_p10_threshold: `{selected['np_p10_threshold']}`",
        f"- count_upper_factor: `{selected['count_upper_factor']}`",
        "",
        "## Test Metrics",
        "",
        metric_line("Dice", "mean_dice", baseline, pruned, delta),
        metric_line("AJI", "mean_aji", baseline, pruned, delta),
        metric_line("PQ", "mean_pq", baseline, pruned, delta),
        metric_line("DQ", "mean_dq", baseline, pruned, delta),
        metric_line("SQ", "mean_sq", baseline, pruned, delta),
        metric_line("Precision", "mean_detection_precision", baseline, pruned, delta),
        metric_line("Recall", "mean_detection_recall", baseline, pruned, delta),
        metric_line("Detection F1", "mean_detection_f1", baseline, pruned, delta),
        metric_line("Global pred/GT", "global_pred_gt_instance_ratio", baseline, pruned, delta),
        "",
        "## Instance Removal",
        "",
        f"- removed instances: `{removed_total}` / `{before_total}`",
        f"- remaining instances: `{pruning_summary.get('total_after_count', 0)}`",
        "",
        "## Interpretation",
        "",
        f"- precision improved: `{delta.get('mean_detection_precision', 0.0) > 0}`",
        f"- pred/GT closer to 1: `{abs(float(pruned.get('global_pred_gt_instance_ratio', 0.0)) - 1.0) < abs(baseline.get('global_pred_gt_instance_ratio', 0.0) - 1.0)}`",
        f"- recall delta: `{delta.get('mean_detection_recall', 0.0):+.6f}`",
        f"- PQ delta decomposition: DQ `{delta.get('mean_dq', 0.0):+.6f}`, SQ `{delta.get('mean_sq', 0.0):+.6f}`",
        f"- worth keeping as inference-time calibration: `{keep_verdict}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_line(label: str, key: str, baseline: dict[str, float], pruned: dict[str, Any], delta: dict[str, float]) -> str:
    return f"- {label}: `{baseline.get(key, 0.0):.6f}` -> `{float(pruned.get(key, 0.0)):.6f}` (delta `{delta.get(key, 0.0):+.6f}`)"


def make_visualizations(
    *,
    out_dir: Path,
    image_dir: Path,
    pred_dir: Path,
    pruned_dir: Path,
    np_dir: Path,
    features_csv: Path,
    baseline_per_image: Path,
    pruned_rows: list[dict[str, Any]],
    pruning_summary: dict[str, Any],
) -> None:
    baseline_rows = {row["image_id"]: row for row in read_csv_dicts(baseline_per_image)}
    pruned_by_id = {str(row["image_id"]): row for row in pruned_rows}
    removed_by_id = {row["image_id"]: row for row in pruning_summary.get("per_image", [])}
    ranked = []
    for image_id, row in pruned_by_id.items():
        old_pq = float(baseline_rows.get(image_id, {}).get("pq", 0.0))
        new_pq = float(row.get("pq", 0.0))
        removed = int(removed_by_id.get(image_id, {}).get("removed_count", 0))
        ranked.append((image_id, new_pq - old_pq, removed))
    selected_ids = []
    selected_ids.extend([x[0] for x in sorted(ranked, key=lambda x: x[1], reverse=True)[:5]])
    selected_ids.extend([x[0] for x in sorted(ranked, key=lambda x: x[1])[:5]])
    selected_ids.extend([x[0] for x in sorted(ranked, key=lambda x: x[2], reverse=True)[:5]])
    selected_ids = list(dict.fromkeys(selected_ids))
    features_by_image: dict[str, list[dict[str, str]]] = {}
    for row in read_csv_dicts(features_csv):
        features_by_image.setdefault(row["image_id"], []).append(row)

    for image_id in selected_ids:
        image = np.asarray(Image.open(image_dir / f"{image_id}.png").convert("RGB"))
        pred = np.load(pred_dir / f"{image_id}.npy", allow_pickle=False)
        pruned = np.load(pruned_dir / f"{image_id}.npy", allow_pickle=False)
        np_prob = np.load(np_dir / f"{image_id}.npy", allow_pickle=False)
        if np_prob.ndim == 3 and np_prob.shape[0] == 1:
            np_prob = np_prob[0]
        removed_mask = np.logical_and(pred > 0, pruned == 0)
        panel = make_panel(image, pred, pruned, removed_mask, np_prob, image_id)
        panel.save(out_dir / f"{image_id}_pruning_panel.png")
        removed_ids = set(str(v) for v in removed_by_id.get(image_id, {}).get("removed_ids", "").split(",") if v)
        removed_rows = [row for row in features_by_image.get(image_id, []) if row.get("instance_id") in removed_ids]
        if removed_rows:
            write_csv_dicts(out_dir / f"{image_id}_removed_instance_features.csv", removed_rows)


def make_panel(image: np.ndarray, pred: np.ndarray, pruned: np.ndarray, removed_mask: np.ndarray, np_prob: np.ndarray, image_id: str) -> Image.Image:
    h, w = image.shape[:2]
    panels = [
        Image.fromarray(image),
        Image.fromarray(colorize_instances(pred)),
        Image.fromarray(colorize_instances(pruned)),
        Image.fromarray(overlay_removed(image, removed_mask)),
        Image.fromarray(heatmap_np(np_prob)),
    ]
    labels = ["image", "original instances", "pruned instances", "removed overlay", "NP heatmap"]
    canvas = Image.new("RGB", (w * len(panels), h + 28), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel.resize((w, h)), (i * w, 28))
        draw.text((i * w + 4, 6), f"{image_id} | {label}", fill=(0, 0, 0))
    return canvas


def colorize_instances(inst: np.ndarray) -> np.ndarray:
    out = np.zeros((*inst.shape, 3), dtype=np.uint8)
    for inst_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        rng = np.random.default_rng(inst_id)
        out[inst == inst_id] = rng.integers(40, 235, size=3, dtype=np.uint8)
    return out


def overlay_removed(image: np.ndarray, removed_mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[removed_mask] = (0.35 * out[removed_mask] + 0.65 * red[removed_mask]).astype(np.uint8)
    return out


def heatmap_np(np_prob: np.ndarray) -> np.ndarray:
    x = np.clip(np_prob, 0.0, 1.0)
    out = np.zeros((*x.shape, 3), dtype=np.uint8)
    out[..., 0] = (255 * x).astype(np.uint8)
    out[..., 1] = (255 * (1.0 - np.abs(x - 0.5) * 2.0)).astype(np.uint8)
    out[..., 2] = (255 * (1.0 - x)).astype(np.uint8)
    return out


if __name__ == "__main__":
    main()
