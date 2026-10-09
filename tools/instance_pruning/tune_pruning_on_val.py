from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.instance_pruning.common import (
    ensure_dir,
    evaluate_pruned_dir,
    load_baseline_overall,
    read_csv_dicts,
    write_csv_dicts,
    write_json,
)
from tools.instance_pruning.prune_instances import apply_pruning_to_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune instance pruning rules on CPM17 val4 only.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--baseline-overall", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modes", default="area_only,np_only,np_area,np_area_center,np_area_center_count")
    parser.add_argument("--min-areas", default="10,20,30,50")
    parser.add_argument("--np-mean-thresholds", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--np-p10-thresholds", default="0.10,0.20,0.30")
    parser.add_argument("--count-upper-factors", default="1.05,1.10,1.15,1.20")
    parser.add_argument("--max-dice-drop", type=float, default=0.005)
    parser.add_argument("--ratio-penalty-weight", type=float, default=0.01)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    parser.add_argument("--save-selected-maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    features = read_csv_dicts(args.features_csv)
    image_ids = sorted({row["image_id"] for row in features})
    if not image_ids:
        raise SystemExit(f"No features found in {args.features_csv}")
    center_available = any(float(row.get("expected_center_count_image", 0) or 0) > 0 for row in features)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not center_available:
        modes = [m for m in modes if not m.startswith("np_area_center")]

    baseline = load_baseline_overall(args.baseline_overall)
    configs = build_configs(
        modes=modes,
        min_areas=parse_int_list(args.min_areas),
        np_mean_thresholds=parse_float_list(args.np_mean_thresholds),
        np_p10_thresholds=parse_float_list(args.np_p10_thresholds),
        count_upper_factors=parse_float_list(args.count_upper_factors),
    )

    rows: list[dict[str, Any]] = []
    scratch = ensure_dir(out_dir / "_scratch_pruned_maps")
    for idx, config in enumerate(configs):
        cfg_dir = ensure_dir(scratch / f"cfg_{idx:04d}")
        pruning_summary = apply_pruning_to_dir(pred_dir=args.pred_dir, features_csv=args.features_csv, out_dir=cfg_dir, config=config)
        _, overall = evaluate_pruned_dir(
            pred_dir=cfg_dir,
            label_dir=args.label_dir,
            image_ids=image_ids,
            matching_radius=args.matching_radius,
        )
        row = {
            "config_index": idx,
            **config,
            **overall,
            "delta_dice": overall["mean_dice"] - baseline.get("mean_dice", 0.0),
            "delta_aji": overall["mean_aji"] - baseline.get("mean_aji", 0.0),
            "delta_pq": overall["mean_pq"] - baseline.get("mean_pq", 0.0),
            "delta_precision": overall["mean_detection_precision"] - baseline.get("mean_detection_precision", 0.0),
            "delta_recall": overall["mean_detection_recall"] - baseline.get("mean_detection_recall", 0.0),
            "ratio_abs_error": abs(overall["global_pred_gt_instance_ratio"] - 1.0),
            "dice_drop_ok": overall["mean_dice"] >= baseline.get("mean_dice", 0.0) - args.max_dice_drop,
            "removed_instance_count": int(pruning_summary.get("total_removed_count", 0)),
            "before_instance_count": int(pruning_summary.get("total_before_count", 0)),
            "after_instance_count": int(pruning_summary.get("total_after_count", 0)),
            "removed_instance_ratio": safe_div(
                int(pruning_summary.get("total_removed_count", 0)),
                int(pruning_summary.get("total_before_count", 0)),
            ),
        }
        row["selection_score"] = selection_score(row, args.ratio_penalty_weight)
        rows.append(row)
        print(
            f"[val_pruning] cfg={idx:04d} mode={config['mode']} pq={overall['mean_pq']:.6f} "
            f"aji={overall['mean_aji']:.6f} ratio={overall['global_pred_gt_instance_ratio']:.4f}"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_dicts(out_dir / "val_pruning_sweep.csv", rows, fieldnames=fieldnames)
    selected = select_best(rows)
    write_json(out_dir / "selected_pruning_config.json", selected)
    selected_dir = ensure_dir(out_dir / "selected_val_pruned_inst")
    selected_config = {k: selected[k] for k in ["mode", "min_area", "np_mean_threshold", "np_p10_threshold", "count_upper_factor"]}
    apply_pruning_to_dir(pred_dir=args.pred_dir, features_csv=args.features_csv, out_dir=selected_dir, config=selected_config)
    val_rows, val_overall = evaluate_pruned_dir(
        pred_dir=selected_dir,
        label_dir=args.label_dir,
        image_ids=image_ids,
        matching_radius=args.matching_radius,
    )
    write_csv_dicts(out_dir / "selected_val_per_image_metrics.csv", val_rows)
    write_json(out_dir / "selected_val_overall_metrics.json", val_overall)
    write_report(out_dir / "VAL_SELECTION_REPORT.md", baseline, selected, center_available, len(configs))
    print(f"Selected pruning config saved to: {(out_dir / 'selected_pruning_config.json').resolve()}")


def build_configs(
    *,
    modes: list[str],
    min_areas: list[int],
    np_mean_thresholds: list[float],
    np_p10_thresholds: list[float],
    count_upper_factors: list[float],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "area_only":
            for min_area in min_areas:
                configs.append(default_config(mode, min_area, 0.0, 0.0, 1.1))
        elif mode == "np_only":
            for np_mean, np_p10 in itertools.product(np_mean_thresholds, np_p10_thresholds):
                configs.append(default_config(mode, 0, np_mean, np_p10, 1.1))
        elif mode in {"np_area", "np_area_center"}:
            for min_area, np_mean, np_p10 in itertools.product(min_areas, np_mean_thresholds, np_p10_thresholds):
                configs.append(default_config(mode, min_area, np_mean, np_p10, 1.1))
        elif mode == "np_area_center_count":
            for min_area, np_mean, np_p10, factor in itertools.product(
                min_areas, np_mean_thresholds, np_p10_thresholds, count_upper_factors
            ):
                configs.append(default_config(mode, min_area, np_mean, np_p10, factor))
    return configs


def default_config(mode: str, min_area: int, np_mean: float, np_p10: float, factor: float) -> dict[str, Any]:
    return {
        "mode": mode,
        "min_area": min_area,
        "np_mean_threshold": np_mean,
        "np_p10_threshold": np_p10,
        "count_upper_factor": factor,
    }


def selection_score(row: dict[str, Any], ratio_penalty_weight: float) -> float:
    score = float(row["mean_pq"])
    score += 0.10 * float(row["mean_aji"])
    score += 0.02 * float(row["mean_detection_precision"])
    score -= ratio_penalty_weight * float(row["ratio_abs_error"])
    if not row["dice_drop_ok"]:
        score -= 1.0
    return score


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["dice_drop_ok"]]
    pool = valid if valid else rows
    return max(
        pool,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["mean_pq"]),
            float(row["mean_aji"]),
            -float(row["ratio_abs_error"]),
            float(row["mean_detection_precision"]),
        ),
    )


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def write_report(path: Path, baseline: dict[str, float], selected: dict[str, Any], center_available: bool, config_count: int) -> None:
    lines = [
        "# CPM17 r015 Val-Only Instance Pruning Selection",
        "",
        "## Leakage Policy",
        "",
        "- This sweep uses CPM17 official val4 GT only for pruning parameter selection.",
        "- It does not read official test32 GT.",
        "- It does not retrain Stage1, PathoSAM, HoVer targets, or HoVer network.",
        "- Test32 must be evaluated once with the fixed selected config.",
        "",
        "## Search Space",
        "",
        f"- configs evaluated: `{config_count}`",
        f"- center features available: `{center_available}`",
        "",
        "## Baseline Val Metrics",
        "",
        f"- Dice: `{baseline.get('mean_dice', 0.0):.6f}`",
        f"- AJI: `{baseline.get('mean_aji', 0.0):.6f}`",
        f"- PQ: `{baseline.get('mean_pq', 0.0):.6f}`",
        f"- precision: `{baseline.get('mean_detection_precision', 0.0):.6f}`",
        f"- recall: `{baseline.get('mean_detection_recall', 0.0):.6f}`",
        f"- global pred/GT: `{baseline.get('global_pred_gt_instance_ratio', 0.0):.6f}`",
        "",
        "## Selected Config",
        "",
        f"- mode: `{selected['mode']}`",
        f"- min_area: `{selected['min_area']}`",
        f"- np_mean_threshold: `{selected['np_mean_threshold']}`",
        f"- np_p10_threshold: `{selected['np_p10_threshold']}`",
        f"- count_upper_factor: `{selected['count_upper_factor']}`",
        "",
        "## Selected Val Metrics",
        "",
        f"- Dice: `{float(selected['mean_dice']):.6f}` (delta `{float(selected['delta_dice']):+.6f}`)",
        f"- AJI: `{float(selected['mean_aji']):.6f}` (delta `{float(selected['delta_aji']):+.6f}`)",
        f"- PQ: `{float(selected['mean_pq']):.6f}` (delta `{float(selected['delta_pq']):+.6f}`)",
        f"- precision: `{float(selected['mean_detection_precision']):.6f}` (delta `{float(selected['delta_precision']):+.6f}`)",
        f"- recall: `{float(selected['mean_detection_recall']):.6f}` (delta `{float(selected['delta_recall']):+.6f}`)",
        f"- global pred/GT: `{float(selected['global_pred_gt_instance_ratio']):.6f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
