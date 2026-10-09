from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
OFFICIAL_HOVERNET_DIR = THIS_DIR.parent / "official_hovernet"
if str(OFFICIAL_HOVERNET_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HOVERNET_DIR))

from common_official_hovernet import ensure_dir, write_csv  # noqa: E402
from official_hovernet_postprocess_sweep import (  # noqa: E402
    build_coarse_configs,
    build_refine_configs,
    cfg,
    config_only,
    dedupe_configs,
    evaluate_config,
    evaluate_config_set,
    select_best,
    validate_raw_dir,
)


REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_SPLIT = REPO_ROOT / "data" / "raw" / "MO" / "train_val_test.json"
DEFAULT_LABELS = REPO_ROOT / "data" / "raw" / "MO" / "labels_instance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoNuSeg val6-only configurable official HoVerNet postprocess sweep with fixed test14 evaluation."
    )
    parser.add_argument("--val-raw-dir", type=Path, required=True, help="Directory containing val6 np_prob/ and hv_pred/.")
    parser.add_argument("--test-raw-dir", type=Path, default=None, help="Directory containing test14 np_prob/ and hv_pred/.")
    parser.add_argument("--selection-only", action="store_true", help="Select on val6 and do not read or evaluate test14.")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    parser.add_argument(
        "--preset",
        choices=["full", "light", "aggressive_fp", "smoke"],
        default="full",
        help=(
            "full keeps the original coarse+refine sweep; light runs about 155 val configs; "
            "aggressive_fp focuses on high-precision false-positive control; smoke runs only three "
            "diagnostic candidates and is not a scientific calibration."
        ),
    )
    parser.add_argument(
        "--max-refine-configs",
        type=int,
        default=0,
        help="Optional cap for refine configs after de-duplication. 0 means no cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    val_ids = list(split["val"])
    validate_raw_dir(args.val_raw_dir, val_ids)
    if not args.selection_only:
        if args.test_raw_dir is None:
            raise SystemExit("--test-raw-dir is required unless --selection-only is set")
        test_ids = list(split["test"])
        validate_raw_dir(args.test_raw_dir, test_ids)

    default_cfg = cfg(0.50, 10, 21, 0.40, 5, 10)
    baseline_val_rows, baseline_val = evaluate_config(
        raw_dir=args.val_raw_dir,
        label_dir=args.label_dir,
        image_ids=val_ids,
        config=default_cfg,
        matching_radius=args.matching_radius,
        pred_out_dir=out_dir / "official_default_val_instance_maps",
    )
    write_csv(out_dir / "official_default_val_metrics.csv", baseline_val_rows)
    (out_dir / "official_default_val_overall.json").write_text(json.dumps(baseline_val, indent=2), encoding="utf-8")

    if args.preset == "smoke":
        coarse_configs = dedupe_configs(build_smoke_coarse_configs())
    elif args.preset == "light":
        coarse_configs = dedupe_configs(build_light_coarse_configs())
    elif args.preset == "aggressive_fp":
        coarse_configs = dedupe_configs(build_aggressive_fp_coarse_configs())
    else:
        coarse_configs = dedupe_configs(build_coarse_configs())

    coarse_rows = evaluate_config_set(
        phase="coarse",
        configs=coarse_configs,
        raw_dir=args.val_raw_dir,
        label_dir=args.label_dir,
        image_ids=val_ids,
        matching_radius=args.matching_radius,
        baseline=baseline_val,
    )
    coarse_best = select_best(coarse_rows)

    if args.preset == "smoke":
        refine_configs = dedupe_configs(build_smoke_refine_configs(coarse_best))
    elif args.preset == "light":
        refine_configs = dedupe_configs(build_light_refine_configs(coarse_best))
    elif args.preset == "aggressive_fp":
        refine_configs = dedupe_configs(build_aggressive_fp_refine_configs(coarse_best))
    else:
        refine_configs = dedupe_configs(build_refine_configs(coarse_best))
    if args.max_refine_configs > 0:
        refine_configs = refine_configs[: args.max_refine_configs]

    refine_rows = evaluate_config_set(
        phase="refine",
        configs=refine_configs,
        raw_dir=args.val_raw_dir,
        label_dir=args.label_dir,
        image_ids=val_ids,
        matching_radius=args.matching_radius,
        baseline=baseline_val,
    )
    rows = coarse_rows + refine_rows
    write_csv(out_dir / "official_postprocess_val_sweep.csv", rows)
    selected = select_best(rows)
    (out_dir / "selected_postprocess_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    val_rows, val_overall = evaluate_config(
        raw_dir=args.val_raw_dir,
        label_dir=args.label_dir,
        image_ids=val_ids,
        config=config_only(selected),
        matching_radius=args.matching_radius,
        pred_out_dir=out_dir / "official_postprocess_best_val_instance_maps",
    )
    write_csv(out_dir / "official_postprocess_best_val_metrics.csv", val_rows)
    (out_dir / "official_postprocess_best_val_overall.json").write_text(json.dumps(val_overall, indent=2), encoding="utf-8")

    test_overall = None
    if not args.selection_only:
        test_rows, test_overall = evaluate_config(
            raw_dir=args.test_raw_dir,
            label_dir=args.label_dir,
            image_ids=test_ids,
            config=config_only(selected),
            matching_radius=args.matching_radius,
            pred_out_dir=out_dir / "official_postprocess_best_test_instance_maps",
        )
        write_csv(out_dir / "official_postprocess_best_test_metrics.csv", test_rows)
        (out_dir / "official_postprocess_best_test_overall.json").write_text(json.dumps(test_overall, indent=2), encoding="utf-8")
    write_report(
        out_dir / "MONUSEG_OFFICIAL_HOVERNET_POSTPROCESS_SWEEP_REPORT.md",
        baseline_val,
        selected,
        val_overall,
        test_overall,
        args.preset,
        len(coarse_configs),
        len(refine_configs),
    )
    print(json.dumps({"selected_config": config_only(selected), "val": val_overall, "test": test_overall}, indent=2))


def build_light_coarse_configs() -> list[dict[str, Any]]:
    configs = [cfg(0.50, 10, 21, 0.40, 5, 10)]
    for np_thr in [0.45, 0.50, 0.55, 0.60]:
        for blob_min in [5, 10, 20, 30]:
            for overall_thr in [0.35, 0.40, 0.45, 0.50]:
                for marker_kernel in [5, 7]:
                    configs.append(cfg(np_thr, blob_min, 15, overall_thr, marker_kernel, 5))
    return configs


def build_smoke_coarse_configs() -> list[dict[str, Any]]:
    return [
        cfg(0.50, 10, 21, 0.40, 5, 10),
        # Deliberately permissive plumbing candidate for very short smoke-trained models.
        cfg(0.10, 1, 15, 0.10, 3, 1),
    ]


def build_smoke_refine_configs(best: dict[str, Any]) -> list[dict[str, Any]]:
    best_cfg = config_only(best)
    return [
        cfg(
            max(0.01, round(float(best_cfg["np_threshold"]) - 0.02, 2)),
            int(best_cfg["blob_min_size"]),
            int(best_cfg["sobel_ksize"]),
            float(best_cfg["overall_threshold"]),
            int(best_cfg["marker_kernel_size"]),
            int(best_cfg["marker_min_size"]),
        )
    ]


def build_light_refine_configs(best: dict[str, Any]) -> list[dict[str, Any]]:
    best_cfg = config_only(best)
    np_values = [
        max(0.01, round(float(best_cfg["np_threshold"]) - 0.02, 2)),
        round(float(best_cfg["np_threshold"]), 2),
        min(0.99, round(float(best_cfg["np_threshold"]) + 0.02, 2)),
    ]
    blob_values = sorted({max(1, int(best_cfg["blob_min_size"]) - 10), int(best_cfg["blob_min_size"]), int(best_cfg["blob_min_size"]) + 10})
    overall_values = [
        max(0.01, round(float(best_cfg["overall_threshold"]) - 0.03, 2)),
        round(float(best_cfg["overall_threshold"]), 2),
        min(0.99, round(float(best_cfg["overall_threshold"]) + 0.03, 2)),
    ]
    return [
        cfg(np_thr, blob_min, int(best_cfg["sobel_ksize"]), overall_thr, int(best_cfg["marker_kernel_size"]), int(best_cfg["marker_min_size"]))
        for np_thr in np_values
        for blob_min in blob_values
        for overall_thr in overall_values
    ]


def build_aggressive_fp_coarse_configs() -> list[dict[str, Any]]:
    configs = [cfg(0.50, 10, 21, 0.40, 5, 10)]
    for np_thr in [0.58, 0.60, 0.62, 0.65, 0.68]:
        for blob_min in [20, 30, 50, 70]:
            for overall_thr in [0.45, 0.50, 0.55, 0.60]:
                for marker_min in [5, 10, 20]:
                    configs.append(cfg(np_thr, blob_min, 15, overall_thr, 7, marker_min))
    return configs


def build_aggressive_fp_refine_configs(best: dict[str, Any]) -> list[dict[str, Any]]:
    best_cfg = config_only(best)
    np_values = [
        max(0.01, round(float(best_cfg["np_threshold"]) - 0.02, 2)),
        round(float(best_cfg["np_threshold"]), 2),
        min(0.99, round(float(best_cfg["np_threshold"]) + 0.02, 2)),
    ]
    blob_values = sorted({max(1, int(best_cfg["blob_min_size"]) - 20), int(best_cfg["blob_min_size"]), int(best_cfg["blob_min_size"]) + 20})
    overall_values = [
        max(0.01, round(float(best_cfg["overall_threshold"]) - 0.03, 2)),
        round(float(best_cfg["overall_threshold"]), 2),
        min(0.99, round(float(best_cfg["overall_threshold"]) + 0.03, 2)),
    ]
    marker_min_values = sorted({max(1, int(best_cfg["marker_min_size"]) - 5), int(best_cfg["marker_min_size"]), int(best_cfg["marker_min_size"]) + 5})
    return [
        cfg(np_thr, blob_min, int(best_cfg["sobel_ksize"]), overall_thr, int(best_cfg["marker_kernel_size"]), marker_min)
        for np_thr in np_values
        for blob_min in blob_values
        for overall_thr in overall_values
        for marker_min in marker_min_values
    ]


def write_report(
    path: Path,
    baseline_val: dict[str, Any],
    selected: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any] | None,
    preset: str,
    coarse_count: int,
    refine_count: int,
) -> None:
    report = f"""# MoNuSeg Official HoVerNet Postprocess Sweep

## No-Leakage Statement

- Postprocess parameters were selected on MoNuSeg val6 only.
- Test14 GT is used only for the fixed evaluation with the val-selected config.
- No training is run in this sweep.

## Search Space

- preset: `{preset}`
- coarse configs: `{coarse_count}`
- refine configs: `{refine_count}`
- total val configs excluding default baseline: `{coarse_count + refine_count}`

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

## Fixed Test14 Metrics

{format_test_metrics(test)}
"""
    path.write_text(report, encoding="utf-8")


def format_test_metrics(test: dict[str, Any] | None) -> str:
    if test is None:
        return "Not run. Test evaluation is deferred until checkpoint, postprocess, and pruning are fixed."
    return "\n".join(
        [
            f"- Dice: `{test['mean_dice']:.6f}`",
            f"- AJI: `{test['mean_aji']:.6f}`",
            f"- PQ: `{test['mean_pq']:.6f}`",
            f"- DQ: `{test['mean_dq']:.6f}`",
            f"- SQ: `{test['mean_sq']:.6f}`",
            f"- precision: `{test['mean_detection_precision']:.6f}`",
            f"- recall: `{test['mean_detection_recall']:.6f}`",
            f"- F1: `{test['mean_detection_f1']:.6f}`",
            f"- global pred/GT: `{test['global_pred_gt_instance_ratio']:.6f}`",
        ]
    )


if __name__ == "__main__":
    main()
