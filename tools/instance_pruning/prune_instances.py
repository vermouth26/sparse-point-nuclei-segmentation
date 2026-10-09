from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.instance_pruning.common import (
    ensure_dir,
    group_features_by_image,
    instance_confidence,
    parse_float,
    parse_int,
    read_csv_dicts,
    remap_instances,
    write_json,
)


MODES = ["area_only", "np_only", "np_area", "np_area_center", "np_area_center_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply rule-based instance pruning to predicted instance maps.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--np-mean-threshold", type=float, default=0.4)
    parser.add_argument("--np-p10-threshold", type=float, default=0.1)
    parser.add_argument("--count-upper-factor", type=float, default=1.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "mode": args.mode,
        "min_area": args.min_area,
        "np_mean_threshold": args.np_mean_threshold,
        "np_p10_threshold": args.np_p10_threshold,
        "count_upper_factor": args.count_upper_factor,
    }
    summary = apply_pruning_to_dir(
        pred_dir=args.pred_dir,
        features_csv=args.features_csv,
        out_dir=args.out_dir,
        config=config,
    )
    write_json(args.out_dir / "pruning_summary.json", summary)
    print(f"Saved pruned instance maps to: {args.out_dir.resolve()}")


def apply_pruning_to_dir(
    *,
    pred_dir: Path,
    features_csv: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = read_csv_dicts(features_csv)
    grouped = group_features_by_image(rows)
    out_dir = ensure_dir(out_dir)
    per_image: list[dict[str, Any]] = []
    total_removed = 0
    total_before = 0
    total_after = 0
    for image_id in sorted(grouped):
        inst = np.load(pred_dir / f"{image_id}.npy", allow_pickle=False).astype(np.int32)
        pruned, removed_ids = prune_instance_map(inst, grouped[image_id], config)
        np.save(out_dir / f"{image_id}.npy", pruned.astype(np.int32))
        total_removed += len(removed_ids)
        total_before += int(inst.max())
        total_after += int(pruned.max())
        per_image.append(
            {
                "image_id": image_id,
                "before_count": int(inst.max()),
                "after_count": int(pruned.max()),
                "removed_count": int(len(removed_ids)),
                "removed_ids": ",".join(str(v) for v in removed_ids),
            }
        )
    return {
        "config": config,
        "image_count": len(per_image),
        "total_before_count": total_before,
        "total_after_count": total_after,
        "total_removed_count": total_removed,
        "per_image": per_image,
    }


def prune_instance_map(
    inst: np.ndarray,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    mode = str(config["mode"])
    min_area = int(config.get("min_area", 20))
    np_mean_threshold = float(config.get("np_mean_threshold", 0.4))
    np_p10_threshold = float(config.get("np_p10_threshold", 0.1))
    count_upper_factor = float(config.get("count_upper_factor", 1.1))

    removed: set[int] = set()
    rows_by_id = {parse_int(row.get("instance_id")): row for row in rows}
    for inst_id, row in rows_by_id.items():
        area = parse_int(row.get("area"))
        np_mean = parse_float(row.get("np_mean"))
        np_p10 = parse_float(row.get("np_p10"))
        center_flag = parse_int(row.get("center_inside_flag"))

        low_area = area < min_area
        low_np = np_mean < np_mean_threshold or np_p10 < np_p10_threshold
        should_remove = False
        if mode == "area_only":
            should_remove = low_area
        elif mode == "np_only":
            should_remove = low_np
        elif mode == "np_area":
            should_remove = low_area or low_np
        elif mode in {"np_area_center", "np_area_center_count"}:
            should_remove = (low_area or low_np) and not center_flag
        else:
            raise ValueError(f"Unsupported pruning mode: {mode}")
        if should_remove:
            removed.add(inst_id)

    if mode == "np_area_center_count":
        expected_count = max(parse_int(rows[0].get("expected_center_count_image")) if rows else 0, 0)
        if expected_count > 0:
            upper = max(1, int(math.ceil(expected_count * count_upper_factor)))
            remaining = [row for row in rows if parse_int(row.get("instance_id")) not in removed]
            overflow = len(remaining) - upper
            if overflow > 0:
                ranked = sorted(remaining, key=instance_confidence)
                for row in ranked[:overflow]:
                    removed.add(parse_int(row.get("instance_id")))

    out = inst.copy()
    for inst_id in removed:
        out[out == inst_id] = 0
    return remap_instances(out), sorted(removed)


if __name__ == "__main__":
    main()
