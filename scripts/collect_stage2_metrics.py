from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the 12 configured C+fallback VAL6 pseudo-instance candidates.")
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for report in sorted(args.trials_root.glob("*/04_eval/pseudo_region_quality_report.json")):
        trial_root = report.parents[1]
        config = json.loads((trial_root / "03_sam_regions/sam_region_manifest.json").read_text(encoding="utf-8"))["config"]
        summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
        metric = next(row for row in summary if row["method"] == "sam2")
        rows.append(
            {
                "trial": trial_root.name,
                "score_threshold": config["min_sam_score"],
                "min_area": config["min_area"],
                "max_area": config["max_area"],
                "PQ": metric["pq"],
                "AJI": metric["aji"],
                "Precision": metric["detection_precision"],
                "Recall": metric["detection_recall"],
                "pred/GT": metric["pred_gt_count_ratio"],
                "Dice": metric["binary_dice"],
                "DQ": metric["dq"],
                "SQ": metric["sq"],
            }
        )
    if len(rows) != 12:
        raise SystemExit(f"Expected 12 complete trials, found {len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
