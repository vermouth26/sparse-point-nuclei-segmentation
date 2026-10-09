from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric(row: dict, *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"None of {names!r} is present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a HoVerNet checkpoint from VAL-only metric JSON files.")
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.metrics_root.rglob("overall_metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_name = path.parent.name
        checkpoint_path = args.checkpoint_dir / f"{checkpoint_name}.pt"
        rows.append(
            {
                **row,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "metrics_path": str(path.resolve()),
            }
        )
    if not rows:
        raise SystemExit(f"No overall_metrics.json files found below {args.metrics_root}")

    rows.sort(
        key=lambda row: (
            -metric(row, "mean_pq"),
            -metric(row, "mean_aji"),
            -metric(row, "mean_dq"),
            abs(metric(row, "global_pred_gt_instance_ratio") - 1.05),
            row["checkpoint_name"],
        )
    )
    selected = rows[0]
    selected.update(
        {
            "selection_split": "validation only",
            "selection_rule": "PQ, AJI, DQ, then abs(global pred/GT - 1.05)",
            "test_used": False,
            "candidate_count": len(rows),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()

