from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def key(row: dict[str, str]) -> tuple[float, ...]:
    return (
        float(row["PQ"]),
        float(row["AJI"]),
        min(float(row["Precision"]), float(row["Recall"])),
        -abs(float(row["pred/GT"]) - 1.0),
        -abs(float(row["score_threshold"]) - 0.65),
        float(row["score_threshold"]),
        float(row["min_area"]),
        -float(row["max_area"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the C+fallback operating point from an existing Val6 table.")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-split", default="VAL6")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.table.open(encoding="utf-8")))
    if len(rows) != 12:
        raise SystemExit(f"Expected exactly 12 C+fallback configurations, found {len(rows)}")
    selected = max(rows, key=key)
    payload = {
        "selection_split": args.selection_split,
        "selection_rule": [
            "maximize macro PQ",
            "maximize AJI",
            "maximize min(Precision, Recall)",
            "minimize abs(pred/GT - 1)",
            "prefer score closest to 0.65",
            "remaining exact ties: higher score, higher min_area, lower max_area",
        ],
        "candidate_count": len(rows),
        "selected": selected,
        "test_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
