from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "experiments/official_hovernet"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from common_official_hovernet import ensure_dir, write_csv  # noqa: E402
from official_hovernet_postprocess_sweep import config_only, evaluate_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one frozen validation-selected postprocess config.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=["val", "test"], required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    args = parser.parse_args()
    image_ids = [Path(item).stem for item in json.loads(args.split.read_text(encoding="utf-8"))[args.subset]]
    selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
    out = ensure_dir(args.out_dir)
    rows, overall = evaluate_config(
        raw_dir=args.raw_dir,
        label_dir=args.label_dir,
        image_ids=image_ids,
        config=config_only(selected),
        matching_radius=args.matching_radius,
        pred_out_dir=out / "instance_maps",
    )
    write_csv(out / "per_image_metrics.csv", rows)
    (out / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (out / "fixed_config.json").write_text(json.dumps(config_only(selected), indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
