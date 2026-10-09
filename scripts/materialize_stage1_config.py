from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a path-local Stage1 config without changing experiment parameters.")
    parser.add_argument("--template", type=Path, default=Path("configs/stage1_ratio010.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Materialize a two-epoch integration config with the curriculum-refresh path enabled. Never use it as a result.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    config["dataset"]["raw_data_dir"] = str(args.data_root.resolve())
    config["dataset"]["processed_data_dir"] = str(args.processed_root.resolve())
    config["dataset"]["save_dir"] = str(args.output_root.resolve())
    config["dataset"]["description"] = args.description
    config["training"]["seed"] = args.seed
    if args.smoke:
        config["pipeline"]["smoke_mode"] = True
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = min(int(config["training"]["batch_size"]), 16)
        config["training"]["checkpoint_freq"] = 1
        config["training"]["update_freq"] = 1
        config["training"]["workers"] = 0
        config["training"]["log_interval"] = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
