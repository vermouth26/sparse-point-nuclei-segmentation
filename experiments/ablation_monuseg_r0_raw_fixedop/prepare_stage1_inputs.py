from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DETECTION = ROOT / "detection"
for path in (ROOT, DETECTION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_data import create_detect_label_from_points, split_patches  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare isolated leakage-safe MoNuSeg R0 Stage-I inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-points",
        type=Path,
        required=True,
        help="Fixed legal r010 sparse-point maps generated under the configured annotation seed.",
    )
    parser.add_argument("--force-audit-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["initialization"]["mode"] != "imagenet" or not bool(config["model"]["pretrained"]):
        raise RuntimeError("R0 must use torchvision ImageNet initialization.")
    if bool(config["training"].get("curriculum_use_gt_points", True)):
        raise RuntimeError("TRAIN hidden full-center access must be disabled.")
    smoke_mode = bool(config.get("pipeline", {}).get("smoke_mode", False))
    expected_epochs = 2 if smoke_mode else 100
    expected_update_freq = 1 if smoke_mode else 30
    if (
        int(config["training"]["epochs"]) != expected_epochs
        or int(config["training"]["update_freq"]) != expected_update_freq
    ):
        raise RuntimeError(
            f"Expected {'two-epoch smoke with one-epoch refreshes' if smoke_mode else 'frozen 100-epoch R0 with 30-epoch refreshes'}."
        )

    raw_root = Path(config["dataset"]["raw_data_dir"])
    processed_root = Path(config["dataset"]["processed_data_dir"])
    description = str(config["dataset"]["description"])
    ratio_dir = raw_root / "MO" / f"ratio0.1{description}"
    processed_ratio = processed_root / "MO" / f"ratio0.1{description}"
    source_points = args.source_points.resolve()
    split_path = raw_root / "MO/train_val_test.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [Path(item).stem for item in split["train"]]
    val_ids = [Path(item).stem for item in split["val"]]
    if (len(train_ids), len(val_ids), len(split["test"])) != (24, 6, 14):
        raise RuntimeError("Expected fixed MoNuSeg TRAIN24/VAL6/TEST14 split.")

    required = [source_points / f"{image_id}_label_point.png" for image_id in train_ids + val_ids]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing legal sparse maps:\n" + "\n".join(missing))

    out_manifest = Path(config["dataset"]["save_dir"]) / "stage1_input_manifest.json"
    if args.force_audit_only:
        print(json.dumps({"pass": True, "train": 24, "val": 6, "test_read": False}, indent=2))
        return
    if ratio_dir.exists() or processed_ratio.exists() or out_manifest.exists():
        raise RuntimeError(f"Refusing to overwrite existing R0 inputs: {ratio_dir}, {processed_ratio}, or {out_manifest}")

    point_dir = ratio_dir / "labels_point/0"
    point_dir.mkdir(parents=True)
    hashes = {}
    for image_id in train_ids + val_ids:
        source = source_points / f"{image_id}_label_point.png"
        target = point_dir / source.name
        shutil.copy2(source, target)
        hashes[image_id] = sha256(target)

    detect_dir = ratio_dir / "labels_detect/0"
    create_detect_label_from_points(str(point_dir), str(detect_dir), train_ids, radius=11)
    split_patches(
        str(detect_dir),
        str(processed_ratio / "labels_detect"),
        train_ids,
        post_fix="label_detect",
        patch_size=250,
        n=7,
    )
    payload = {
        "protocol": "MoNuSeg R0 raw fixed operating point v1",
        "split": {"train": train_ids, "val": val_ids, "test_count_only": 14},
        "source_sparse_points": str(source_points),
        "source_sparse_points_note": "Configured ratio-0.10 sparse center annotations used as immutable Stage-I supervision.",
        "destination_sparse_points": str(point_dir),
        "point_sha256": hashes,
        "initialization": "torchvision ImageNet ResNet34",
        "train_hidden_gt_read": False,
        "test_images_read": False,
        "test_gt_read": False,
        "config": str(config_path),
        "smoke_mode": smoke_mode,
        "scientific_result": not smoke_mode,
    }
    out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Prepared isolated R0 Stage-I inputs: {out_manifest}")


if __name__ == "__main__":
    main()
