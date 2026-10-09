from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shuffle(coords: list[tuple[int, int]], image_id: str, seed: int) -> list[tuple[int, int]]:
    local_seed = int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()[:16], 16)
    copied = list(coords)
    random.Random(local_seed).shuffle(copied)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic MoNuSeg sparse-point subset.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    dataset = args.data_root.resolve() / "MO"
    gt_dir = dataset / "labels_point"
    split_path = dataset / "train_val_test.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [Path(item).stem for item in split["train"]]
    val_ids = [Path(item).stem for item in split["val"]]
    image_ids = train_ids + val_ids
    out_dir = args.out_dir or dataset / f"ratio{args.ratio}_nested_seed{args.seed}" / "labels_point" / "0"
    out_dir = out_dir.resolve()
    manifest_path = out_dir.parents[2] / f"seed{args.seed}_point_manifest.json"
    if out_dir.exists() and any(out_dir.glob("*_label_point.png")):
        raise SystemExit(f"Refusing to overwrite existing sparse points: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    train_set = set(train_ids)
    for image_id in image_ids:
        source = gt_dir / f"{image_id}_label_point.png"
        full = imageio.imread(source)
        coords = sorted((int(y), int(x)) for y, x in np.argwhere(full > 0))
        ordered = stable_shuffle(coords, image_id=image_id, seed=args.seed)
        count = max(1, int(round(len(ordered) * args.ratio))) if ordered else 0
        partial = np.zeros_like(full, dtype=np.uint8)
        for y, x in ordered[:count]:
            partial[y, x] = 255
        target = out_dir / f"{image_id}_label_point.png"
        imageio.imwrite(target, partial)
        records.append(
            {
                "image_id": image_id,
                "split": "train" if image_id in train_set else "val",
                "gt_point_count": len(coords),
                "selected_point_count": count,
                "sha256": sha256(target),
            }
        )
    payload = {
        "dataset": "MoNuSeg",
        "ratio": args.ratio,
        "annotation_seed": args.seed,
        "sampling": "per-image SHA256-derived deterministic shuffle over sorted GT center coordinates; round(N*ratio)",
        "scope": "TRAIN24 and VAL6 annotation simulation only; TEST14 not read",
        "test_used": False,
        "point_dir": str(out_dir),
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
