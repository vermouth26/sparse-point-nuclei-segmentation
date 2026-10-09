from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
OFFICIAL_HOVERNET_DIR = THIS_DIR.parent / "official_hovernet"
if str(OFFICIAL_HOVERNET_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HOVERNET_DIR))

from common_official_hovernet import add_hovernet_root, ensure_dir, read_rgb, relabel_contiguous  # noqa: E402


REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_SPLIT = REPO_ROOT / "data" / "raw" / "MO" / "train_val_test.json"
DEFAULT_IMAGES = REPO_ROOT / "data" / "raw" / "MO" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate official HoVerNet patch .npy files from MoNuSeg weak pseudo instance maps. "
            "Each patch is HWC [R,G,B,pseudo_instance_id], compatible with the official FileLoader."
        )
    )
    parser.add_argument("--hovernet-root", type=Path, required=True)
    parser.add_argument("--hover-target-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--subset", choices=["train", "val", "trainval"], default="train")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--win-size", type=int, default=540)
    parser.add_argument("--step-size", type=int, default=164)
    parser.add_argument("--extract-type", choices=["mirror", "valid"], default="mirror")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_hovernet_root(args.hovernet_root)
    if not hasattr(np.lib, "pad"):
        np.lib.pad = np.pad  # type: ignore[attr-defined]
    from misc.patch_extractor import PatchExtractor  # noqa: WPS433

    split = json.loads(args.split.read_text(encoding="utf-8"))
    split_ids = select_subset(split, args.subset)

    manifest_path = args.hover_target_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inst_dir = Path(manifest["paths"]["inst_map"]).resolve()
    manifest_ids = [row["image_id"] for row in manifest["records"]]
    manifest_id_set = set(manifest_ids)
    missing = [image_id for image_id in split_ids if image_id not in manifest_id_set]
    extra = [image_id for image_id in manifest_ids if image_id not in set(split_ids)]
    if missing:
        raise SystemExit(f"Manifest is missing {len(missing)} requested {args.subset} image(s): {missing[:5]}")
    if extra:
        raise SystemExit(
            f"Manifest contains {len(extra)} image(s) outside requested {args.subset} subset. "
            f"First examples: {extra[:5]}"
        )
    image_ids = split_ids
    if args.expected_count is not None and len(image_ids) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} image(s), found {len(image_ids)} from {args.split}:{args.subset}")

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and args.overwrite:
        for path in out_dir.glob("*.npy"):
            path.unlink()
    ensure_dir(out_dir)
    xtractor = PatchExtractor([args.win_size, args.win_size], [args.step_size, args.step_size])

    records = []
    total_patches = 0
    for image_id in tqdm(image_ids, desc=f"extract MoNuSeg {args.subset} weak official HoVerNet patches"):
        image_path = args.images / f"{image_id}.png"
        inst_path = inst_dir / f"{image_id}.npy"
        if not image_path.exists():
            raise SystemExit(f"Missing image: {image_path}")
        if not inst_path.exists():
            raise SystemExit(f"Missing pseudo instance map: {inst_path}")

        image = read_rgb(image_path)
        pseudo_inst = relabel_contiguous(np.load(inst_path, allow_pickle=False).astype(np.int32))
        if image.shape[:2] != pseudo_inst.shape:
            raise SystemExit(f"Shape mismatch for {image_id}: image={image.shape[:2]} pseudo={pseudo_inst.shape}")

        stacked = np.concatenate([image.astype(np.int32), pseudo_inst[..., None]], axis=-1)
        patches = xtractor.extract(stacked, args.extract_type)
        for idx, patch in enumerate(patches):
            patch = patch.astype(np.int32, copy=False)
            patch[..., :3] = np.clip(patch[..., :3], 0, 255)
            np.save(out_dir / f"{image_id}_{idx:03d}.npy", patch)
        records.append(
            {
                "image_id": image_id,
                "image_shape": list(image.shape),
                "pseudo_instance_count": int(pseudo_inst.max()),
                "patch_count": len(patches),
            }
        )
        total_patches += len(patches)

    out_manifest = {
        "protocol": "MoNuSeg weak official HoVerNet patch set from pseudo instance maps",
        "source_hover_target_manifest": str(manifest_path),
        "source_inst_map_dir": str(inst_dir),
        "split": str(args.split.resolve()),
        "subset": args.subset,
        "images": str(args.images.resolve()),
        "win_size": args.win_size,
        "step_size": args.step_size,
        "extract_type": args.extract_type,
        "format": "HWC numpy array with channels [R,G,B,pseudo_instance_id]",
        "records": records,
        "total_patches": total_patches,
        "note": "Pseudo instance maps come only from the requested split subset. This script does not read GT masks.",
    }
    (out_dir / "patch_manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Saved {total_patches} MoNuSeg weak official HoVerNet patches to: {out_dir}")


def select_subset(split: dict[str, list[str]], subset: str) -> list[str]:
    if subset == "train":
        return list(split["train"])
    if subset == "val":
        return list(split["val"])
    if subset == "trainval":
        return list(split["train"]) + list(split["val"])
    raise SystemExit(f"Unsupported subset: {subset}")


if __name__ == "__main__":
    main()
