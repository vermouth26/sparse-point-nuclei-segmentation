from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
OFFICIAL_HOVERNET_DIR = THIS_DIR.parent / "official_hovernet"
if str(OFFICIAL_HOVERNET_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_HOVERNET_DIR))

from common_official_hovernet import (  # noqa: E402
    add_hovernet_root,
    ensure_dir,
    image_label_paths,
    make_overlay,
    read_instance,
    read_rgb,
    relabel_contiguous,
    safe_div,
    write_csv,
)
from hovernet_evaluation_common import (  # noqa: E402
    compute_metrics,
    sliding_window_predict_official,
    summarize,
    vars_for_json,
)


REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_SPLIT = REPO_ROOT / "data" / "raw" / "MO" / "train_val_test.json"
DEFAULT_IMAGES = REPO_ROOT / "data" / "raw" / "MO" / "images"
DEFAULT_LABELS = REPO_ROOT / "data" / "raw" / "MO" / "labels_instance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official HoVerNet checkpoint on MoNuSeg split.")
    parser.add_argument("--hovernet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--subset", choices=["train", "val", "test"], required=True)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["original", "fast"], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--matching-radius", type=float, default=15.0)
    parser.add_argument("--postprocess", choices=["official", "simple_np"], default="official")
    parser.add_argument("--simple-np-threshold", type=float, default=0.5)
    parser.add_argument("--simple-min-object-size", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug(args, "start")
    add_hovernet_root(args.hovernet_root)
    from models.hovernet.net_desc import create_model  # noqa: WPS433

    process = None
    if args.postprocess == "official":
        from models.hovernet.post_proc import process  # noqa: WPS433

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    mode = args.mode or cfg.get("mode", "original")
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = create_model(input_ch=3, nr_types=None, freeze=False, mode=mode).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    image_ids = list(split[args.subset])
    leakage_note = (
        "Validation GT is used only for checkpoint/postprocess selection."
        if args.subset == "val"
        else "Test GT is used only for this fixed final evaluation."
        if args.subset == "test"
        else "Train GT is used only for diagnostic evaluation."
    )

    out_dir = ensure_dir(args.out_dir)
    pred_dir = ensure_dir(out_dir / "inst_pred")
    np_dir = ensure_dir(out_dir / "np_prob")
    hv_dir = ensure_dir(out_dir / "hv_pred")
    overlay_dir = ensure_dir(out_dir / "overlays")

    rows = []
    for image_id in image_ids:
        image_path, label_path = image_label_paths(image_id, args.images, args.labels)
        image_u8 = read_rgb(image_path)
        gt_inst = read_instance(label_path)
        np_prob, hv_pred = sliding_window_predict_official(
            model=model,
            image_u8=image_u8,
            device=device,
            mode=mode,
            batch_size=args.batch_size,
        )
        pred_map = np.dstack([np_prob, hv_pred[0], hv_pred[1]]).astype(np.float32)
        if args.postprocess == "official":
            assert process is not None
            inst_map, _ = process(pred_map, nr_types=None, return_centroids=False)
        else:
            inst_map = simple_np_postprocess(
                np_prob=np_prob,
                threshold=args.simple_np_threshold,
                min_object_size=args.simple_min_object_size,
            )
        inst_map = relabel_contiguous(inst_map)
        metrics = compute_metrics(inst_map, gt_inst, matching_radius=args.matching_radius)
        row = {
            "image_id": image_id,
            "pred_instance_count": int(inst_map.max()),
            "gt_instance_count": int(gt_inst.max()),
            "pred_gt_instance_ratio": safe_div(float(inst_map.max()), float(gt_inst.max())),
            **metrics,
        }
        rows.append(row)
        np.save(pred_dir / f"{image_id}.npy", inst_map.astype(np.int32))
        np.save(np_dir / f"{image_id}.npy", np_prob.astype(np.float32))
        np.save(hv_dir / f"{image_id}.npy", hv_pred.astype(np.float32))
        Image.fromarray(make_overlay(image_u8, inst_map)).save(overlay_dir / f"{image_id}_overlay.png")
        print(
            f"[monuseg_official_hovernet_{args.subset}] {image_id} dice={metrics['dice']:.4f} "
            f"aji={metrics['aji']:.4f} pq={metrics['pq']:.4f} detection_f1={metrics['detection_f1']:.4f}"
        )

    overall = summarize(rows)
    write_csv(out_dir / "per_image_metrics.csv", rows)
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (out_dir / "eval_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "MoNuSeg official HoVerNet-style image-level inference",
                "subset": args.subset,
                "leakage_note": leakage_note,
                "checkpoint": str(args.checkpoint.resolve()),
                "hovernet_root": str(args.hovernet_root.resolve()),
                "mode": mode,
                "device_used": device_name,
                "image_dir": str(args.images.resolve()),
                "label_dir": str(args.labels.resolve()),
                "overall": overall,
                "config": vars_for_json(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved MoNuSeg official HoVerNet {args.subset} evaluation outputs to: {out_dir.resolve()}")


def simple_np_postprocess(np_prob: np.ndarray, threshold: float, min_object_size: int) -> np.ndarray:
    from scipy import ndimage as ndi

    mask = np_prob >= threshold
    labels, count = ndi.label(mask)
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for idx in range(1, count + 1):
        region = labels == idx
        if int(region.sum()) < min_object_size:
            continue
        out[region] = next_id
        next_id += 1
    return out


def debug(args: argparse.Namespace, message: str) -> None:
    if args.debug:
        print(f"[debug] {message}", flush=True)


if __name__ == "__main__":
    main()
