from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official SAM2 helper for batched point-prompt mask prediction.")
    parser.add_argument("--image-npy", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-variant", default="sam2.1_hiera_base_plus")
    parser.add_argument("--confirm-official-general-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_npy).resolve()
    prompts_path = Path(args.prompts_json).resolve()
    output_path = Path(args.output_npz).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    model_config = Path(args.model_config).resolve()

    if not checkpoint.exists():
        raise SystemExit(f"SAM2 checkpoint not found: {checkpoint}")
    if not model_config.exists():
        raise SystemExit(f"SAM2 config not found: {model_config}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise SystemExit(
            "Failed to import official SAM2 package. Install SAM2 in this Python environment first.\n"
            "Expected imports: sam2.build_sam.build_sam2 and sam2.sam2_image_predictor.SAM2ImagePredictor.\n"
            f"Original error: {exc}"
        )

    image = np.load(image_path, allow_pickle=False)
    if image.ndim != 3 or image.shape[2] != 3:
        raise SystemExit(f"Expected RGB image array [H,W,3], got {image.shape}")
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    model, used_model_config = build_sam2_with_config_fallback(build_sam2, model_config, checkpoint, args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    setup_seconds = time.perf_counter() - t0

    all_masks: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_prompt_candidate_count: list[int] = []
    predict_seconds = 0.0
    for prompt in prompts:
        positive_xy = tuple(prompt["positive_xy"])
        negative_xy = [tuple(item) for item in prompt["negative_xy"]]
        points = [positive_xy] + negative_xy
        labels = [1] + [0] * len(negative_xy)
        coords = np.asarray(points, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)
        t1 = time.perf_counter()
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        predict_seconds += time.perf_counter() - t1
        masks = canonicalize_masks(masks, image.shape[:2])
        scores = canonicalize_scores(scores, masks.shape[0])
        all_masks.append(masks.astype(np.uint8))
        all_scores.append(scores.astype(np.float32))
        per_prompt_candidate_count.append(int(masks.shape[0]))

    if all_masks:
        candidate_count = max(item.shape[0] for item in all_masks)
        masks_arr = np.zeros((len(all_masks), candidate_count, image.shape[0], image.shape[1]), dtype=np.uint8)
        scores_arr = np.zeros((len(all_scores), candidate_count), dtype=np.float32)
        for idx, (masks, scores) in enumerate(zip(all_masks, all_scores)):
            masks_arr[idx, : masks.shape[0]] = masks
            scores_arr[idx, : scores.shape[0]] = scores
    else:
        masks_arr = np.zeros((0, 3, image.shape[0], image.shape[1]), dtype=np.uint8)
        scores_arr = np.zeros((0, 3), dtype=np.float32)

    meta = {
        "backend": "sam2",
        "sam2_version": get_sam2_version(),
        "model_variant": args.model_variant,
        "model_config": str(model_config),
        "model_config_used_by_build_sam2": used_model_config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": args.device,
        "target_dense_mask_exposure": False if args.confirm_official_general_checkpoint else "unknown",
        "input_image_shape": [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
        "wrapper_pre_resize": False,
        "prompt_coordinate_convention": "xy",
        "array_coordinate_convention": "row_col_yx",
        "multimask_output": True,
        "prompt_count": int(len(prompts)),
        "positive_point_count": int(len(prompts)),
        "negative_point_count": int(sum(len(item["negative_xy"]) for item in prompts)),
        "per_prompt_candidate_count": per_prompt_candidate_count,
        "output_masks_shape": list(masks_arr.shape),
        "output_scores_shape": list(scores_arr.shape),
        "setup_seconds": float(setup_seconds),
        "predict_seconds": float(predict_seconds),
        "note": "SAM2ImagePredictor outputs are canonicalized to masks [num_prompts,num_candidates,H,W] and scores [num_prompts,num_candidates].",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        masks=masks_arr,
        scores=scores_arr,
        meta_json=np.asarray(json.dumps(meta)),
    )


def canonicalize_masks(masks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise SystemExit(f"SAM2 returned masks with unsupported shape {arr.shape}")
    h, w = image_shape
    if arr.shape[-2:] == (h, w):
        return arr.astype(bool)
    if arr.shape[:2] == (h, w):
        return np.moveaxis(arr, -1, 0).astype(bool)
    raise SystemExit(f"SAM2 returned masks shape {arr.shape}, incompatible with image shape {(h, w)}")


def build_sam2_with_config_fallback(build_sam2_func, model_config: Path, checkpoint: Path, device: str):
    errors = []
    for config_name in candidate_config_names(model_config):
        try:
            return build_sam2_func(config_name, str(checkpoint), device=device), config_name
        except Exception as exc:
            errors.append(f"{config_name}: {exc}")
    raise SystemExit("Failed to initialize SAM2 with all config path variants:\n" + "\n".join(errors))


def candidate_config_names(model_config: Path) -> list[str]:
    names = [str(model_config)]
    parts = list(model_config.parts)
    if "configs" in parts:
        idx = parts.index("configs")
        names.append("/".join(parts[idx:]))
    unique = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def canonicalize_scores(scores: np.ndarray, candidate_count: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == candidate_count:
        return arr
    if arr.size == 1 and candidate_count == 1:
        return arr
    raise SystemExit(f"SAM2 returned scores shape {np.asarray(scores).shape}, candidate_count={candidate_count}")


def get_sam2_version() -> str:
    for name in ("sam2", "SAM-2", "segment-anything-2"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
