from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    binary_dilation,
    clu_label_to_instance_map,
    colorize_instances,
    decode_clu,
    ensure_dir,
    existing_pred_point_path,
    image_path,
    imread,
    mask_boundary,
    nearest_positive_label,
    partial_point_path,
    point_map_to_coords,
    read_rgb,
    relabel_contiguous,
    save_csv,
    save_instance_png,
    save_json,
    save_png,
    tile_images,
    verify_dataset_layout,
    write_step_report,
)


class SamBackend:
    def __init__(self, model_type: str, checkpoint: Path, device: str) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception:
            raise SystemExit(
                "Missing package: segment_anything. Install it in the target environment first, then rerun this script."
            )
        if not checkpoint.exists():
            raise SystemExit(
                f"SAM checkpoint not found: {checkpoint}\n"
                "Provide --sam-checkpoint pointing to a valid checkpoint such as sam_vit_b_01ec64.pth."
            )
        try:
            model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        except KeyError:
            raise SystemExit(f"Unsupported SAM model type: {model_type}")
        except Exception as exc:
            raise SystemExit(f"Failed to initialize SAM model from checkpoint: {exc}")
        self.predictor = SamPredictor(model)
        self.predictor.model.to(device=device)
        self.device = device

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.predictor.set_image(image_rgb)

    def predict(self, positive_xy: tuple[int, int], negative_xy: list[tuple[int, int]]) -> list[dict[str, Any]]:
        points = [positive_xy] + negative_xy
        labels = [1] + [0] * len(negative_xy)
        coords = np.asarray(points, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        candidates = []
        for idx in range(masks.shape[0]):
            candidates.append(
                {
                    "mask": masks[idx].astype(bool),
                    "sam_score": float(scores[idx]),
                }
            )
        return candidates

    def predict_many(self, prompts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        outputs = []
        for prompt in prompts:
            outputs.append(self.predict(tuple(prompt["positive_xy"]), [tuple(v) for v in prompt["negative_xy"]]))
        return outputs


class PathoSamBackend:
    def __init__(
        self,
        python_exe: Path,
        model_type: str,
        checkpoint: Path | None,
        device: str,
        cache_dir: Path,
        helper_script: Path,
    ) -> None:
        if not python_exe.exists():
            raise SystemExit(f"PathoSAM python not found: {python_exe}")
        if not helper_script.exists():
            raise SystemExit(f"PathoSAM helper script not found: {helper_script}")
        self.python_exe = python_exe
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.cache_dir = cache_dir
        self.helper_script = helper_script
        self.image_rgb: np.ndarray | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_rgb = image_rgb

    def predict_many(self, prompts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if self.image_rgb is None:
            raise RuntimeError("PathoSamBackend.set_image must be called before predict_many.")
        with tempfile.TemporaryDirectory(prefix="pathosam_backend_", dir=str(self.cache_dir.parent)) as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_npy = tmpdir_path / "image.npy"
            prompts_json = tmpdir_path / "prompts.json"
            output_npz = tmpdir_path / "predictions.npz"
            np.save(image_npy, self.image_rgb.astype(np.uint8))
            prompts_json.write_text(json.dumps(prompts), encoding="utf-8")

            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            patho_root = self.python_exe.parent.parent
            extra_path = f"{patho_root};{patho_root / 'Library' / 'bin'};{patho_root / 'Scripts'}"
            env["PATH"] = extra_path + ";" + env.get("PATH", "")
            env["MICROSAM_CACHEDIR"] = str(self.cache_dir.resolve())

            cmd = [
                str(self.python_exe),
                str(self.helper_script),
                "--image-npy",
                str(image_npy),
                "--prompts-json",
                str(prompts_json),
                "--output-npz",
                str(output_npz),
                "--model-type",
                self.model_type,
                "--device",
                self.device,
                "--cache-dir",
                str(self.cache_dir),
            ]
            if self.checkpoint is not None:
                cmd.extend(["--checkpoint", str(self.checkpoint)])
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise SystemExit(
                    "PathoSAM helper failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )
            with np.load(output_npz) as payload:
                masks_arr = payload["masks"].copy()
                scores_arr = payload["scores"].copy()
            results: list[list[dict[str, Any]]] = []
            for idx in range(masks_arr.shape[0]):
                cur = []
                for jdx in range(masks_arr.shape[1]):
                    cur.append(
                        {
                            "mask": masks_arr[idx, jdx].astype(bool),
                            "sam_score": float(scores_arr[idx, jdx]),
                        }
                    )
                results.append(cur)
            return results


class Sam2Backend:
    def __init__(
        self,
        python_exe: Path,
        checkpoint: Path,
        model_config: Path,
        device: str,
        helper_script: Path,
        model_variant: str,
        confirm_official_general_checkpoint: bool,
    ) -> None:
        if not python_exe.exists():
            raise SystemExit(f"SAM2 python not found: {python_exe}")
        if not checkpoint.exists():
            raise SystemExit(
                f"SAM2 checkpoint not found: {checkpoint}\n"
                "Download the official SAM 2.1 checkpoint yourself, e.g. sam2.1_hiera_base_plus.pt, "
                "and pass its absolute path with --sam2-checkpoint. This script never downloads weights silently."
            )
        if not model_config.exists():
            raise SystemExit(
                f"SAM2 model config not found: {model_config}\n"
                "Pass the official config path, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml from the SAM2 repository."
            )
        if not helper_script.exists():
            raise SystemExit(f"SAM2 helper script not found: {helper_script}")
        self.python_exe = python_exe
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.device = device
        self.helper_script = helper_script
        self.model_variant = model_variant
        self.confirm_official_general_checkpoint = confirm_official_general_checkpoint
        self.image_rgb: np.ndarray | None = None
        self.last_meta: dict[str, Any] = {}

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_rgb = image_rgb

    def predict_many(self, prompts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if self.image_rgb is None:
            raise RuntimeError("Sam2Backend.set_image must be called before predict_many.")
        with tempfile.TemporaryDirectory(prefix="sam2_backend_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_npy = tmpdir_path / "image.npy"
            prompts_json = tmpdir_path / "prompts.json"
            output_npz = tmpdir_path / "predictions.npz"
            np.save(image_npy, self.image_rgb.astype(np.uint8))
            prompts_json.write_text(json.dumps(prompts), encoding="utf-8")

            env = os.environ.copy()
            sam2_root = self.python_exe.parent.parent
            extra_path = os.pathsep.join(
                [
                    str(sam2_root),
                    str(sam2_root / "Library" / "bin"),
                    str(sam2_root / "Scripts"),
                ]
            )
            env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")

            cmd = [
                str(self.python_exe),
                str(self.helper_script),
                "--image-npy",
                str(image_npy),
                "--prompts-json",
                str(prompts_json),
                "--output-npz",
                str(output_npz),
                "--checkpoint",
                str(self.checkpoint),
                "--model-config",
                str(self.model_config),
                "--device",
                self.device,
                "--model-variant",
                self.model_variant,
            ]
            if self.confirm_official_general_checkpoint:
                cmd.append("--confirm-official-general-checkpoint")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise SystemExit(
                    "SAM2 helper failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )
            with np.load(output_npz) as payload:
                masks_arr = payload["masks"].copy()
                scores_arr = payload["scores"].copy()
                if "meta_json" in payload:
                    self.last_meta = json.loads(str(payload["meta_json"].item()))
                else:
                    self.last_meta = {}
            if masks_arr.ndim != 4:
                raise SystemExit(f"SAM2 helper returned masks with invalid shape {masks_arr.shape}; expected [N,C,H,W].")
            if scores_arr.ndim != 2:
                raise SystemExit(f"SAM2 helper returned scores with invalid shape {scores_arr.shape}; expected [N,C].")
            if masks_arr.shape[0] != len(prompts) or scores_arr.shape[0] != len(prompts):
                raise SystemExit(
                    "SAM2 helper prompt count mismatch: "
                    f"prompts={len(prompts)}, masks={masks_arr.shape}, scores={scores_arr.shape}"
                )
            results: list[list[dict[str, Any]]] = []
            for idx in range(masks_arr.shape[0]):
                cur = []
                for jdx in range(masks_arr.shape[1]):
                    # SAM2 returns [C,H,W] candidates; downstream uses one Boolean
                    # HxW mask and score for each candidate.
                    cur.append(
                        {
                            "mask": masks_arr[idx, jdx].astype(bool),
                            "sam_score": float(scores_arr[idx, jdx]),
                        }
                    )
                results.append(cur)
            return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM point-prompt pseudo regions from merged BoNuS stage1 centers.")
    parser.add_argument("--points-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--backend", choices=["sam", "pathosam", "sam2"], default="sam")
    parser.add_argument("--preset", choices=["none", "hover_conservative_v1", "hover_balanced_v1"], default="none")
    parser.add_argument("--sam-checkpoint", default="")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--pathosam-python", default="", help="Python executable for a PathoSAM environment; required for backend=pathosam.")
    parser.add_argument("--pathosam-model-type", default="vit_b_histopathology")
    parser.add_argument("--pathosam-checkpoint", default="", help="Optional custom PathoSAM checkpoint. Leave empty to use official model download/cache.")
    parser.add_argument("--pathosam-device", default="cuda")
    parser.add_argument("--pathosam-cache-dir", default="", help="Where micro_sam / PathoSAM should cache downloaded official model weights.")
    parser.add_argument("--sam2-python", default="", help="Python executable for an environment with the official SAM2 package. Defaults to current Python.")
    parser.add_argument("--sam2-checkpoint", default="", help="Required for backend=sam2. Example: /path/to/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-model-config", default="", help="Required for backend=sam2. Example: /path/to/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2-model-variant", default="sam2.1_hiera_base_plus")
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument(
        "--sam2-confirm-official-general-checkpoint",
        action="store_true",
        help="Set target_dense_mask_exposure=false in the manifest. Use only after verifying this is an official general SAM2 checkpoint.",
    )
    parser.add_argument("--negative-radius", type=float, default=48.0)
    parser.add_argument("--max-negative-points", type=int, default=6)
    parser.add_argument("--negative-strategy", choices=["radius", "nearest"], default="radius")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area", type=int, default=5000)
    parser.add_argument("--min-sam-score", type=float, default=0.0)
    parser.add_argument("--min-vor-iou", type=float, default=0.05)
    parser.add_argument("--min-cell-coverage", type=float, default=0.05)
    parser.add_argument("--max-outside-ratio", type=float, default=0.65)
    parser.add_argument(
        "--candidate-selection-mode",
        choices=["geometry_constrained", "sam_only"],
        default="geometry_constrained",
        help=(
            "geometry_constrained applies Voronoi overlap gates and geometry-aware ranking; "
            "sam_only keeps center/area/score checks but does not gate or rank by Voronoi/cluster geometry."
        ),
    )
    parser.add_argument("--strict-center-exclusion-radius", type=int, default=0)
    parser.add_argument(
        "--fallback",
        choices=["ignore", "voronoi", "cluster", "sam_then_cluster"],
        default="cluster",
        help=(
            "Fallback after strict candidate selection. sam_then_cluster first reuses a candidate that passes "
            "the SAM-only safety checks, and uses cluster only when no such image-conditioned mask exists."
        ),
    )
    parser.add_argument("--boundary-dilate", type=int, default=1)
    parser.add_argument("--tie-margin", type=float, default=0.015)
    parser.add_argument("--conflict-policy", choices=["winner", "ignore"], default="winner")
    parser.add_argument("--save-candidate-visualizations", action="store_true")
    parser.add_argument("--max-candidate-visualizations", type=int, default=8)
    parser.add_argument(
        "--candidate-cache-dir",
        default="",
        help="Optional shared raw candidate cache. Stores only prompt masks/scores before selection, geometry, or fallback.",
    )
    parser.add_argument(
        "--candidate-cache-mode",
        choices=["off", "auto", "write", "reuse"],
        default="off",
        help="auto reuses existing per-image candidates and writes missing entries; reuse fails if any entry is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_preset(args)
    points_manifest = Path(args.points_manifest).resolve()
    baseline_manifest = Path(args.baseline_manifest).resolve()
    if not points_manifest.exists():
        raise SystemExit(f"Points manifest not found: {points_manifest}")
    if not baseline_manifest.exists():
        raise SystemExit(f"Baseline manifest not found: {baseline_manifest}")
    points_payload = json.loads(points_manifest.read_text(encoding="utf-8"))
    baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    out_root = Path(args.out_dir).resolve() if args.out_dir else points_manifest.parents[1]
    step_root = ensure_dir(out_root / "03_sam_regions")
    sam_inst_dir = ensure_dir(step_root / "sam_instance_map")
    sam_bin_dir = ensure_dir(step_root / "sam_binary_np")
    sam_boundary_dir = ensure_dir(step_root / "sam_boundary")
    sam_ignore_dir = ensure_dir(step_root / "sam_ignore")
    overlay_dir = ensure_dir(step_root / "overlays")
    candidate_overlay_dir = ensure_dir(step_root / "candidate_overlays") if args.save_candidate_visualizations else None

    raw_root = Path(points_payload["raw_root"]).resolve()
    processed_root = Path(points_payload["processed_root"]).resolve()
    dataset = points_payload["dataset"]
    ratio = float(points_payload["ratio"])
    description = str(points_payload["description"])
    layout = verify_dataset_layout(raw_root, processed_root, dataset, ratio, description)
    merged_dir = Path(points_payload["paths"]["merged_dir"]).resolve()
    vor_inst_dir = Path(baseline_payload["paths"]["bonus_vor_instance_dir"]).resolve()
    clu_inst_dir = Path(baseline_payload["paths"]["bonus_clu_instance_dir"]).resolve()
    clu_raw_dir = Path(baseline_payload["paths"]["labels_clu_dir"]).resolve()
    candidate_cache_dir = Path(args.candidate_cache_dir).resolve() if args.candidate_cache_dir else None
    if args.candidate_cache_mode != "off" and candidate_cache_dir is None:
        raise SystemExit("--candidate-cache-dir is required when candidate cache mode is not off")
    if candidate_cache_dir is not None:
        candidate_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "sam":
        checkpoint = Path(args.sam_checkpoint).resolve() if args.sam_checkpoint else None
        if checkpoint is None:
            raise SystemExit(
                "Missing --sam-checkpoint.\n"
                "Pass a SAM ViT-B checkpoint such as sam_vit_b_01ec64.pth.\n"
                "Example: --sam-checkpoint /path/to/sam_vit_b_01ec64.pth"
            )
        backend_device = args.sam_device if args.sam_device == "cpu" or torch_cuda_available() else "cpu"
        backend = SamBackend(args.sam_model_type, checkpoint, backend_device)
        backend_meta = {
            "checkpoint": str(checkpoint),
            "model_type": args.sam_model_type,
            "device": backend_device,
        }
    elif args.backend == "pathosam":
        if not args.pathosam_python:
            raise SystemExit("--pathosam-python is required when backend=pathosam")
        pathosam_python = Path(args.pathosam_python).resolve()
        pathosam_checkpoint = Path(args.pathosam_checkpoint).resolve() if args.pathosam_checkpoint else None
        pathosam_device = args.pathosam_device
        helper_script = Path(__file__).resolve().parent / "pathosam_point_backend.py"
        cache_dir = Path(args.pathosam_cache_dir).resolve() if args.pathosam_cache_dir else ensure_dir(step_root / "pathosam_cache")
        backend = PathoSamBackend(
            python_exe=pathosam_python,
            model_type=args.pathosam_model_type,
            checkpoint=pathosam_checkpoint,
            device=pathosam_device,
            cache_dir=cache_dir,
            helper_script=helper_script,
        )
        backend_meta = {
            "checkpoint": str(pathosam_checkpoint) if pathosam_checkpoint else None,
            "model_type": args.pathosam_model_type,
            "device": pathosam_device,
            "python": str(pathosam_python),
            "cache_dir": str(cache_dir),
            "target_dense_mask_exposure": "target-exposed PathoSAM generalist upper bound",
        }
    else:
        sam2_python = Path(args.sam2_python).resolve() if args.sam2_python else Path(sys.executable).resolve()
        sam2_checkpoint = Path(args.sam2_checkpoint).resolve() if args.sam2_checkpoint else None
        sam2_model_config = Path(args.sam2_model_config).resolve() if args.sam2_model_config else None
        if sam2_checkpoint is None:
            raise SystemExit(
                "Missing --sam2-checkpoint for backend=sam2.\n"
                "This script will not download SAM2 weights silently. Download an official checkpoint such as "
                "sam2.1_hiera_base_plus.pt and pass its absolute path."
            )
        if sam2_model_config is None:
            raise SystemExit(
                "Missing --sam2-model-config for backend=sam2.\n"
                "Pass the official SAM2 config, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml."
            )
        helper_script = Path(__file__).resolve().parent / "sam2_point_backend.py"
        backend = Sam2Backend(
            python_exe=sam2_python,
            checkpoint=sam2_checkpoint,
            model_config=sam2_model_config,
            device=args.sam2_device,
            helper_script=helper_script,
            model_variant=args.sam2_model_variant,
            confirm_official_general_checkpoint=args.sam2_confirm_official_general_checkpoint,
        )
        backend_meta = {
            "checkpoint": str(sam2_checkpoint),
            "checkpoint_sha256": sha256_file(sam2_checkpoint),
            "model_config": str(sam2_model_config),
            "model_variant": args.sam2_model_variant,
            "device": args.sam2_device,
            "python": str(sam2_python),
            "helper_script": str(helper_script),
            "multimask_output": True,
            "prompt_coordinate_convention": "xy",
            "mask_output_shape": "[num_prompts, num_candidates, H, W]",
            "score_dtype": "float32",
            "mask_dtype": "uint8/bool",
            "target_dense_mask_exposure": False if args.sam2_confirm_official_general_checkpoint else "unknown",
            "git_hash": get_git_hash(),
        }

    cache_context = {
        "points_manifest_sha256": sha256_file(points_manifest),
        "backend": args.backend,
        "negative_strategy": args.negative_strategy,
        "negative_radius": float(args.negative_radius),
        "max_negative_points": int(args.max_negative_points),
        "sam2_checkpoint_sha256": backend_meta.get("checkpoint_sha256"),
        "sam2_model_config": backend_meta.get("model_config"),
        "sam2_model_variant": backend_meta.get("model_variant"),
        "prompt_coordinate_convention": backend_meta.get("prompt_coordinate_convention", "xy"),
    }
    cache_manifest_path = candidate_cache_dir / "candidate_cache_manifest.json" if candidate_cache_dir else None
    if cache_manifest_path is not None and cache_manifest_path.is_file():
        prior_cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if prior_cache.get("context") != cache_context:
            raise SystemExit(
                "Candidate cache context mismatch; refusing to mix centers, prompts, or SAM2 checkpoints:\n"
                + json.dumps({"expected": cache_context, "found": prior_cache.get("context")}, indent=2)
            )

    per_instance_rows = []
    per_image_rows = []
    per_prompt_rows = []

    candidate_vis_count = 0
    for row in points_payload["records"]:
        stem = row["image_id"]
        image_rgb = read_rgb(image_path(layout["dataset_dir"], stem))
        merged_map = imread(merged_dir / f"{stem}_label_point.png").astype(np.uint8)
        coords = point_map_to_coords(merged_map)
        vor_inst = imread(vor_inst_dir / f"{stem}.png").astype(np.int32)
        clu_inst = imread(clu_inst_dir / f"{stem}.png").astype(np.int32)
        clu_raw = read_rgb(clu_raw_dir / f"{stem}_label_clu.png")
        fallback_region_map = clu_label_to_instance_map(clu_raw, merged_map) if args.fallback == "cluster" else clu_inst

        selected_regions = {}
        prompt_specs = []
        for y, x in coords:
            negative = choose_negative_points(
                (y, x),
                coords,
                args.negative_radius,
                args.max_negative_points,
                args.negative_strategy,
            )
            prompt_specs.append(
                {
                    "positive_xy": [int(x), int(y)],
                    "negative_xy": [[int(xx), int(yy)] for yy, xx in negative],
                }
            )
        cache_path = candidate_cache_dir / f"{stem}.npz" if candidate_cache_dir else None
        use_cached = cache_path is not None and cache_path.is_file() and args.candidate_cache_mode in {"auto", "reuse"}
        if use_cached:
            with np.load(cache_path, allow_pickle=False) as cached:
                masks_arr = cached["masks"].astype(bool)
                scores_arr = cached["scores"].astype(np.float32)
                cached_prompts = json.loads(str(cached["prompts_json"].item()))
            if cached_prompts != prompt_specs:
                raise SystemExit(f"Candidate cache prompt mismatch: {cache_path}")
            if masks_arr.ndim != 4 or scores_arr.ndim != 2 or masks_arr.shape[:2] != scores_arr.shape:
                raise SystemExit(f"Invalid candidate cache shapes in {cache_path}: {masks_arr.shape}, {scores_arr.shape}")
            batched_candidates = [
                [
                    {"mask": masks_arr[i, j], "sam_score": float(scores_arr[i, j])}
                    for j in range(masks_arr.shape[1])
                ]
                for i in range(masks_arr.shape[0])
            ]
            predict_seconds = 0.0
        else:
            if args.candidate_cache_mode == "reuse":
                raise SystemExit(f"Required candidate cache entry is missing: {cache_path}")
            if args.candidate_cache_mode == "write" and cache_path is not None and cache_path.exists():
                raise SystemExit(f"Refusing to overwrite candidate cache entry: {cache_path}")
            backend.set_image(image_rgb)
            t0 = time.perf_counter()
            batched_candidates = backend.predict_many(prompt_specs)
            predict_seconds = time.perf_counter() - t0
            if args.backend == "sam2" and isinstance(backend, Sam2Backend):
                backend_meta.update({f"last_{k}": v for k, v in backend.last_meta.items()})
            if cache_path is not None and args.candidate_cache_mode in {"auto", "write"}:
                counts = {len(items) for items in batched_candidates}
                if len(counts) != 1:
                    raise SystemExit(f"Candidate cache requires a fixed candidate count per prompt, found {sorted(counts)}")
                masks_arr = np.asarray(
                    [[item["mask"].astype(np.uint8) for item in items] for items in batched_candidates],
                    dtype=np.uint8,
                )
                scores_arr = np.asarray(
                    [[float(item["sam_score"]) for item in items] for items in batched_candidates],
                    dtype=np.float32,
                )
                np.savez_compressed(
                    cache_path,
                    masks=masks_arr,
                    scores=scores_arr,
                    prompts_json=np.asarray(json.dumps(prompt_specs)),
                )
        candidate_counts = [len(items) for items in batched_candidates]
        candidate_areas = [int(np.sum(item["mask"])) for items in batched_candidates for item in items]
        candidate_scores = [float(item["sam_score"]) for items in batched_candidates for item in items]
        for prompt_index, (prompt, candidates) in enumerate(zip(prompt_specs, batched_candidates)):
            areas = [int(np.sum(item["mask"])) for item in candidates]
            scores = [float(item["sam_score"]) for item in candidates]
            per_prompt_rows.append(
                {
                    "image_id": stem,
                    "prompt_index": int(prompt_index),
                    "positive_x": int(prompt["positive_xy"][0]),
                    "positive_y": int(prompt["positive_xy"][1]),
                    "negative_point_count": int(len(prompt["negative_xy"])),
                    "candidate_count": int(len(candidates)),
                    "candidate_area_mean": float(np.mean(areas)) if areas else 0.0,
                    "candidate_area_min": int(min(areas)) if areas else 0,
                    "candidate_area_max": int(max(areas)) if areas else 0,
                    "candidate_score_mean": float(np.mean(scores)) if scores else 0.0,
                    "candidate_score_min": float(min(scores)) if scores else 0.0,
                    "candidate_score_max": float(max(scores)) if scores else 0.0,
                    "empty_mask_ratio": float(np.mean([area == 0 for area in areas])) if areas else 0.0,
                    "tiny_mask_ratio": float(np.mean([0 < area < args.min_area for area in areas])) if areas else 0.0,
                    "large_mask_ratio": float(np.mean([area > args.max_area for area in areas])) if areas else 0.0,
                }
            )
        if candidate_overlay_dir is not None and candidate_vis_count < args.max_candidate_visualizations:
            for prompt, candidates in zip(prompt_specs, batched_candidates):
                if candidate_vis_count >= args.max_candidate_visualizations:
                    break
                save_candidate_panel(
                    image_rgb=image_rgb,
                    positive_xy=tuple(prompt["positive_xy"]),
                    negative_xy=[tuple(v) for v in prompt["negative_xy"]],
                    candidates=candidates,
                    out_path=candidate_overlay_dir / f"{stem}_prompt{candidate_vis_count:03d}_candidates.png",
                )
                candidate_vis_count += 1
        for inst_idx, (y, x) in enumerate(coords, start=1):
            candidates = batched_candidates[inst_idx - 1]
            vor_region = region_for_center(vor_inst, (y, x))
            chosen = select_candidate(
                center=(y, x),
                all_centers=coords,
                candidates=candidates,
                vor_region=vor_region,
                min_area=args.min_area,
                max_area=args.max_area,
                min_sam_score=args.min_sam_score,
                min_vor_iou=args.min_vor_iou,
                min_cell_coverage=args.min_cell_coverage,
                max_outside_ratio=args.max_outside_ratio,
                strict_center_exclusion_radius=args.strict_center_exclusion_radius,
                selection_mode=args.candidate_selection_mode,
            )
            geometry_rejected = False
            if chosen is None and args.candidate_selection_mode == "geometry_constrained":
                sam_only_audit = select_candidate(
                    center=(y, x),
                    all_centers=coords,
                    candidates=candidates,
                    vor_region=vor_region,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    min_sam_score=args.min_sam_score,
                    min_vor_iou=args.min_vor_iou,
                    min_cell_coverage=args.min_cell_coverage,
                    max_outside_ratio=args.max_outside_ratio,
                    strict_center_exclusion_radius=args.strict_center_exclusion_radius,
                    selection_mode="sam_only",
                )
                geometry_rejected = sam_only_audit is not None
            if chosen is None and args.fallback == "sam_then_cluster":
                chosen = select_candidate(
                    center=(y, x),
                    all_centers=coords,
                    candidates=candidates,
                    vor_region=vor_region,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    min_sam_score=args.min_sam_score,
                    min_vor_iou=args.min_vor_iou,
                    min_cell_coverage=args.min_cell_coverage,
                    max_outside_ratio=args.max_outside_ratio,
                    strict_center_exclusion_radius=args.strict_center_exclusion_radius,
                    selection_mode="sam_only",
                )
                if chosen is not None:
                    chosen["status"] = "sam_relaxed"
            if chosen is None:
                fallback_mode = "cluster" if args.fallback == "sam_then_cluster" else args.fallback
                fallback_mask, fallback_source = build_fallback_region(
                    fallback_mode,
                    center=(y, x),
                    vor_inst=vor_inst,
                    clu_inst=fallback_region_map,
                )
                if fallback_mask is None:
                    per_instance_rows.append(
                        {
                            "image_id": stem,
                            "instance_id": inst_idx,
                            "center_y": y,
                            "center_x": x,
                            "status": "ignore",
                            "source": "ignore",
                            "sam_score": 0.0,
                            "area": 0,
                            "contained_center_count": 1,
                            "other_center_count": 0,
                            "vor_iou": 0.0,
                            "cell_coverage": 0.0,
                            "outside_ratio": 1.0,
                            "geometry_rejected": bool(geometry_rejected),
                        }
                    )
                    continue
                chosen = {
                    "mask": fallback_mask,
                    "sam_score": 0.0,
                    "vor_iou": 0.0,
                    "cell_coverage": 0.0,
                    "outside_ratio": 0.0,
                    "status": f"fallback_{fallback_source}",
                }
            elif "status" not in chosen:
                chosen["status"] = "sam_selected"
            chosen["instance_id"] = inst_idx
            chosen["center"] = (y, x)
            selected_regions[inst_idx] = chosen
            per_instance_rows.append(
                {
                    "image_id": stem,
                    "instance_id": inst_idx,
                    "center_y": y,
                    "center_x": x,
                    "status": chosen["status"],
                    "source": chosen["status"],
                    "sam_score": round(float(chosen["sam_score"]), 6),
                    "area": int(np.sum(chosen["mask"])),
                    "contained_center_count": int(chosen.get("contained_center_count", 1)),
                    "other_center_count": int(chosen.get("other_center_count", 0)),
                    "vor_iou": round(float(chosen.get("vor_iou", 0.0)), 6),
                    "cell_coverage": round(float(chosen.get("cell_coverage", 0.0)), 6),
                    "outside_ratio": round(float(chosen.get("outside_ratio", 0.0)), 6),
                    "geometry_rejected": bool(geometry_rejected),
                }
            )

        inst_map, boundary_mask, ignore_mask = resolve_instance_conflicts(
            image_shape=merged_map.shape,
            selected_regions=selected_regions,
            tie_margin=args.tie_margin,
            boundary_dilate=args.boundary_dilate,
            conflict_policy=args.conflict_policy,
        )
        inst_map = relabel_contiguous(inst_map)
        binary_map = (inst_map > 0).astype(np.uint8) * 255
        save_instance_png(sam_inst_dir / f"{stem}.png", inst_map)
        save_png(sam_bin_dir / f"{stem}.png", binary_map)
        save_png(sam_boundary_dir / f"{stem}.png", boundary_mask.astype(np.uint8) * 255)
        save_png(sam_ignore_dir / f"{stem}.png", ignore_mask.astype(np.uint8) * 255)
        tile_images(
            [
                ("image", image_rgb),
                ("merged_points", draw_points(image_rgb, coords)),
                ("sam_binary", np.stack([binary_map] * 3, axis=2)),
                ("sam_instance", colorize_instances(inst_map)),
                ("sam_boundary", np.stack([boundary_mask.astype(np.uint8) * 255] * 3, axis=2)),
                ("sam_ignore", np.stack([ignore_mask.astype(np.uint8) * 255] * 3, axis=2)),
            ],
            overlay_dir / f"{stem}_sam_panel.png",
        )
        per_image_rows.append(
            {
                "image_id": stem,
                "split": row["split"],
                "merged_point_count": len(coords),
                "selected_instance_count": int(inst_map.max()),
                "sam_selected_count": int(
                    sum(1 for item in selected_regions.values() if item["status"] in {"sam_selected", "sam_relaxed"})
                ),
                "sam_relaxed_count": int(sum(1 for item in selected_regions.values() if item["status"] == "sam_relaxed")),
                "fallback_count": int(sum(1 for item in selected_regions.values() if item["status"].startswith("fallback_"))),
                "geometry_rejected_count": int(
                    sum(1 for item in per_instance_rows if item["image_id"] == stem and item.get("geometry_rejected"))
                ),
                "boundary_pixel_count": int(np.sum(boundary_mask)),
                "ignore_pixel_count": int(np.sum(ignore_mask)),
                "binary_foreground_pixels": int(np.sum(inst_map > 0)),
                "predict_seconds": round(float(predict_seconds), 6),
                "positive_prompt_count": int(len(prompt_specs)),
                "negative_prompt_count": int(sum(len(item["negative_xy"]) for item in prompt_specs)),
                "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
                "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
                "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
                "raw_candidate_area_mean": float(np.mean(candidate_areas)) if candidate_areas else 0.0,
                "raw_empty_mask_ratio": float(np.mean([area == 0 for area in candidate_areas])) if candidate_areas else 0.0,
                "raw_tiny_mask_ratio": float(np.mean([0 < area < args.min_area for area in candidate_areas])) if candidate_areas else 0.0,
                "raw_large_mask_ratio": float(np.mean([area > args.max_area for area in candidate_areas])) if candidate_areas else 0.0,
                "raw_score_mean": float(np.mean(candidate_scores)) if candidate_scores else 0.0,
                "image_shape": f"{int(image_rgb.shape[0])}x{int(image_rgb.shape[1])}",
                "wrapper_pre_resize": False,
            }
        )

    if args.backend == "sam2" and isinstance(backend, Sam2Backend) and backend.last_meta:
        for key in ("sam2_version", "output_masks_shape", "output_scores_shape", "per_prompt_candidate_count"):
            if key in backend.last_meta:
                backend_meta[key] = backend.last_meta[key]

    manifest = {
        "dataset": dataset,
        "ratio": ratio,
        "description": description,
        "points_manifest": str(points_manifest),
        "points_manifest_sha256": sha256_file(points_manifest),
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "backend": args.backend,
        "preset": args.preset,
        "backend_meta": backend_meta,
        "candidate_cache": {
            "mode": args.candidate_cache_mode,
            "dir": str(candidate_cache_dir) if candidate_cache_dir is not None else "",
            "context": cache_context if candidate_cache_dir is not None else None,
            "raw_candidates_only": True,
        },
        "config": {
            "negative_strategy": args.negative_strategy,
            "negative_radius": float(args.negative_radius),
            "max_negative_points": int(args.max_negative_points),
            "min_area": int(args.min_area),
            "max_area": int(args.max_area),
            "min_sam_score": float(args.min_sam_score),
            "min_vor_iou": float(args.min_vor_iou),
            "min_cell_coverage": float(args.min_cell_coverage),
            "max_outside_ratio": float(args.max_outside_ratio),
            "candidate_selection_mode": args.candidate_selection_mode,
            "fallback": args.fallback,
            "boundary_dilate": int(args.boundary_dilate),
            "tie_margin": float(args.tie_margin),
            "conflict_policy": args.conflict_policy,
        },
        "records": per_image_rows,
        "paths": {
            "sam_instance_dir": str(sam_inst_dir),
            "sam_binary_dir": str(sam_bin_dir),
            "sam_boundary_dir": str(sam_boundary_dir),
            "sam_ignore_dir": str(sam_ignore_dir),
            "overlay_dir": str(overlay_dir),
            "candidate_overlay_dir": str(candidate_overlay_dir) if candidate_overlay_dir is not None else "",
        },
        "prompt_audit": {
            "multimask_output": True,
            "positive_points_per_prompt": 1,
            "negative_strategy": args.negative_strategy,
            "negative_radius": args.negative_radius,
            "max_negative_points": args.max_negative_points,
            "coordinate_convention": "prompt coordinates are xy; numpy image/mask arrays are row,col (y,x)",
            "wrapper_pre_resize": False,
        },
    }
    save_json(step_root / "sam_region_manifest.json", manifest)
    if cache_manifest_path is not None:
        cache_records = [
            {"image_id": row["image_id"], "path": str((candidate_cache_dir / f"{row['image_id']}.npz").resolve())}
            for row in points_payload["records"]
        ]
        save_json(
            cache_manifest_path,
            {
                "context": cache_context,
                "records": cache_records,
                "selection_applied": False,
                "geometry_applied": False,
                "fallback_applied": False,
            },
        )
    save_csv(step_root / "sam_region_per_instance.csv", per_instance_rows)
    save_csv(step_root / "sam_region_per_image_summary.csv", per_image_rows)
    save_csv(step_root / "sam_region_prompt_summary.csv", per_prompt_rows)
    write_step_report(
        step_root / "sam_region_report.md",
        "SAM Point Regions",
        [
            f"backend={args.backend}, backend_meta={backend_meta}",
            f"fallback_policy={args.fallback}",
            f"processed_images={', '.join(row['image_id'] for row in per_image_rows)}",
        ],
    )
    print(f"Saved SAM region outputs to: {step_root}")


def choose_negative_points(
    center: tuple[int, int],
    coords: list[tuple[int, int]],
    radius: float,
    max_points: int,
    strategy: str,
) -> list[tuple[int, int]]:
    cy, cx = center
    pairs = []
    for y, x in coords:
        if (y, x) == center:
            continue
        dist2 = float((y - cy) * (y - cy) + (x - cx) * (x - cx))
        if strategy == "nearest" or dist2 <= radius * radius:
            pairs.append((dist2, y, x))
    pairs.sort()
    return [(y, x) for _, y, x in pairs[:max_points]]


def region_for_center(inst_map: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    y, x = center
    inst_id = nearest_positive_label(inst_map, y, x, radius=8)
    if inst_id <= 0:
        return np.zeros(inst_map.shape, dtype=bool)
    return inst_map == inst_id


def select_candidate(
    center: tuple[int, int],
    all_centers: list[tuple[int, int]],
    candidates: list[dict[str, Any]],
    vor_region: np.ndarray,
    min_area: int,
    max_area: int,
    min_sam_score: float,
    min_vor_iou: float,
    min_cell_coverage: float,
    max_outside_ratio: float,
    strict_center_exclusion_radius: int,
    selection_mode: str = "geometry_constrained",
) -> dict[str, Any] | None:
    cy, cx = center
    kept = []
    vor_area = int(np.sum(vor_region))
    for item in candidates:
        mask = item["mask"]
        if float(item["sam_score"]) < float(min_sam_score):
            continue
        area = int(np.sum(mask))
        if area < min_area or area > max_area:
            continue
        if not mask[cy, cx]:
            continue
        expanded_mask = binary_dilation(mask, strict_center_exclusion_radius) if strict_center_exclusion_radius > 0 else mask
        contained = [(y, x) for y, x in all_centers if expanded_mask[y, x]]
        other_count = len([pt for pt in contained if pt != center])
        if other_count > 0:
            continue
        if vor_area > 0:
            inter = int(np.sum(mask & vor_region))
            union = int(np.sum(mask | vor_region))
            vor_iou = inter / max(union, 1)
            cell_coverage = inter / max(vor_area, 1)
            outside_ratio = float(np.sum(mask & (~vor_region))) / max(area, 1)
        else:
            vor_iou = 0.0
            cell_coverage = 0.0
            outside_ratio = 1.0
        if selection_mode == "geometry_constrained":
            if vor_iou < min_vor_iou:
                continue
            if cell_coverage < min_cell_coverage:
                continue
            if outside_ratio > max_outside_ratio:
                continue
        kept.append(
            {
                **item,
                "area": area,
                "contained_center_count": len(contained),
                "other_center_count": other_count,
                "vor_iou": float(vor_iou),
                "cell_coverage": float(cell_coverage),
                "outside_ratio": float(outside_ratio),
            }
        )
    if not kept:
        return None
    if selection_mode == "sam_only":
        kept.sort(key=lambda item: (item["sam_score"], -abs(item["area"] - 200)), reverse=True)
    else:
        kept.sort(
            key=lambda item: (
                item["sam_score"] + 0.35 * item["vor_iou"] + 0.2 * item["cell_coverage"] - 0.2 * item["outside_ratio"],
                -item["outside_ratio"],
                -abs(item["area"] - 200),
            ),
            reverse=True,
        )
    return kept[0]


def build_fallback_region(
    fallback: str,
    center: tuple[int, int],
    vor_inst: np.ndarray,
    clu_inst: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if fallback == "ignore":
        return None, "ignore"
    source = vor_inst if fallback == "voronoi" else clu_inst
    inst_id = nearest_positive_label(source, center[0], center[1], radius=8)
    if inst_id <= 0:
        return None, fallback
    return source == inst_id, fallback


def resolve_instance_conflicts(
    image_shape: tuple[int, int],
    selected_regions: dict[int, dict[str, Any]],
    tie_margin: float,
    boundary_dilate: int,
    conflict_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    inst_map = np.zeros((h, w), dtype=np.int32)
    ignore = np.zeros((h, w), dtype=bool)
    boundary = np.zeros((h, w), dtype=bool)
    if not selected_regions:
        return inst_map, boundary, ignore
    stacked = np.zeros((len(selected_regions), h, w), dtype=bool)
    scores = []
    centers = []
    ids = []
    for idx, (inst_id, item) in enumerate(sorted(selected_regions.items())):
        stacked[idx] = item["mask"]
        scores.append(float(item["sam_score"]))
        centers.append(item["center"])
        ids.append(inst_id)
    scores = np.asarray(scores, dtype=np.float32)
    cover_count = np.sum(stacked, axis=0)
    for y, x in np.argwhere(cover_count == 1):
        idx = int(np.argmax(stacked[:, y, x]))
        inst_map[y, x] = int(ids[idx])
    for y, x in np.argwhere(cover_count > 1):
        idxs = np.where(stacked[:, y, x])[0]
        if conflict_policy == "ignore":
            ignore[y, x] = True
            boundary[y, x] = True
            continue
        pixel_scores = []
        for idx in idxs:
            cy, cx = centers[idx]
            dist = math.sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))
            pixel_scores.append(float(scores[idx]) + 0.03 / (1.0 + dist))
        order = np.argsort(np.asarray(pixel_scores))[::-1]
        best_idx = idxs[int(order[0])]
        best_score = pixel_scores[int(order[0])]
        second_score = pixel_scores[int(order[1])] if len(order) > 1 else -1e9
        if best_score - second_score < tie_margin:
            ignore[y, x] = True
            boundary[y, x] = True
            continue
        inst_map[y, x] = int(ids[best_idx])
    boundary |= mask_boundary(inst_map > 0)
    if boundary_dilate > 0:
        boundary = binary_dilation(boundary, boundary_dilate)
    ignore |= boundary & (inst_map > 0)
    return inst_map, boundary, ignore


def draw_points(image_rgb: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    for y, x in coords:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 0, 0), width=2)
    return np.asarray(canvas)


def save_candidate_panel(
    image_rgb: np.ndarray,
    positive_xy: tuple[int, int],
    negative_xy: list[tuple[int, int]],
    candidates: list[dict[str, Any]],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    panels: list[tuple[str, np.ndarray]] = [("image+prompts", draw_prompt_overlay(image_rgb, positive_xy, negative_xy))]
    for idx, candidate in enumerate(candidates[:3]):
        mask = candidate["mask"].astype(bool)
        overlay = image_rgb.copy()
        color = np.zeros_like(overlay)
        color[..., 1] = 255
        overlay = np.where(mask[..., None], (0.55 * overlay + 0.45 * color).astype(np.uint8), overlay)
        overlay = draw_prompt_overlay(overlay, positive_xy, negative_xy)
        panels.append((f"cand{idx}_score{float(candidate['sam_score']):.3f}", overlay))
    tile_images(panels, out_path)


def draw_prompt_overlay(
    image_rgb: np.ndarray,
    positive_xy: tuple[int, int],
    negative_xy: list[tuple[int, int]],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    px, py = positive_xy
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(0, 255, 0), width=3)
    for nx, ny in negative_xy:
        draw.line((nx - 5, ny - 5, nx + 5, ny + 5), fill=(255, 0, 0), width=3)
        draw.line((nx - 5, ny + 5, nx + 5, ny - 5), fill=(255, 0, 0), width=3)
    return np.asarray(canvas)


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "hover_conservative_v1":
        args.negative_radius = 36.0
        args.max_negative_points = 8
        args.negative_strategy = "nearest"
        args.min_area = 24
        args.max_area = 3500
        args.min_sam_score = 0.88
        args.min_vor_iou = 0.10
        args.min_cell_coverage = 0.10
        args.max_outside_ratio = 0.35
        args.strict_center_exclusion_radius = 6
        args.fallback = "ignore"
        args.boundary_dilate = 2
        args.tie_margin = 0.03
        args.conflict_policy = "ignore"
        return
    if args.preset == "hover_balanced_v1":
        args.negative_radius = 42.0
        args.max_negative_points = 6
        args.negative_strategy = "nearest"
        args.min_area = 20
        args.max_area = 4200
        args.min_sam_score = 0.80
        args.min_vor_iou = 0.06
        args.min_cell_coverage = 0.06
        args.max_outside_ratio = 0.55
        args.strict_center_exclusion_radius = 2
        args.fallback = "cluster"
        args.boundary_dilate = 1
        args.tie_margin = 0.02
        args.conflict_policy = "ignore"


if __name__ == "__main__":
    main()
