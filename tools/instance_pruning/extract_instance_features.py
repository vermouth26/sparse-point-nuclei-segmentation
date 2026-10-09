from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.instance_pruning.common import ensure_dir, list_npy_ids, load_np_prob, write_csv_dicts

try:
    from scipy import ndimage as ndi
    from skimage.measure import regionprops
except Exception:  # pragma: no cover - server env normally has scipy/skimage.
    ndi = None
    regionprops = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-instance pruning features from HoVer predictions.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with predicted instance maps (*.npy).")
    parser.add_argument("--np-dir", type=Path, required=True, help="Directory with NP probability maps (*.npy).")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--center-dir", type=Path, default=None, help="Optional center maps directory.")
    parser.add_argument("--neighbor-radius", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = extract_features(
        pred_dir=args.pred_dir,
        np_dir=args.np_dir,
        center_dir=args.center_dir,
        neighbor_radius=args.neighbor_radius,
    )
    write_csv_dicts(args.out_csv, rows)
    print(f"Saved {len(rows)} instance feature rows to: {args.out_csv.resolve()}")


def extract_features(
    *,
    pred_dir: Path,
    np_dir: Path,
    center_dir: Path | None,
    neighbor_radius: float,
) -> list[dict]:
    if not pred_dir.exists() or not np_dir.exists():
        raise SystemExit(f"Missing pred or NP dir: {pred_dir}, {np_dir}")
    image_ids = list_npy_ids(pred_dir)
    if not image_ids:
        raise SystemExit(f"No predicted instance maps found in {pred_dir}")

    rows: list[dict] = []
    for image_id in image_ids:
        inst = np.load(pred_dir / f"{image_id}.npy", allow_pickle=False).astype(np.int32)
        np_prob = load_np_prob(np_dir / f"{image_id}.npy")
        if np_prob.shape != inst.shape:
            raise SystemExit(f"Shape mismatch for {image_id}: inst={inst.shape}, np={np_prob.shape}")
        center_map = load_center_map(center_dir, image_id, inst.shape) if center_dir else None
        rows.extend(extract_image_features(image_id, inst, np_prob, center_map, neighbor_radius))
    return rows


def load_center_map(center_dir: Path | None, image_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    if center_dir is None:
        return None
    candidates = [
        center_dir / f"{image_id}.npy",
        center_dir / f"{image_id}_label_point.npy",
        center_dir / f"{image_id}_label_point.png",
        center_dir / f"{image_id}.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        else:
            arr = np.asarray(Image.open(path).convert("L"))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != shape:
            raise SystemExit(f"Center map shape mismatch for {image_id}: {arr.shape} vs {shape}")
        return arr.astype(np.float32)
    return None


def extract_image_features(
    image_id: str,
    inst: np.ndarray,
    np_prob: np.ndarray,
    center_map: np.ndarray | None,
    neighbor_radius: float,
) -> list[dict]:
    ids = [int(v) for v in np.unique(inst) if int(v) > 0]
    if not ids:
        return []
    centroids: dict[int, tuple[float, float]] = {}
    props_by_id = region_props(inst) if regionprops is not None else {}
    centers_yx = center_coordinates(center_map) if center_map is not None else np.empty((0, 2), dtype=np.float32)

    raw_rows: list[dict] = []
    for inst_id in ids:
        mask = inst == inst_id
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        area = int(ys.size)
        cy = float(ys.mean())
        cx = float(xs.mean())
        centroids[inst_id] = (cy, cx)
        vals = np_prob[mask]
        prop = props_by_id.get(inst_id, {})
        perimeter = float(prop.get("perimeter", estimate_perimeter(mask)))
        compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-6))
        center_stats = center_features(mask, center_map, centers_yx, cy, cx)
        raw_rows.append(
            {
                "image_id": image_id,
                "instance_id": inst_id,
                "area": area,
                "bbox_y0": y0,
                "bbox_x0": x0,
                "bbox_y1": y1,
                "bbox_x1": x1,
                "bbox_area": int((y1 - y0) * (x1 - x0)),
                "centroid_y": cy,
                "centroid_x": cx,
                "np_mean": float(vals.mean()),
                "np_max": float(vals.max()),
                "np_p10": float(np.percentile(vals, 10)),
                "np_p90": float(np.percentile(vals, 90)),
                "solidity": float(prop.get("solidity", 0.0)),
                "eccentricity": float(prop.get("eccentricity", 0.0)),
                "compactness": compactness,
                **center_stats,
            }
        )

    for row in raw_rows:
        inst_id = int(row["instance_id"])
        cy, cx = centroids[inst_id]
        distances = []
        for other_id, (oy, ox) in centroids.items():
            if other_id == inst_id:
                continue
            distances.append(float(math.hypot(cy - oy, cx - ox)))
        row["nearest_instance_distance"] = min(distances) if distances else -1.0
        row["local_neighbor_count"] = int(sum(d <= neighbor_radius for d in distances))
    return raw_rows


def region_props(inst: np.ndarray) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for prop in regionprops(inst):
        out[int(prop.label)] = {
            "solidity": float(getattr(prop, "solidity", 0.0)),
            "eccentricity": float(getattr(prop, "eccentricity", 0.0)),
            "perimeter": float(getattr(prop, "perimeter", 0.0)),
        }
    return out


def estimate_perimeter(mask: np.ndarray) -> float:
    if ndi is None:
        return float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    return float(np.logical_and(mask, ~eroded).sum())


def center_coordinates(center_map: np.ndarray | None) -> np.ndarray:
    if center_map is None:
        return np.empty((0, 2), dtype=np.float32)
    ys, xs = np.nonzero(center_map > 0)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)


def center_features(
    mask: np.ndarray,
    center_map: np.ndarray | None,
    centers_yx: np.ndarray,
    cy: float,
    cx: float,
) -> dict[str, float | int]:
    if center_map is None or centers_yx.size == 0:
        return {
            "num_centers_inside": 0,
            "max_center_score_inside": 0.0,
            "distance_to_nearest_center": -1.0,
            "center_inside_flag": 0,
            "expected_center_count_image": 0,
        }
    inside_vals = center_map[mask]
    num_inside = int(np.count_nonzero(inside_vals > 0))
    max_inside = float(inside_vals.max()) if inside_vals.size else 0.0
    dists = np.sqrt((centers_yx[:, 0] - cy) ** 2 + (centers_yx[:, 1] - cx) ** 2)
    return {
        "num_centers_inside": num_inside,
        "max_center_score_inside": max_inside,
        "distance_to_nearest_center": float(dists.min()) if dists.size else -1.0,
        "center_inside_flag": int(num_inside > 0),
        "expected_center_count_image": int(centers_yx.shape[0]),
    }


if __name__ == "__main__":
    main()
