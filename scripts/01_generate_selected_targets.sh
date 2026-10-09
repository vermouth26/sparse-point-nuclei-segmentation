#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

: "${POINTS_MANIFEST:?Set POINTS_MANIFEST to the frozen TRAIN24 completed-center manifest}"
: "${BASELINE_MANIFEST:?Set BASELINE_MANIFEST to the matching Voronoi/cluster manifest}"
: "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to sam2.1_hiera_base_plus.pt}"
: "${SAM2_CONFIG:?Set SAM2_CONFIG to sam2.1_hiera_b+.yaml}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"

SAM2_PYTHON="${SAM2_PYTHON:-$PYTHON}"
TARGET_ROOT="$OUTPUT_ROOT/selected_c_plus_fallback"
SAM_SCORE="${SAM_SCORE:-0.65}"
MIN_AREA="${MIN_AREA:-20}"
MAX_AREA="${MAX_AREA:-3000}"
MIN_VOR_IOU="${MIN_VOR_IOU:-0.03}"
MIN_CELL_COVERAGE="${MIN_CELL_COVERAGE:-0.03}"
MAX_OUTSIDE_RATIO="${MAX_OUTSIDE_RATIO:-0.60}"

cd "$ROOT"

"$PYTHON" -B experiments/vit_region_prior_sam/03_generate_sam_point_regions.py \
  --points-manifest "$POINTS_MANIFEST" \
  --baseline-manifest "$BASELINE_MANIFEST" \
  --out-dir "$TARGET_ROOT" \
  --backend sam2 \
  --preset none \
  --sam2-python "$SAM2_PYTHON" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-model-config "$SAM2_CONFIG" \
  --sam2-model-variant sam2.1_hiera_base_plus \
  --sam2-device cuda \
  --sam2-confirm-official-general-checkpoint \
  --negative-strategy radius \
  --negative-radius 32 \
  --max-negative-points 4 \
  --candidate-selection-mode sam_only \
  --min-area "$MIN_AREA" \
  --max-area "$MAX_AREA" \
  --min-sam-score "$SAM_SCORE" \
  --min-vor-iou "$MIN_VOR_IOU" \
  --min-cell-coverage "$MIN_CELL_COVERAGE" \
  --max-outside-ratio "$MAX_OUTSIDE_RATIO" \
  --fallback cluster \
  --conflict-policy winner

"$PYTHON" -B experiments/vit_region_prior_sam/05_convert_sam_region_to_hover_targets.py \
  --points-manifest "$POINTS_MANIFEST" \
  --sam-manifest "$TARGET_ROOT/03_sam_regions/sam_region_manifest.json" \
  --out-dir "$TARGET_ROOT" \
  --use-sam-ignore \
  --no-gt-diagnostics

echo "Hover target manifest: $TARGET_ROOT/05_hover_targets/hover_target_manifest.json"
