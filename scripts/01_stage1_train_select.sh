#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MAIN_CONFIG="${MAIN_CONFIG:-$ROOT/configs/ratio010_mainline.json}"
read -r ANNOTATION_SEED STAGE1_SEED < <(
  "$PYTHON" -B "$ROOT/scripts/read_experiment_config.py" --config "$MAIN_CONFIG" \
    --fields sparse_points.annotation_seed stage1.seed
)
STAGE1_DESCRIPTION="_stage1_r0_raw_fixedop_v1_seed${STAGE1_SEED}"

: "${DATA_ROOT:?Set DATA_ROOT to the raw data root containing MO/}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to the prepared-data root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new Stage1 output root}"

RUNTIME_CONFIG="$OUTPUT_ROOT/stage1_ratio010_runtime.yaml"
SOURCE_POINTS="${SOURCE_POINTS:-$DATA_ROOT/MO/ratio0.1_nested_seed${ANNOTATION_SEED}/labels_point/0}"
DESCRIPTION="$STAGE1_DESCRIPTION"
CHECKPOINT_DIR="$OUTPUT_ROOT/MO/ratio0.1${DESCRIPTION}/checkpoints"
SELECTION_DIR="$OUTPUT_ROOT/selection"

cd "$ROOT"

"$PYTHON" -B scripts/materialize_stage1_config.py \
  --template configs/stage1_ratio010.yaml \
  --data-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed "$STAGE1_SEED" \
  --description "$DESCRIPTION" \
  --out "$RUNTIME_CONFIG"

"$PYTHON" -B experiments/ablation_monuseg_r0_raw_fixedop/prepare_stage1_inputs.py \
  --config "$RUNTIME_CONFIG" \
  --source-points "$SOURCE_POINTS"

"$PYTHON" -B experiments/ablation_monuseg_r0_raw_fixedop/train_r0.py \
  --config "$RUNTIME_CONFIG" \
  --workers 8 \
  --curriculum-tile-batch-size 16

"$PYTHON" -B experiments/stage1_uniform_p084_protocol/select_stage1_protocol.py \
  --phase epoch \
  --project-root "$ROOT" \
  --raw-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --dataset MO \
  --ratio 0.1 \
  --description "$DESCRIPTION" \
  --stage1-config "$RUNTIME_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --no-precision-floor \
  --epoch-threshold 0.65 \
  --epoch-min-area 12 \
  --epoch-max-area 150 \
  --merge-radius 15 \
  --matching-radius 11 \
  --device cuda \
  --out-dir "$SELECTION_DIR"

echo "Stage1 selection: $SELECTION_DIR/selected_epoch.json"
echo "Export TRAIN24 and VAL6 completed centers with 01_prepare_stage1_points.py as documented in README.md."
