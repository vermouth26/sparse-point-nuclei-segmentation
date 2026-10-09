#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

: "${VAL_RAW_DIR:?Set VAL_RAW_DIR to frozen VAL6 raw NP/HV predictions}"
: "${DATA_ROOT:?Set DATA_ROOT to the raw data root containing MO/}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new calibration directory}"

cd "$ROOT"

"$PYTHON" -B experiments/monuseg_official_hovernet/official_hovernet_postprocess_sweep_monuseg.py \
  --val-raw-dir "$VAL_RAW_DIR" \
  --selection-only \
  --split "$DATA_ROOT/MO/train_val_test.json" \
  --label-dir "$DATA_ROOT/MO/labels_instance" \
  --out-dir "$OUTPUT_ROOT/postprocess" \
  --matching-radius 15 \
  --preset light

"$PYTHON" -B tools/instance_pruning/extract_instance_features.py \
  --pred-dir "$OUTPUT_ROOT/postprocess/official_postprocess_best_val_instance_maps" \
  --np-dir "$VAL_RAW_DIR/np_prob" \
  --out-csv "$OUTPUT_ROOT/features_val.csv"

"$PYTHON" -B tools/instance_pruning/tune_pruning_on_val.py \
  --pred-dir "$OUTPUT_ROOT/postprocess/official_postprocess_best_val_instance_maps" \
  --features-csv "$OUTPUT_ROOT/features_val.csv" \
  --label-dir "$DATA_ROOT/MO/labels_instance" \
  --baseline-overall "$OUTPUT_ROOT/postprocess/official_postprocess_best_val_overall.json" \
  --out-dir "$OUTPUT_ROOT/pruning" \
  --save-selected-maps

echo "Calibration is complete on VAL6 only. Freeze its selected JSON files before any TEST14 run."

