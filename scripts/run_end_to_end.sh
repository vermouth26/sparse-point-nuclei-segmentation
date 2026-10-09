#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SAM2_PYTHON="${SAM2_PYTHON:-$PYTHON}"
HOVERNET_ROOT="${HOVERNET_ROOT:-$ROOT/third_party/hover_net}"
MAIN_CONFIG="${MAIN_CONFIG:-$ROOT/configs/ratio010_mainline.json}"
read -r ANNOTATION_SEED STAGE1_SEED HOVERNET_SEED < <(
  "$PYTHON" -B "$ROOT/scripts/read_experiment_config.py" --config "$MAIN_CONFIG" \
    --fields sparse_points.annotation_seed stage1.seed hovernet.seed
)
STAGE1_DESCRIPTION="_stage1_r0_raw_fixedop_v1_seed${STAGE1_SEED}"
RUN_TEST="${RUN_TEST:-0}"
RESUME_AFTER_STAGE1="${RESUME_AFTER_STAGE1:-0}"
RESUME_FROM_STAGE2="${RESUME_FROM_STAGE2:-0}"
RESUME_FROM_HOVER_STAGE2="${RESUME_FROM_HOVER_STAGE2:-0}"

: "${DATA_ROOT:?Set DATA_ROOT to the raw data root containing MO/}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to a writable processed-data root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new, empty ratio010 run root}"
: "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to sam2.1_hiera_base_plus.pt}"
: "${SAM2_CONFIG:?Set SAM2_CONFIG to sam2.1_hiera_b+.yaml}"
: "${HOVERNET_PRETRAIN:?Set HOVERNET_PRETRAIN to the official Preact-ResNet50 ImageNet weights}"

if [[ -e "$OUTPUT_ROOT" && "$RESUME_AFTER_STAGE1" != "1" && "$RESUME_FROM_STAGE2" != "1" && "$RESUME_FROM_HOVER_STAGE2" != "1" ]]; then
  echo "[STOP] OUTPUT_ROOT already exists; use a new path: $OUTPUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
cd "$ROOT"

export OUTPUT_ROOT_STAGE1="$OUTPUT_ROOT/stage1"
if [[ "$RESUME_FROM_HOVER_STAGE2" == "1" ]]; then
  echo "[RESUME] Reuse frozen Stage1, centers, Stage2 OP and TRAIN24 pseudo targets"
  test -f "$OUTPUT_ROOT_STAGE1/selection/selected_epoch.json"
elif [[ "$RESUME_FROM_STAGE2" == "1" ]]; then
  echo "[RESUME] Reuse frozen Stage1 selection, completed centers and support regions"
  test -f "$OUTPUT_ROOT_STAGE1/selection/selected_epoch.json"
elif [[ "$RESUME_AFTER_STAGE1" != "1" ]]; then
  echo "[0/12] Static package check"
  "$PYTHON" -B scripts/check_package.py

  echo "[1/12] Build frozen 10% annotation set with seed $ANNOTATION_SEED (TRAIN24+VAL6 only)"
  POINT_SOURCE="$OUTPUT_ROOT/annotations/ratio0.1_nested_seed${ANNOTATION_SEED}/labels_point/0"
  "$PYTHON" -B scripts/prepare_sparse_points.py \
    --data-root "$DATA_ROOT" --seed "$ANNOTATION_SEED" --ratio 0.10 --out-dir "$POINT_SOURCE"

  echo "[2/12] Stage1 seed $STAGE1_SEED training and max-VAL-F1 epoch selection"
  export SOURCE_POINTS="$POINT_SOURCE"
  OUTPUT_ROOT="$OUTPUT_ROOT_STAGE1" bash scripts/01_stage1_train_select.sh
else
  echo "[RESUME] Reuse completed Stage1 training and run only max-VAL-F1 selection"
  test -d "$OUTPUT_ROOT_STAGE1/MO/ratio0.1${STAGE1_DESCRIPTION}/checkpoints"
  mkdir -p "$OUTPUT_ROOT_STAGE1/selection"
  "$PYTHON" -B experiments/stage1_uniform_p084_protocol/select_stage1_protocol.py \
    --phase epoch --project-root "$ROOT" --raw-root "$DATA_ROOT" \
    --processed-root "$PROCESSED_ROOT" --dataset MO --ratio 0.1 \
    --description "$STAGE1_DESCRIPTION" \
    --stage1-config "$OUTPUT_ROOT_STAGE1/stage1_ratio010_runtime.yaml" \
    --checkpoint-dir "$OUTPUT_ROOT_STAGE1/MO/ratio0.1${STAGE1_DESCRIPTION}/checkpoints" \
    --no-precision-floor --epoch-threshold 0.65 --epoch-min-area 12 \
    --epoch-max-area 150 --merge-radius 15 --matching-radius 11 \
    --device cuda --out-dir "$OUTPUT_ROOT_STAGE1/selection"
fi
STAGE1_CONFIG="$OUTPUT_ROOT_STAGE1/stage1_ratio010_runtime.yaml"
STAGE1_SELECTED="$OUTPUT_ROOT_STAGE1/selection/selected_epoch.json"
STAGE1_CHECKPOINT=$("$PYTHON" - "$STAGE1_SELECTED" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("best", {}).get("checkpoint") or d.get("checkpoint") or d.get("checkpoint_path"))
PY
)
test -f "$STAGE1_CHECKPOINT"

TRAIN_CENTERS="$OUTPUT_ROOT/train24_centers"
VAL_CENTERS="$OUTPUT_ROOT/val6_centers"
if [[ "$RESUME_FROM_STAGE2" != "1" && "$RESUME_FROM_HOVER_STAGE2" != "1" ]]; then
  echo "[3/12] Export TRAIN24 and VAL6 completed centers"
  COMMON=(--dataset MO --ratio 0.1 --description "$STAGE1_DESCRIPTION" \
    --raw-root "$DATA_ROOT" --processed-root "$PROCESSED_ROOT" \
    --stage1-config "$STAGE1_CONFIG" --checkpoint "$STAGE1_CHECKPOINT" \
    --device cuda --threshold 0.65 --min-area 12 --max-area 150 --merge-radius 15)
  "$PYTHON" -B experiments/vit_region_prior_sam/01_prepare_stage1_points.py \
    "${COMMON[@]}" --split train --max-images 24 --no-gt-diagnostics --out-dir "$TRAIN_CENTERS"
  "$PYTHON" -B experiments/vit_region_prior_sam/01_prepare_stage1_points.py \
    "${COMMON[@]}" --split val --max-images 6 --out-dir "$VAL_CENTERS"

  echo "[4/12] Generate point-derived support regions"
  "$PYTHON" -B experiments/vit_region_prior_sam/02_generate_bonus_region_baseline.py \
    --points-manifest "$TRAIN_CENTERS/01_points/prepare_stage1_points_manifest.json" \
    --out-dir "$TRAIN_CENTERS" --no-gt-diagnostics
  "$PYTHON" -B experiments/vit_region_prior_sam/02_generate_bonus_region_baseline.py \
    --points-manifest "$VAL_CENTERS/01_points/prepare_stage1_points_manifest.json" \
    --out-dir "$VAL_CENTERS"
else
  test -f "$TRAIN_CENTERS/01_points/prepare_stage1_points_manifest.json"
  test -f "$TRAIN_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json"
  test -f "$VAL_CENTERS/01_points/prepare_stage1_points_manifest.json"
  test -f "$VAL_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json"
fi

STAGE2_TABLE="$OUTPUT_ROOT/stage2_val/val6_candidate_metrics.csv"
STAGE2_SELECTED="$OUTPUT_ROOT/stage2_val/selected_stage2_config.json"
STAGE2_TRAIN_ROOT="$OUTPUT_ROOT/stage2_train"
HOVER_TARGET_MANIFEST="$STAGE2_TRAIN_ROOT/selected_c_plus_fallback/05_hover_targets/hover_target_manifest.json"
if [[ "$RESUME_FROM_HOVER_STAGE2" != "1" ]]; then
  echo "[5/12] VAL6-only 12-candidate C+fallback operating-point selection"
  TRIALS="$OUTPUT_ROOT/stage2_val/trials"
  CACHE="$OUTPUT_ROOT/stage2_val/candidate_cache"
  for score in 0.65 0.70 0.75; do
    for min_area in 10 20; do
      for max_area in 3000 5000; do
        tag="score${score/./p}_min${min_area}_max${max_area}"
        trial="$TRIALS/$tag"
        echo "  [$tag]"
        "$PYTHON" -B experiments/vit_region_prior_sam/03_generate_sam_point_regions.py \
          --points-manifest "$VAL_CENTERS/01_points/prepare_stage1_points_manifest.json" \
          --baseline-manifest "$VAL_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json" \
          --backend sam2 --sam2-python "$SAM2_PYTHON" \
          --sam2-checkpoint "$SAM2_CHECKPOINT" --sam2-model-config "$SAM2_CONFIG" \
          --sam2-model-variant sam2.1_hiera_base_plus --sam2-device cuda \
          --sam2-confirm-official-general-checkpoint --negative-strategy radius \
          --negative-radius 32 --max-negative-points 4 --candidate-selection-mode sam_only \
          --fallback cluster --min-sam-score "$score" --min-area "$min_area" --max-area "$max_area" \
          --max-outside-ratio 0.60 --min-vor-iou 0.03 --min-cell-coverage 0.03 \
          --conflict-policy winner --candidate-cache-dir "$CACHE" --candidate-cache-mode auto \
          --out-dir "$trial"
        "$PYTHON" -B experiments/vit_region_prior_sam/04_eval_pseudo_region_quality.py \
          --points-manifest "$VAL_CENTERS/01_points/prepare_stage1_points_manifest.json" \
          --baseline-manifest "$VAL_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json" \
          --sam-manifest "$trial/03_sam_regions/sam_region_manifest.json" \
          --sam-label sam2 --out-dir "$trial"
      done
    done
  done
  "$PYTHON" -B scripts/collect_stage2_metrics.py --trials-root "$TRIALS" --out "$STAGE2_TABLE"
  "$PYTHON" -B scripts/select_stage2_mainline.py --table "$STAGE2_TABLE" --out "$STAGE2_SELECTED"
  read -r SAM_SCORE MIN_AREA MAX_AREA < <("$PYTHON" - "$STAGE2_SELECTED" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))["selected"]
print(r["score_threshold"], r["min_area"], r["max_area"])
PY
  )

  echo "[6/12] Generate selected TRAIN24 C+fallback pseudo instances"
  export POINTS_MANIFEST="$TRAIN_CENTERS/01_points/prepare_stage1_points_manifest.json"
  export BASELINE_MANIFEST="$TRAIN_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json"
  export SAM_SCORE MIN_AREA MAX_AREA
  OUTPUT_ROOT="$STAGE2_TRAIN_ROOT" bash scripts/01_generate_selected_targets.sh
else
  test -f "$STAGE2_SELECTED"
  test -f "$HOVER_TARGET_MANIFEST"
fi

echo "[7/12] Train official HoVerNet with seed $HOVERNET_SEED"
HOVER_OUT="$OUTPUT_ROOT/hovernet"
export HOVER_TARGET_MANIFEST
if [[ "$RESUME_FROM_HOVER_STAGE2" == "1" ]]; then
  PATCH_DIR="$HOVER_OUT/hovernet_patches"
  STAGE1_HOVER="$HOVER_OUT/official_hovernet/stage1_freeze_encoder/checkpoints/hovernet_epoch_030.pt"
  STAGE2_HOVER="$HOVER_OUT/official_hovernet/stage2_finetune_all"
  test -f "$STAGE1_HOVER"
  test -d "$PATCH_DIR"
  if [[ -e "$STAGE2_HOVER" ]]; then
    ARCHIVE="${STAGE2_HOVER}_interrupted_reboot_epoch031"
    [[ ! -e "$ARCHIVE" ]] || { echo "[STOP] Archive already exists: $ARCHIVE" >&2; exit 4; }
    mv "$STAGE2_HOVER" "$ARCHIVE"
    echo "Archived incomplete Stage2 directory: $ARCHIVE"
  fi
  "$PYTHON" -B experiments/official_hovernet/train_official_hovernet_patch_fullsup.py \
    --hovernet-root "$HOVERNET_ROOT" --patch-dir "$PATCH_DIR" --out-dir "$STAGE2_HOVER" \
    --mode original --device cuda --epochs 80 --batch-size 4 --workers 4 --lr 1e-4 \
    --seed "$HOVERNET_SEED" --lambda-hv-mse 1.0 --lambda-hv-msge 1.0 --save-every 5 \
    --resume "$STAGE1_HOVER"
else
  OUTPUT_ROOT="$HOVER_OUT" bash scripts/02_train_hovernet.sh
fi

echo "[8/12] Evaluate every HoVerNet checkpoint on VAL6 and select"
CKPT_DIR="$HOVER_OUT/official_hovernet/stage2_finetune_all/checkpoints"
VAL_EVAL="$OUTPUT_ROOT/hovernet_val"
for checkpoint in "$CKPT_DIR"/hovernet_epoch_*.pt; do
  name=$(basename "$checkpoint" .pt)
  "$PYTHON" -B experiments/monuseg_official_hovernet/eval_official_hovernet_monuseg.py \
    --hovernet-root "$HOVERNET_ROOT" --checkpoint "$checkpoint" \
    --split "$DATA_ROOT/MO/train_val_test.json" --subset val \
    --images "$DATA_ROOT/MO/images" --labels "$DATA_ROOT/MO/labels_instance" \
    --out-dir "$VAL_EVAL/$name" --mode original --device cuda --batch-size 8 --postprocess official
done
HOVER_SELECTED="$OUTPUT_ROOT/hovernet_selected_checkpoint.json"
"$PYTHON" -B scripts/select_hovernet_checkpoint.py \
  --metrics-root "$VAL_EVAL" --checkpoint-dir "$CKPT_DIR" --out "$HOVER_SELECTED"
SELECTED_CHECKPOINT=$("$PYTHON" -c "import json; print(json.load(open('$HOVER_SELECTED'))['checkpoint_path'])")
SELECTED_NAME=$("$PYTHON" -c "import json; print(json.load(open('$HOVER_SELECTED'))['checkpoint_name'])")
VAL_RAW="$VAL_EVAL/$SELECTED_NAME"

echo "[9/12] VAL6-only postprocess and pruning calibration"
CALIBRATION="$OUTPUT_ROOT/calibration"
export VAL_RAW_DIR="$VAL_RAW"
OUTPUT_ROOT="$CALIBRATION" bash scripts/03_calibrate_val.sh

echo "[10/12] Freeze pre-test provenance"
"$PYTHON" - "$OUTPUT_ROOT/PRETEST_COMPLETE.json" "$STAGE1_SELECTED" "$STAGE2_SELECTED" "$HOVER_SELECTED" "$CALIBRATION" "$ANNOTATION_SEED" "$STAGE1_SEED" "$HOVERNET_SEED" <<'PY'
import json, sys
from pathlib import Path
out, s1, s2, hover, cal = map(Path, sys.argv[1:6])
annotation_seed, stage1_seed, hovernet_seed = map(int, sys.argv[6:9])
payload={"status":"complete","annotation_seed":annotation_seed,"stage1_seed":stage1_seed,"hovernet_seed":hovernet_seed,"stage1_selection":str(s1.resolve()),
"stage2_selection":str(s2.resolve()),"hover_selection":str(hover.resolve()),
"postprocess_selection":str((cal/'postprocess/selected_postprocess_config.json').resolve()),
"pruning_selection":str((cal/'pruning/selected_pruning_config.json').resolve()),
"selection_split":"VAL6 only","test_used":False}
out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
PY

if [[ "$RUN_TEST" != "1" ]]; then
  echo "[DONE] Configured ratio010 run completed through VAL-only freeze. TEST14 was not read."
  echo "Rerun is intentionally blocked by the existing OUTPUT_ROOT. Inspect PRETEST_COMPLETE.json first."
  exit 0
fi

echo "[11/12] One frozen TEST14 evaluation"
TEST_RAW="$OUTPUT_ROOT/test14_raw"
"$PYTHON" -B experiments/monuseg_official_hovernet/eval_official_hovernet_monuseg.py \
  --hovernet-root "$HOVERNET_ROOT" --checkpoint "$SELECTED_CHECKPOINT" \
  --split "$DATA_ROOT/MO/train_val_test.json" --subset test \
  --images "$DATA_ROOT/MO/images" --labels "$DATA_ROOT/MO/labels_instance" \
  --out-dir "$TEST_RAW" --mode original --device cuda --batch-size 8 --postprocess official

TEST_PP="$OUTPUT_ROOT/test14_fixed/postprocess"
"$PYTHON" -B scripts/apply_fixed_postprocess.py \
  --raw-dir "$TEST_RAW" --split "$DATA_ROOT/MO/train_val_test.json" --subset test \
  --label-dir "$DATA_ROOT/MO/labels_instance" \
  --selected-config "$CALIBRATION/postprocess/selected_postprocess_config.json" --out-dir "$TEST_PP"
"$PYTHON" -B tools/instance_pruning/extract_instance_features.py \
  --pred-dir "$TEST_PP/instance_maps" --np-dir "$TEST_RAW/np_prob" \
  --out-csv "$OUTPUT_ROOT/test14_fixed/features_test.csv"
"$PYTHON" -B tools/instance_pruning/apply_pruning_to_test.py \
  --pred-dir "$TEST_PP/instance_maps" --np-dir "$TEST_RAW/np_prob" \
  --features-csv "$OUTPUT_ROOT/test14_fixed/features_test.csv" \
  --label-dir "$DATA_ROOT/MO/labels_instance" --image-dir "$DATA_ROOT/MO/images" \
  --baseline-overall "$TEST_PP/overall_metrics.json" --baseline-per-image "$TEST_PP/per_image_metrics.csv" \
  --selected-config "$CALIBRATION/pruning/selected_pruning_config.json" \
  --out-dir "$OUTPUT_ROOT/test14_fixed/postprocess_pruning"

echo "[12/12] Fixed TEST14 result"
cat "$OUTPUT_ROOT/test14_fixed/postprocess_pruning/overall_metrics_pruned.json"
echo "[DONE] Configured ratio010 end-to-end run complete. Do not use TEST14 to revise any setting."
