#!/usr/bin/env bash
set -euo pipefail

# Reduced end-to-end check for the ratio010 workflow.
# It calls the production training and inference entrypoints with small budgets.
# TEST14 is excluded, and the resulting metrics are not experimental results.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SAM2_PYTHON="${SAM2_PYTHON:-$PYTHON}"
HOVERNET_ROOT="${HOVERNET_ROOT:-$ROOT/third_party/hover_net}"
MAIN_CONFIG="${MAIN_CONFIG:-$ROOT/configs/ratio010_mainline.json}"

: "${DATA_ROOT:?Set DATA_ROOT to the raw data root containing MO/}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to a writable processed-data root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new, empty smoke directory}"
: "${SAM2_CHECKPOINT:?Set SAM2_CHECKPOINT to sam2.1_hiera_base_plus.pt}"
: "${SAM2_CONFIG:?Set SAM2_CONFIG to sam2.1_hiera_b+.yaml}"
: "${HOVERNET_PRETRAIN:?Set HOVERNET_PRETRAIN to the official Preact-ResNet50 ImageNet weights}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "[STOP] OUTPUT_ROOT already exists; choose a new smoke path: $OUTPUT_ROOT" >&2
  exit 3
fi

read -r ANNOTATION_SEED STAGE1_SEED HOVERNET_SEED < <(
  "$PYTHON" -B "$ROOT/scripts/read_experiment_config.py" --config "$MAIN_CONFIG" \
    --fields sparse_points.annotation_seed stage1.seed hovernet.seed
)

SMOKE_ID="${SMOKE_ID:-$(basename "$OUTPUT_ROOT")}"
SMOKE_ID="${SMOKE_ID//[^a-zA-Z0-9_]/_}"
DESCRIPTION="_stage1_r0_full_smoke_${SMOKE_ID}_seed${STAGE1_SEED}"

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
cd "$ROOT"

echo "[0/12] Static package and input checks"
"$PYTHON" -B scripts/check_package.py
for path in \
  "$DATA_ROOT/MO/train_val_test.json" \
  "$SAM2_CHECKPOINT" \
  "$SAM2_CONFIG" \
  "$HOVERNET_PRETRAIN"; do
  test -f "$path" || { echo "[MISSING] $path" >&2; exit 4; }
done
test -d "$DATA_ROOT/MO/images"
test -d "$DATA_ROOT/MO/labels_instance"

echo "[1/12] Generate configured sparse points (TRAIN24+VAL6 only; TEST14 untouched)"
POINT_SOURCE="$OUTPUT_ROOT/annotations/ratio0.1_nested_seed${ANNOTATION_SEED}/labels_point/0"
"$PYTHON" -B scripts/prepare_sparse_points.py \
  --data-root "$DATA_ROOT" --seed "$ANNOTATION_SEED" --ratio 0.10 --out-dir "$POINT_SOURCE"

echo "[2/12] Run two Stage1 epochs with curriculum refresh and max-VAL-F1 selection"
STAGE1_ROOT="$OUTPUT_ROOT/stage1"
STAGE1_CONFIG="$STAGE1_ROOT/stage1_ratio010_smoke.yaml"
STAGE1_SELECTION="$STAGE1_ROOT/selection"
CHECKPOINT_DIR="$STAGE1_ROOT/MO/ratio0.1${DESCRIPTION}/checkpoints"

"$PYTHON" -B scripts/materialize_stage1_config.py \
  --template configs/stage1_ratio010.yaml \
  --data-root "$DATA_ROOT" --processed-root "$PROCESSED_ROOT" \
  --output-root "$STAGE1_ROOT" --seed "$STAGE1_SEED" \
  --description "$DESCRIPTION" --smoke --out "$STAGE1_CONFIG"

"$PYTHON" -B experiments/ablation_monuseg_r0_raw_fixedop/prepare_stage1_inputs.py \
  --config "$STAGE1_CONFIG" --source-points "$POINT_SOURCE"

"$PYTHON" -B experiments/ablation_monuseg_r0_raw_fixedop/train_r0.py \
  --config "$STAGE1_CONFIG" --workers 0 --curriculum-tile-batch-size 4

"$PYTHON" -B experiments/stage1_uniform_p084_protocol/select_stage1_protocol.py \
  --phase epoch --project-root "$ROOT" --raw-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" --dataset MO --ratio 0.1 \
  --description "$DESCRIPTION" --stage1-config "$STAGE1_CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" --no-precision-floor \
  --epoch-threshold 0.65 --epoch-min-area 12 --epoch-max-area 150 \
  --merge-radius 15 --matching-radius 11 --device cuda --out-dir "$STAGE1_SELECTION"

STAGE1_SELECTED="$STAGE1_SELECTION/selected_epoch.json"
STAGE1_CHECKPOINT=$("$PYTHON" - "$STAGE1_SELECTED" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("best", {}).get("checkpoint") or d.get("checkpoint") or d.get("checkpoint_path"))
PY
)
test -f "$STAGE1_CHECKPOINT"

echo "[3/12] Export completed centers for Train4 and Val2"
TRAIN_CENTERS="$OUTPUT_ROOT/train4_centers"
VAL_CENTERS="$OUTPUT_ROOT/val2_centers"
COMMON=(--dataset MO --ratio 0.1 --description "$DESCRIPTION" \
  --raw-root "$DATA_ROOT" --processed-root "$PROCESSED_ROOT" \
  --stage1-config "$STAGE1_CONFIG" --checkpoint "$STAGE1_CHECKPOINT" \
  --device cuda --threshold 0.65 --min-area 12 --max-area 150 --merge-radius 15)
"$PYTHON" -B experiments/vit_region_prior_sam/01_prepare_stage1_points.py \
  "${COMMON[@]}" --split train --max-images 4 --no-gt-diagnostics --out-dir "$TRAIN_CENTERS"
"$PYTHON" -B experiments/vit_region_prior_sam/01_prepare_stage1_points.py \
  "${COMMON[@]}" --split val --max-images 2 --out-dir "$VAL_CENTERS"

echo "[4/12] Generate point-derived support regions and a Train4/Val2 smoke split"
"$PYTHON" -B experiments/vit_region_prior_sam/02_generate_bonus_region_baseline.py \
  --points-manifest "$TRAIN_CENTERS/01_points/prepare_stage1_points_manifest.json" \
  --out-dir "$TRAIN_CENTERS" --no-gt-diagnostics
"$PYTHON" -B experiments/vit_region_prior_sam/02_generate_bonus_region_baseline.py \
  --points-manifest "$VAL_CENTERS/01_points/prepare_stage1_points_manifest.json" \
  --out-dir "$VAL_CENTERS"

SMOKE_SPLIT="$OUTPUT_ROOT/smoke_split_train4_val2_test0.json"
"$PYTHON" - "$DATA_ROOT/MO/train_val_test.json" "$SMOKE_SPLIT" <<'PY'
import json, sys
from pathlib import Path
source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "train": list(source["train"][:4]),
    "val": list(source["val"][:2]),
    "test": [],
    "smoke_only": True,
    "scientific_result": False,
    "test_images_read": False,
    "test_gt_read": False,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "[5/12] Run the real 12-candidate C+fallback OP selector on Val2"
TRIALS="$OUTPUT_ROOT/stage2_val2/trials"
CACHE="$OUTPUT_ROOT/stage2_val2/candidate_cache"
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

STAGE2_TABLE="$OUTPUT_ROOT/stage2_val2/candidate_metrics.csv"
STAGE2_SELECTED="$OUTPUT_ROOT/stage2_val2/selected_stage2_config.json"
"$PYTHON" -B scripts/collect_stage2_metrics.py --trials-root "$TRIALS" --out "$STAGE2_TABLE"
"$PYTHON" -B scripts/select_stage2_mainline.py \
  --table "$STAGE2_TABLE" --selection-split "smoke Val2 only" --out "$STAGE2_SELECTED"
read -r SAM_SCORE MIN_AREA MAX_AREA < <("$PYTHON" - "$STAGE2_SELECTED" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["selected"]
print(r["score_threshold"], r["min_area"], r["max_area"])
PY
)

echo "[6/12] Generate selected Train4 C+fallback maps and NP/HV targets"
STAGE2_TRAIN="$OUTPUT_ROOT/stage2_train4"
export POINTS_MANIFEST="$TRAIN_CENTERS/01_points/prepare_stage1_points_manifest.json"
export BASELINE_MANIFEST="$TRAIN_CENTERS/02_bonus_region_baseline/bonus_region_baseline_manifest.json"
export SAM_SCORE MIN_AREA MAX_AREA SAM2_CHECKPOINT SAM2_CONFIG SAM2_PYTHON PYTHON
OUTPUT_ROOT="$STAGE2_TRAIN" bash scripts/01_generate_selected_targets.sh
HOVER_TARGET_MANIFEST="$STAGE2_TRAIN/selected_c_plus_fallback/05_hover_targets/hover_target_manifest.json"
test -f "$HOVER_TARGET_MANIFEST"

echo "[7/12] Extract Train4 patches and run short 10+20 epoch HoVerNet phases"
HOVER_OUT="$OUTPUT_ROOT/hovernet_smoke"
PATCH_DIR="$HOVER_OUT/hovernet_patches"
HOVER_STAGE1="$HOVER_OUT/official_hovernet/stage1_freeze_encoder"
HOVER_STAGE2="$HOVER_OUT/official_hovernet/stage2_finetune_all"
"$PYTHON" -B experiments/monuseg_official_hovernet/generate_weak_hovernet_patches_from_manifest.py \
  --hovernet-root "$HOVERNET_ROOT" --hover-target-manifest "$HOVER_TARGET_MANIFEST" \
  --split "$SMOKE_SPLIT" --subset train --images "$DATA_ROOT/MO/images" \
  --out-dir "$PATCH_DIR" --win-size 540 --step-size 164 --extract-type mirror --expected-count 4

"$PYTHON" -B experiments/official_hovernet/train_official_hovernet_patch_fullsup.py \
  --hovernet-root "$HOVERNET_ROOT" --patch-dir "$PATCH_DIR" --out-dir "$HOVER_STAGE1" \
  --mode original --device cuda --freeze-encoder --epochs 10 --batch-size 4 --workers 0 \
  --lr 1e-4 --seed "$HOVERNET_SEED" --lambda-hv-mse 1.0 --lambda-hv-msge 1.0 \
  --save-every 10 --imagenet-pretrained "$HOVERNET_PRETRAIN"

"$PYTHON" -B experiments/official_hovernet/train_official_hovernet_patch_fullsup.py \
  --hovernet-root "$HOVERNET_ROOT" --patch-dir "$PATCH_DIR" --out-dir "$HOVER_STAGE2" \
  --mode original --device cuda --epochs 30 --batch-size 4 --workers 0 \
  --lr 1e-4 --seed "$HOVERNET_SEED" --lambda-hv-mse 1.0 --lambda-hv-msge 1.0 \
  --save-every 10 --resume "$HOVER_STAGE1/checkpoints/hovernet_epoch_010.pt"

echo "[8/12] Infer Val2 and run the HoVerNet checkpoint selector"
VAL_EVAL="$OUTPUT_ROOT/hovernet_val2"
CKPT_DIR="$HOVER_STAGE2/checkpoints"
for checkpoint in "$CKPT_DIR"/hovernet_epoch_*.pt; do
  name=$(basename "$checkpoint" .pt)
  "$PYTHON" -B experiments/monuseg_official_hovernet/eval_official_hovernet_monuseg.py \
    --hovernet-root "$HOVERNET_ROOT" --checkpoint "$checkpoint" \
    --split "$SMOKE_SPLIT" --subset val --images "$DATA_ROOT/MO/images" \
    --labels "$DATA_ROOT/MO/labels_instance" --out-dir "$VAL_EVAL/$name" \
    --mode original --device cuda --batch-size 2 --postprocess official
done
HOVER_SELECTED="$OUTPUT_ROOT/hovernet_selected_checkpoint.json"
"$PYTHON" -B scripts/select_hovernet_checkpoint.py \
  --metrics-root "$VAL_EVAL" --checkpoint-dir "$CKPT_DIR" --out "$HOVER_SELECTED"
SELECTED_NAME=$("$PYTHON" -c "import json; print(json.load(open('$HOVER_SELECTED'))['checkpoint_name'])")
VAL_RAW="$VAL_EVAL/$SELECTED_NAME"

echo "[9/12] Run a three-candidate postprocess smoke calibration on Val2"
CALIBRATION="$OUTPUT_ROOT/calibration_smoke"
"$PYTHON" -B experiments/monuseg_official_hovernet/official_hovernet_postprocess_sweep_monuseg.py \
  --val-raw-dir "$VAL_RAW" --selection-only --split "$SMOKE_SPLIT" \
  --label-dir "$DATA_ROOT/MO/labels_instance" --out-dir "$CALIBRATION/postprocess" \
  --matching-radius 15 --preset smoke

echo "[10/12] Run a two-candidate pruning smoke calibration on Val2"
"$PYTHON" - "$CALIBRATION/postprocess/official_postprocess_best_val_overall.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
if int(metrics.get("sum_pred_instances", 0)) <= 0:
    raise SystemExit(
        "[SMOKE UNDERFIT] HoVerNet produced zero Val2 instances even after the permissive smoke "
        "postprocess candidate. Pruning cannot be exercised; this is a smoke-budget failure, "
        "not a formal-method result."
    )
print(f"[SMOKE] postprocessed Val2 instances={int(metrics['sum_pred_instances'])}")
PY
"$PYTHON" -B tools/instance_pruning/extract_instance_features.py \
  --pred-dir "$CALIBRATION/postprocess/official_postprocess_best_val_instance_maps" \
  --np-dir "$VAL_RAW/np_prob" --out-csv "$CALIBRATION/features_val.csv"
"$PYTHON" -B tools/instance_pruning/tune_pruning_on_val.py \
  --pred-dir "$CALIBRATION/postprocess/official_postprocess_best_val_instance_maps" \
  --features-csv "$CALIBRATION/features_val.csv" \
  --label-dir "$DATA_ROOT/MO/labels_instance" \
  --baseline-overall "$CALIBRATION/postprocess/official_postprocess_best_val_overall.json" \
  --out-dir "$CALIBRATION/pruning" --modes area_only,np_only \
  --min-areas 10 --np-mean-thresholds 0.50 --np-p10-thresholds 0.10 \
  --count-upper-factors 1.10 --save-selected-maps

echo "[11/12] Audit required smoke artifacts"
for path in \
  "$STAGE1_SELECTED" \
  "$STAGE2_SELECTED" \
  "$HOVER_TARGET_MANIFEST" \
  "$HOVER_SELECTED" \
  "$CALIBRATION/postprocess/selected_postprocess_config.json" \
  "$CALIBRATION/pruning/selected_pruning_config.json"; do
  test -s "$path" || { echo "[MISSING OR EMPTY] $path" >&2; exit 5; }
done

echo "[12/12] Write completion provenance (no Test14)"
"$PYTHON" - "$OUTPUT_ROOT/FULL_PIPELINE_SMOKE_COMPLETE.json" \
  "$STAGE1_SELECTED" "$STAGE2_SELECTED" "$HOVER_SELECTED" "$CALIBRATION" \
  "$ANNOTATION_SEED" "$STAGE1_SEED" "$HOVERNET_SEED" <<'PY'
import json, sys
from pathlib import Path
out, stage1, stage2, hover, calibration = map(Path, sys.argv[1:6])
annotation_seed, stage1_seed, hovernet_seed = map(int, sys.argv[6:9])
payload = {
    "status": "complete",
    "purpose": "full pipeline integration smoke only",
    "scientific_result": False,
    "annotation_seed": annotation_seed,
    "stage1_seed": stage1_seed,
    "hovernet_seed": hovernet_seed,
    "stage1_budget": "2 epochs with a smoke-only refresh every epoch; full Train24 input view; checkpoint selected on Val6",
    "stage2_budget": "12 real OP candidates on Val2; selected target generated for Train4",
    "hovernet_budget": "10 frozen-encoder epochs + 20 fine-tuning epochs on Train4 patches",
    "calibration_budget": "3 postprocess candidates + 2 pruning candidates on Val2",
    "stage1_selection": str(stage1.resolve()),
    "stage2_selection": str(stage2.resolve()),
    "hovernet_selection": str(hover.resolve()),
    "postprocess_selection": str((calibration / "postprocess/selected_postprocess_config.json").resolve()),
    "pruning_selection": str((calibration / "pruning/selected_pruning_config.json").resolve()),
    "test_used": False,
    "test_evaluation_performed": False,
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "[DONE] Full ratio010 smoke passed. TEST14 was not read."
echo "Completion marker: $OUTPUT_ROOT/FULL_PIPELINE_SMOKE_COMPLETE.json"
