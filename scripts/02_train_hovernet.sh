#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
HOVERNET_ROOT="${HOVERNET_ROOT:-$ROOT/third_party/hover_net}"
MAIN_CONFIG="${MAIN_CONFIG:-$ROOT/configs/ratio010_mainline.json}"
read -r HOVERNET_SEED < <(
  "$PYTHON" -B "$ROOT/scripts/read_experiment_config.py" --config "$MAIN_CONFIG" \
    --fields hovernet.seed
)

: "${HOVER_TARGET_MANIFEST:?Set HOVER_TARGET_MANIFEST to the frozen TRAIN24 target manifest}"
: "${DATA_ROOT:?Set DATA_ROOT to the raw data root containing MO/}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
: "${HOVERNET_PRETRAIN:?Set HOVERNET_PRETRAIN to the official Preact-ResNet50 ImageNet weights}"

PATCH_DIR="$OUTPUT_ROOT/hovernet_patches"
RUN_DIR="$OUTPUT_ROOT/official_hovernet"
STAGE1_DIR="$RUN_DIR/stage1_freeze_encoder"
STAGE2_DIR="$RUN_DIR/stage2_finetune_all"

cd "$ROOT"

"$PYTHON" -B experiments/monuseg_official_hovernet/generate_weak_hovernet_patches_from_manifest.py \
  --hovernet-root "$HOVERNET_ROOT" \
  --hover-target-manifest "$HOVER_TARGET_MANIFEST" \
  --split "$DATA_ROOT/MO/train_val_test.json" \
  --subset train \
  --images "$DATA_ROOT/MO/images" \
  --out-dir "$PATCH_DIR" \
  --win-size 540 \
  --step-size 164 \
  --extract-type mirror

"$PYTHON" -B experiments/official_hovernet/train_official_hovernet_patch_fullsup.py \
  --hovernet-root "$HOVERNET_ROOT" \
  --patch-dir "$PATCH_DIR" \
  --out-dir "$STAGE1_DIR" \
  --mode original \
  --device cuda \
  --freeze-encoder \
  --epochs 30 \
  --batch-size 8 \
  --workers 4 \
  --lr 1e-4 \
  --seed "$HOVERNET_SEED" \
  --lambda-hv-mse 1.0 \
  --lambda-hv-msge 1.0 \
  --save-every 10 \
  --imagenet-pretrained "$HOVERNET_PRETRAIN"

"$PYTHON" -B experiments/official_hovernet/train_official_hovernet_patch_fullsup.py \
  --hovernet-root "$HOVERNET_ROOT" \
  --patch-dir "$PATCH_DIR" \
  --out-dir "$STAGE2_DIR" \
  --mode original \
  --device cuda \
  --epochs 80 \
  --batch-size 4 \
  --workers 4 \
  --lr 1e-4 \
  --seed "$HOVERNET_SEED" \
  --lambda-hv-mse 1.0 \
  --lambda-hv-msge 1.0 \
  --save-every 5 \
  --resume "$STAGE1_DIR/checkpoints/hovernet_epoch_030.pt"

echo "Evaluate every saved Stage2 checkpoint on VAL6, then run scripts/select_hovernet_checkpoint.py."
