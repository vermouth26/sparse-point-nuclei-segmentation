# Ratio-0.10 C+fallback Reproduction Package

Code in this directory runs the MoNuSeg ratio-0.10 C+fallback pipeline:

```text
sparse center annotations
-> Stage-I ResUNet34 center completion
-> immutable manual/recovered center merge
-> SAM2 point-prompt proposal map C
-> cluster fallback for completed centers not covered by C
-> integer pseudo-instance maps
-> official HoVerNet training with online NP/HV target generation
-> validation-only selection and calibration
-> optional fixed test evaluation
```

The archive does not include datasets, model weights, trained checkpoints, or
predictions. Their locations come from environment variables. Experiment
parameters and random seeds live in `configs/ratio010_mainline.json`, so neither
filenames nor entrypoints depend on a particular seed.

## Contents

```text
reproduction_ratio010/
|-- configs/                  experiment configuration and path template
|-- detection/                Stage-I data, model, training, and inference code
|-- ssl_stage0/               Stage-I configuration and checkpoint helpers
|-- segmentation/             instance segmentation metrics
|-- experiments/              center, SAM2, HoVerNet, and selection entrypoints
|-- third_party/hover_net/     vendored HoVerNet source and upstream license
|-- tools/instance_pruning/    validation calibration and fixed application
|-- scripts/                   orchestration and deterministic selectors
|-- environment/               direct Python dependencies
|-- provenance/                split, protocol, and calibration-grid records
|-- detail.md                  implementation guide
`-- LICENSE
```

## Ubuntu 24.04 setup

The current integration test used Ubuntu 24.04, Python 3.10, and an NVIDIA GPU.
Install the PyTorch build that matches the local CUDA driver before installing
the remaining dependencies.

```bash
conda create -n ratio010-repro python=3.10 -y
conda activate ratio010-repro

# Install torch and torchvision for the local CUDA runtime first.
pip install -r environment/requirements_runtime.txt

# Install the official SAM2 repository in this environment, or provide a
# separate SAM2 Python executable through SAM2_PYTHON.
pip install -e /path/to/sam2
```

HoVerNet is vendored under `third_party/hover_net` with its upstream license.
SAM2 code and weights are not redistributed.

## External inputs

The expected MoNuSeg layout is:

```text
DATA_ROOT/MO/
|-- images/*.png
|-- labels_point/*.png
|-- labels_instance/*_label.npy
`-- train_val_test.json
```

Training reads RGB images and generated sparse center annotations. The declared
validation selectors and the optional fixed test evaluation are the only steps
that read full instance labels. Stage-I completion, pseudo-target construction,
and network training do not read hidden training instance labels.

The following external weights are required:

- `sam2.1_hiera_base_plus.pt`;
- the matching official `sam2.1_hiera_b+.yaml` configuration;
- the HoVerNet Preact-ResNet50 ImageNet initialization.

## Path configuration

Copy the path template outside the package and fill in the paths for the target
system:

```bash
cp configs/paths.example.env "$HOME/ratio010_paths.env"
nano "$HOME/ratio010_paths.env"
source "$HOME/ratio010_paths.env"
cd "$PACKAGE_ROOT"
```

Keep the local path file outside the archive. It defines only filesystem
locations: `PYTHON`, `SAM2_PYTHON`, `PACKAGE_ROOT`, `DATA_ROOT`,
`PROCESSED_ROOT`, `OUTPUT_ROOT`, `SAM2_CHECKPOINT`, `SAM2_CONFIG`,
`HOVERNET_PRETRAIN`, and `HOVERNET_ROOT`.

## Seed configuration

Edit only `configs/ratio010_mainline.json` to change experimental seeds:

| Key | Consumer | Scope |
|---|---|---|
| `sparse_points.annotation_seed` | `prepare_sparse_points.py` | sparse center sampling |
| `stage1.seed` | Stage-I training | initialization, sampling, augmentation |
| `hovernet.seed` | HoVerNet training | both training phases and data loaders |

A seed change starts a new experiment, so use a new `OUTPUT_ROOT`. SAM2 proposal
generation is deterministic inference in this workflow and has no seed option.

## Package check

```bash
source "$HOME/ratio010_paths.env"
cd "$PACKAGE_ROOT"
bash scripts/00_check.sh
```

The checker verifies required files, Python syntax, method settings, private
absolute paths, editor metadata, cached outputs, logs, and binary weights.

## Full integration smoke test

The integration smoke runs every major stage with reduced data and epoch budgets.
It does not read TEST14.

```bash
source "$HOME/ratio010_paths.env"
export OUTPUT_ROOT=/path/to/new/smoke-output
cd "$PACKAGE_ROOT"
bash scripts/run_full_pipeline_smoke.sh
```

On success, the script creates `FULL_PIPELINE_SMOKE_COMPLETE.json` with
`scientific_result=false`, `test_used=false`, and
`test_evaluation_performed=false`. Smoke metrics are integration diagnostics,
not scientific results.

## Full ratio-0.10 run

By default, the run completes validation selection and stops before TEST14:

```bash
source "$HOME/ratio010_paths.env"
export OUTPUT_ROOT=/path/to/new/ratio010-output
cd "$PACKAGE_ROOT"
bash scripts/run_end_to_end.sh
```

The run order is:

1. deterministic ratio-0.10 center sampling;
2. 100-epoch Stage-I training;
3. maximum VAL6 macro detection-F1 checkpoint selection, with the earlier
   epoch selected on an exact tie;
4. TRAIN24 and VAL6 completed-center export;
5. VAL6-only 12-candidate Stage-II operating-point selection;
6. selected TRAIN24 C+fallback pseudo-instance generation;
7. official HoVerNet training and VAL6-only checkpoint selection;
8. VAL6-only postprocess and pruning calibration;
9. creation of `PRETEST_COMPLETE.json`.

The Stage-II grid is:

```text
SAM score = {0.65, 0.70, 0.75}
min area  = {10, 20}
max area  = {3000, 5000}
```

To include one fixed TEST14 evaluation in a new uninterrupted run:

```bash
source "$HOME/ratio010_paths.env"
export OUTPUT_ROOT=/path/to/new/ratio010-output-with-test
export RUN_TEST=1
cd "$PACKAGE_ROOT"
bash scripts/run_end_to_end.sh
```

Do not change a checkpoint, operating point, postprocess configuration, pruning
configuration, or random seed after reading test results.

## Important outputs

Archive at least:

- `stage1/selection/selected_epoch.json` and its checkpoint;
- TRAIN24 and VAL6 completed-center manifests;
- `stage2_val/selected_stage2_config.json`;
- the selected pseudo-target manifest;
- `hovernet_selected_checkpoint.json` and its checkpoint;
- selected postprocess and pruning JSON files;
- `PRETEST_COMPLETE.json`;
- split, sparse-point, configuration, and final artifact hashes.

## Method constraints

- Stage-I uses ImageNet-pretrained ResUNet34 without Hibou initialization,
  AlphaMix initialization, or BN recalibration.
- Manual sparse centers remain immutable during center completion.
- Stage II uses SAM-only candidate selection plus cluster fallback for uncovered
  completed centers. It is not a geometry-clipped SAM method.
- HoVerNet generates NP/HV targets online from the selected integer
  pseudo-instance maps.
- Stage-I, Stage-II, HoVerNet, postprocess, and pruning selection use validation
  data only.

See `detail.md` for implementation entrypoints and `provenance/OURS_PROTOCOL.md`
for the compact protocol specification.

## Release validation and freezing

Before creating an archive, run the package check and a fresh integration smoke.
Then create and verify the file-hash manifest from the package root:

```bash
cd "$PACKAGE_ROOT"
bash scripts/00_check.sh

find . -type f ! -name SHA256SUMS -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

Create the ZIP from the parent directory to keep one top-level folder:

```bash
cd "$(dirname "$PACKAGE_ROOT")"
python -m zipfile -c reproduction_ratio010_submission.zip reproduction_ratio010
sha256sum reproduction_ratio010_submission.zip \
  > reproduction_ratio010_submission.zip.sha256
```

Any later source or documentation edit invalidates `SHA256SUMS` and the ZIP
hash. Repeat the package check, smoke test, manifest generation, and archive
creation after such an edit.

## License

The package license is in `LICENSE`. Vendored HoVerNet code retains its upstream
license in `third_party/hover_net/LICENSE`.
