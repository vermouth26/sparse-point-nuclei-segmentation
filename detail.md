# Implementation Guide

Execution begins in `scripts/run_end_to_end.sh`. The sections below follow that
order and cover the first-party modules it calls. HoVerNet keeps its upstream
layout and documentation under `third_party/hover_net`.

## Directory tree

```text
reproduction_ratio010/
|-- configs/
|   |-- paths.example.env
|   |-- ratio010_mainline.json
|   `-- stage1_ratio010.yaml
|-- detection/
|-- environment/requirements_runtime.txt
|-- experiments/
|   |-- ablation_monuseg_r0_raw_fixedop/
|   |-- official_hovernet/
|   |-- hover_stage23_alpha050_bn_region_prior/stage2_bridge/
|   |-- monuseg_official_hovernet/
|   |-- stage1_uniform_p084_protocol/
|   `-- vit_region_prior_sam/
|-- provenance/
|-- scripts/
|-- segmentation/metrics.py
|-- ssl_stage0/
|-- third_party/hover_net/
|-- tools/instance_pruning/
|-- LICENSE
|-- README.md
`-- detail.md
```

## Configuration

`configs/ratio010_mainline.json` holds the split size, annotation ratio, seeds,
Stage-I parameters, Stage-II grid, HoVerNet schedule, and calibration protocol.
`configs/stage1_ratio010.yaml` supplies path-independent Stage-I defaults.
`scripts/materialize_stage1_config.py` adds runtime paths, the Stage-I seed, and
the run description. External file locations are listed separately in
`configs/paths.example.env`.

## Orchestration scripts

- `run_end_to_end.sh` coordinates the complete workflow and stops before TEST14
  unless `RUN_TEST=1`.
- `run_full_pipeline_smoke.sh` exercises the same major stages with reduced
  budgets and never reads TEST14.
- `01_stage1_train_select.sh` prepares the Stage-I view, trains the detector,
  and invokes the epoch selector.
- `01_generate_selected_targets.sh` applies the selected C+fallback operating
  point to TRAIN24 and converts the resulting instance maps.
- `02_train_hovernet.sh` prepares patches and runs both HoVerNet phases.
- `03_calibrate_val.sh` performs validation-only postprocess and pruning
  selection.
- `00_check.sh` runs the package checker without loading data or a GPU model.

The remaining Python utilities read scalar configuration values, prepare sparse
points, materialize runtime YAML, collect Stage-II metrics, select checkpoints
and operating points, and apply the frozen postprocess settings.
`check_package.py` checks the directory structure, syntax, method settings, and
packaging exclusions.

## Stage-I implementation

The `detection/` package contains the ResUNet34 model, datasets, transforms,
training loop, tiled inference, and supporting utilities. Configuration parsing,
checkpoint handling, initialization, and the Stage-I wrapper are in
`ssl_stage0/`.

R0 training starts in
`experiments/ablation_monuseg_r0_raw_fixedop/train_r0.py`. It uses ImageNet
initialization with training-GT curriculum access disabled.
`prepare_stage1_inputs.py` creates the isolated sparse-supervision view.

`experiments/stage1_uniform_p084_protocol/select_stage1_protocol.py` evaluates
all saved epochs on VAL6 and selects maximum macro detection F1. The packaged
entrypoint passes `--no-precision-floor`; exact ties favor the earlier epoch.

## Center and pseudo-instance generation

`experiments/vit_region_prior_sam/common.py` contains path validation, point-map
conversion, immutable center merging, Stage-I loading, serialization, and
diagnostic metrics.

`01_prepare_stage1_points.py` runs the selected Stage-I checkpoint, extracts
recovered centers, merges them with sparse centers, and writes the completed
center manifest. TRAIN24 export uses `--no-gt-diagnostics`.

`02_generate_bonus_region_baseline.py` builds point-derived Voronoi and cluster
support regions. They provide fallback candidates and do not clip SAM-selected
masks in the C+fallback method.

`03_generate_sam_point_regions.py` implements the prompt backends, candidate
selection, fallback construction, and joint conflict resolution. The configured
mainline uses SAM2, `candidate_selection_mode=sam_only`, `fallback=cluster`, and
the `winner` conflict policy. Outputs include integer pseudo-instance maps and
`sam_region_manifest.json`.

`04_eval_pseudo_region_quality.py` computes validation-only metrics for each
Stage-II candidate. `05_convert_sam_region_to_hover_targets.py` converts selected
pseudo instances into integer maps and diagnostic NP/HV and ignore arrays.
Official HoVerNet training still generates NP/HV targets online from augmented
integer maps.

The NP/HV helper module remains under
`experiments/hover_stage23_alpha050_bn_region_prior/stage2_bridge/common.py`
because that is its inherited source location. The current method does not use
AlphaMix or BN recalibration.

## HoVerNet downstream

`generate_weak_hovernet_patches_from_manifest.py` reads RGB images and selected
pseudo-instance maps and creates the patch dataset.

`experiments/official_hovernet/train_official_hovernet_patch_fullsup.py`
is the shared patch trainer used for both HoVerNet phases. The implementation is
dataset-independent after receiving a patch directory.

`experiments/official_hovernet/hovernet_evaluation_common.py` provides the tiled
inference and metric helpers shared by the MoNuSeg evaluator and postprocess
calibration code.

`eval_official_hovernet_monuseg.py` performs MoNuSeg validation or test inference
and writes NP/HV predictions, instance maps, per-image metrics, and overall
metrics. `official_hovernet_postprocess_sweep_monuseg.py` implements the
predefined validation postprocess search; its smoke preset is used only by the
integration test.

`third_party/hover_net/` contains the unedited upstream network, target
generation, postprocessing, and license files.

## Calibration and metrics

`tools/instance_pruning/extract_instance_features.py` extracts per-instance
features. `tune_pruning_on_val.py` evaluates the predefined validation
candidates. `apply_pruning_to_test.py` applies the selected configuration without
searching. Shared logic is in `common.py` and `prune_instances.py`.

`segmentation/metrics.py` supplies Dice, AJI, DQ, SQ, PQ, precision, recall, F1,
and instance-count metrics.

## Data flow

```text
main JSON + private path environment
-> sparse center maps
-> Stage-I checkpoints and selected epoch
-> TRAIN24/VAL6 completed-center manifests
-> point-derived support-region manifests
-> VAL6 Stage-II candidate metrics and selected configuration
-> TRAIN24 C+fallback pseudo-instance maps
-> HoVerNet patches and online NP/HV targets
-> HoVerNet checkpoints and selected validation checkpoint
-> selected postprocess and pruning configurations
-> optional fixed TEST14 metrics
```

All selection finishes before the optional test evaluation. Generated files go
under `OUTPUT_ROOT`, outside the source tree.
