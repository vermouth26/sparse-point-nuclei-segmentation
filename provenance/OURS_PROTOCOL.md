# Ratio-0.10 C+fallback Protocol

## Scope

The MoNuSeg ratio-0.10 C+fallback run uses the TRAIN24, VAL6, and TEST14 split in
`monuseg_split_24_6_14.json`.

## Sparse supervision

Each training image contributes 10% of its center annotations. The configured
`sparse_points.annotation_seed` fixes the sample. A different seed produces a
different annotation realization and starts a new experiment.

Stage-I training, center completion, Stage-II proposal generation,
pseudo-instance construction, and HoVerNet training do not read training
instance masks.

## Stage I

- Model: ResUNet34 with torchvision ImageNet initialization.
- Training: 100 epochs with parameters from the main JSON and Stage-I YAML.
- Curriculum recovery: `curriculum_use_gt_points=false`.
- Checkpoint rule: maximum VAL6 macro center-detection F1 across all saved
  epochs; an exact tie selects the earlier epoch.
- Center operating point: threshold 0.65, component area 12--150, and merge
  radius 15.
- Output: immutable manual centers merged with nonduplicate recovered centers.

## Stage II

SAM2.1 Hiera Base+ receives completed centers as positive prompts. Candidate
selection mode is `sam_only`: masks are checked by center, score, and area, but
are not clipped or ranked by Voronoi/cluster overlap.

The validation grid contains 12 candidates:

- SAM score: 0.65, 0.70, 0.75;
- minimum area: 10, 20;
- maximum area: 3000, 5000.

Selection uses VAL6 macro PQ, then AJI, then the smaller of precision and recall,
then distance of pred/GT from 1, followed by deterministic configuration
tie-breaks.

Fallback is checked after assembling the SAM-only map C and applies only to
completed centers that C does not cover. Valid cluster-derived fallback masks
and SAM-selected masks share the same instance assembly. The configured
`winner` policy resolves pixel conflicts. Zero denotes background in the output
instance map.

## HoVerNet

The selected TRAIN24 pseudo-instance maps are converted to the patch format used
by the official HoVerNet training path. HoVerNet generates NP/HV targets online
from each augmented integer instance map.

Training has a 30-epoch frozen-encoder phase followed by fine-tuning through
epoch 80. All saved fine-tuning checkpoints are evaluated on VAL6. Selection is
by PQ, AJI, DQ, and then distance of global pred/GT from 1.05.

## Calibration and evaluation

`scripts/03_calibrate_val.sh` selects postprocess and instance-pruning settings
on VAL6. TEST14 is evaluated only after the Stage-I, Stage-II, HoVerNet,
postprocess, and pruning artifacts are frozen. Its results cannot change a
selection or trigger a rerun.

The primary metrics are image-level macro Dice, AJI, DQ, SQ, PQ, detection
precision, recall, F1, and pred/GT. Pooled instance counts must be identified
separately from macro metrics.

## Configuration authority

`configs/ratio010_mainline.json` is the machine-readable experiment
configuration. `configs/paths.example.env` contains path placeholders only and
does not define scientific parameters. Runtime artifacts must record the
resolved configuration, split, selected files, and hashes.
