# Face Processing Migration Handoff As Of 2026-04-13

## Scope

This note captures the current state of the dataset `process` pipeline after the
first integration pass with `face_processing`, plus the decision made in chat:

- move the full `process` stage into the standalone
  `https://github.com/semyonvlasov/face_processing` repo
- include the processing orchestrator, not just ranking/export code
- keep `fetch`, training, and eval in `rep_lipsync_training`

This handoff is intended to continue the migration in a new chat without having
to re-discover the current architecture or rerun the same experiments.

## Decision

We should treat `face_processing` as the canonical home for the full processing
system:

- analysis
- segmentation
- ranking
- stabilization
- export
- archive-level processing orchestration
- Drive/local batch processing state machine
- dockerized CPU/GPU workers

What should remain in `rep_lipsync_training`:

- `fetch_hdtf_*`
- `fetch_talkvid_*`
- training
- eval
- thin wrappers only if still needed temporarily

Rationale:

- current ranking/export logic is already generic and no longer depends on
  TalkVid manifest metadata
- source of truth is currently split between two repos, which is fragile
- current setup is the worst of both worlds: neither a clean dependency nor a
  fully self-contained processing implementation

## Current Architecture Problem

Confirmed current runtime behavior:

- the Python interpreter comes from the external repo:
  `/Users/semenvlasov/Documents/repos/face_processing/.venv/bin/python`
- the MediaPipe model asset comes from the external repo:
  `/Users/semenvlasov/Documents/repos/face_processing/assets/face_landmarker_v2_with_blendshapes.task`
- but the imported `face_processing` module is the vendored copy inside
  `rep_lipsync_training`

Verified by `importlib.util.find_spec(...)`:

- `face_processing` resolves to
  `/Users/semenvlasov/Documents/repos/lipsync_test/rep_lipsync_training/face_processing/__init__.py`
- `face_processing.cli` resolves to
  `/Users/semenvlasov/Documents/repos/lipsync_test/rep_lipsync_training/face_processing/cli.py`

So the current architecture is:

- external repo provides `venv` and model asset
- current repo provides the executed processing code

This is the core thing to fix.

## Latest Pushed State

Latest pushed commit in `rep_lipsync_training`:

- `2a6e06d Add disposable Docker process worker and harden resume`

`origin/main` currently equals local `main` at that commit.

Nothing from the new stabilization/demotion work is committed yet.

## Uncommitted Processing WIP In rep_lipsync_training

These files contain the current local-only stabilization/demotion changes:

- `dataset_prepare/process/common/export_faceclip_batch.py`
- `dataset_prepare/process/configs/common.yaml`
- `face_processing/config.py`
- `face_processing/crop_export.py`
- `face_processing/logging_utils.py`
- `face_processing/models.py`
- `face_processing/pipeline.py`
- `face_processing/ranking.py`

Unrelated local user changes also exist and should not be mixed into the
processing commit:

- `training/configs/generator_mirror_gan_hdtf_talkvid.yaml`
- untracked `AGENTS.md`
- untracked `docs/handoffs/...`
- untracked `training/configs/...`
- untracked `training/scripts/...`
- untracked `training/snapshots/...`

## What The Local WIP Actually Changes

### 1. Stabilized crop geometry

Added anchor-based crop stabilization in vendored `face_processing`:

- compute per-frame anchor distances from:
  - left eye
  - right eye
  - mouth
- compute raw rotated crop geometry from rotated landmarks
- compute smoothed reference transform:
  - `cx`
  - `cy`
  - `scale`
- export can now use stabilized crop geometry instead of raw per-frame crop

Files:

- `face_processing/crop_export.py`
- `face_processing/models.py`
- `face_processing/logging_utils.py`
- `face_processing/pipeline.py`

### 2. New ranking metrics for stability

Added segment-level metrics:

- `eye_dist_std_ratio`
- `eye_mouth_std_ratio`
- `scale_outlier_ratio`

Ranking now checks those metrics for `confident` and `medium`.

Files:

- `face_processing/ranking.py`
- `face_processing/models.py`
- `face_processing/config.py`
- `dataset_prepare/process/configs/common.yaml`

### 3. Config plumbing

Added:

- `stabilization.enabled`
- `stabilization.window`
- `stabilization.scale_outlier_threshold_ratio`

And wired these values from process YAML into the vendored `face_processing`
config used by the exporter.

Files:

- `face_processing/config.py`
- `dataset_prepare/process/common/export_faceclip_batch.py`
- `dataset_prepare/process/configs/common.yaml`

## Experiment Run On Real Clip

Original raw input used for comparison:

- `/Users/semenvlasov/Downloads/videovideoiq9-qgw1xGY-scene26-scene1_raw.mp4`

Comparison outputs were produced locally in `/tmp`:

- baseline:
  `/tmp/faceproc_compare_baseline/videovideoiq9-qgw1xGY-scene26-scene1_raw/videovideoiq9-qgw1xGY-scene26-scene1_raw_seg_000.mp4`
- stabilized:
  `/tmp/faceproc_compare_stabilized/videovideoiq9-qgw1xGY-scene26-scene1_raw/videovideoiq9-qgw1xGY-scene26-scene1_raw_seg_000.mp4`
- strict-demotion:
  `/tmp/faceproc_compare_strict/videovideoiq9-qgw1xGY-scene26-scene1_raw/videovideoiq9-qgw1xGY-scene26-scene1_raw_seg_000.mp4`

### Baseline run

Config:

- CPU
- stabilization disabled

Observed segment rank:

- `medium`

Metrics:

- `mean_abs_yaw = 3.92`
- `mean_abs_pitch = 3.81`
- `mean_abs_roll = 3.78`
- `max_abs_yaw = 9.56`
- `max_abs_pitch = 9.23`
- `max_abs_roll = 8.21`
- `face_size_std_ratio = 0.0284`
- `std_cx = 13.62`
- `std_cy = 6.24`
- `eye_dist_std_ratio = 0.0214`
- `eye_mouth_std_ratio = 0.0338`
- `scale_outlier_ratio = 0.0157`

Main reason it is not `confident`:

- `eye_mouth_std_ratio = 0.0338`
- current `conf_eye_mouth_std_ratio = 0.03`

### Stabilized run

Config:

- CPU
- stabilization enabled
- same ranking thresholds as baseline

Observed segment rank:

- still `medium`

Important detail:

- the rank does **not** change because ranking is computed from the analyzed
  segment geometry, not from a second pass over the already exported video
- the stabilized export changes the output crop behavior, but not the original
  segment metrics used for ranking in this implementation

### Measured effect on final exported clip

To verify actual visual stability, the baseline and stabilized **output clips**
were re-analyzed at `400x400`.

Measured output stability:

- baseline:
  - `cx_std = 4.11`
  - `cy_std = 1.09`
  - `eye_dist_std_ratio = 0.01539`
  - `eye_mouth_std_ratio = 0.03145`
- stabilized:
  - `cx_std = 1.84`
  - `cy_std = 0.69`
  - `eye_dist_std_ratio = 0.00937`
  - `eye_mouth_std_ratio = 0.01739`

Interpretation:

- stabilized export materially reduces final output jitter
- especially:
  - horizontal drift
  - eye-distance variability
  - eye-to-mouth variability

### Strict demotion check

Used the same stabilized export, but tightened only:

- `med_eye_mouth_std_ratio = 0.03`

Since the clip's segment metric is:

- `eye_mouth_std_ratio = 0.0338`

The rank became:

- `unconfident`

This confirms the demotion path works mechanically.

Important:

- `strict` was not meant as a visual comparison
- it was only used to verify that the new demotion plumbing is wired correctly

## Current Local Processing Config Used For MPS

File:

- `dataset_prepare/process/configs/process_raw_archives_to_lazy_faceclips_gdrive_local_mps.yaml`

Current contents:

- extends `process_raw_archives_to_lazy_faceclips_gdrive.yaml`
- `runtime.python_bin = /Users/semenvlasov/Documents/repos/face_processing/.venv/bin/python`
- `face_processing.detection.model_path = /Users/semenvlasov/Documents/repos/face_processing/assets/face_landmarker_v2_with_blendshapes.task`
- `face_processing.detection.use_gpu = true`

This config is also local-only and not committed.

## Raw Drive Reset Already Done

Raw Drive folder:

- `1v06momk8fR-eqw79Z93zczBx_InWsCS9`

All `.processed` suffixes were already reverted back to `.tar`.

Confirmed reverted:

- `hdtf_clips_batch_0000.tar` ... `hdtf_clips_batch_0007.tar`
- `talkvid_raw_batch_0039.tar` ... `talkvid_raw_batch_0043.tar`

The rest were already plain `.tar`.

## Local Restart Command Used For Current Repo

This is the command used to restart local raw-drive processing from the current
repo:

```bash
rm -rf data/dataset_prepare/process/raw_faceclips_cycle
mkdir -p data/dataset_prepare/process/logs

MPLCONFIGDIR=/tmp/mplconfig \
/Users/semenvlasov/Documents/repos/face_processing/.venv/bin/python \
dataset_prepare/process/common/process_raw_archives_to_lazy_faceclips_gdrive.py \
dataset_prepare/process/configs/process_raw_archives_to_lazy_faceclips_gdrive_local_mps.yaml \
2>&1 | tee -a data/dataset_prepare/process/logs/raw_drive.log
```

Tail:

```bash
tail -f data/dataset_prepare/process/logs/raw_drive.log
```

## Recommended Migration Plan

### Target repo split

Move into external `face_processing` repo:

- vendored `face_processing/` package code
- `dataset_prepare/process/common/export_faceclip_batch.py` logic, adapted into
  canonical batch/archive processing CLI
- raw-archive processing orchestration currently in:
  - `dataset_prepare/process/common/process_raw_archives_to_lazy_faceclips_gdrive.py`
  - `dataset_prepare/process/common/process_local_batches_to_lazy_faceclips.py`
  - `dataset_prepare/process/common/pipeline_utils.py`
- process configs
- docker worker logic

Leave in `rep_lipsync_training`:

- `dataset_prepare/fetch/**`
- training
- eval
- temporary thin wrappers only if needed for transition

### Suggested migration order

1. Port the current local-only stabilization/demotion changes from
   `rep_lipsync_training/face_processing/` into the standalone
   `face_processing` repo.
2. Add canonical CLIs in `face_processing` for:
   - single video
   - input directory
   - local batch folder
   - gdrive raw-archive processing
3. Move process-stage state machine and archive claim/download/upload logic into
   `face_processing`.
4. In `rep_lipsync_training`, replace process scripts with thin wrappers or
   remove them entirely after validation.
5. Stop using the vendored `face_processing/` copy inside
   `rep_lipsync_training`.

## Known Open Questions

### 1. Ranking timing vs stabilized export

Current implementation computes rank from source-segment metrics after
`prepare_segment_crop_geometry(...)`, but before any second-pass measurement on
the final exported `400x400` clip.

Question:

- should ranking be based on:
  - source geometry with stabilization-derived metrics, or
  - actual final exported clip stability?

Current evidence:

- stabilized export clearly improved final output stability
- but the rank stayed `medium`

So ranking semantics may still need adjustment.

### 2. Commit boundary

The current stabilization work is not committed. The next chat should decide:

- commit it in `rep_lipsync_training` temporarily, then port it
- or port it directly into external `face_processing` first and avoid another
  short-lived commit in the current repo

Given the migration decision, direct porting to external `face_processing`
looks cleaner.

## Important Commands Used During Investigation

Check which `face_processing` is actually imported:

```bash
/Users/semenvlasov/Documents/repos/face_processing/.venv/bin/python - <<'PY'
import importlib.util
print(importlib.util.find_spec('face_processing').origin)
print(importlib.util.find_spec('face_processing.cli').origin)
PY
```

Re-run the comparison clip:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/Users/semenvlasov/Documents/repos/face_processing/.venv/bin/python \
-m face_processing.cli \
--input '/Users/semenvlasov/Downloads/videovideoiq9-qgw1xGY-scene26-scene1_raw.mp4' \
--output-dir /tmp/faceproc_compare_baseline \
--config /tmp/faceproc_baseline.json \
--save-frame-log
```

## Copy-Paste Prompt For The Next Chat

```text
Continue from docs/handoffs/2026-04-13-face-processing-migration-handoff.md.

Goal:
- move the full process stage from rep_lipsync_training into the standalone
  face_processing repo, including orchestration
- stop using the vendored face_processing copy inside rep_lipsync_training

Important facts:
- current pushed rep_lipsync_training main is 2a6e06d
- stabilization/demotion changes are only local in rep_lipsync_training
- external face_processing repo is currently used only for venv + model asset
- actual imported face_processing code still comes from the vendored copy in
  rep_lipsync_training

Please start by:
1. reviewing the uncommitted processing diffs listed in the handoff
2. proposing the exact target file/CLI structure in the external face_processing repo
3. then porting the stabilization/demotion work there first
4. after that, migrate the process orchestrator into face_processing
```
