# rep_lipsync_training

Training-only repository for a LipSync generator + SyncNet teacher workflow.

This repo intentionally keeps only the current training pipeline, configs,
orchestration scripts, handoffs, and deployment helpers. Runtime data,
checkpoints, archives, and output artifacts are excluded from git.

## Structure

- `training/configs`
  Current configs for the latest mixed `HDTF + TalkVid` training flow, the
  dedicated `SyncNet` recipe, local validation, and generator teacher
  comparison runs.
- `training/data`
  Runtime dataset code only: audio processing and dataset loaders.
- `training/models`
  Local training models: generator, discriminator, local SyncNet.
- `training/scripts`
  Shared Python entrypoints used by fetch, preprocess, train, comparison,
  validation, and export workflows.
- `training/workflows/fetch`
  Public dataset fetch + archive upload flows for `HDTF` and `TalkVid`.
- `training/workflows/preprocess`
  Local sequential preprocessing and processed-archive upload flow.
- `training/workflows/train`
  Full mixed training pipeline, SyncNet snapshot training, generator teacher
  comparison, and remote sync/launch wrappers.
- `training/workflows/deploy`
  Generator validation plus CoreML export entrypoint.
- `models/official_syncnet`
  External reference SyncNet code and SFD face detector code. Checkpoints are
  expected locally under `models/official_syncnet/checkpoints/` but are not
  stored in git.
  The upstream reference model file keeps its original filename
  `models/wav2lip.py`; only our local workflow/config/script names were
  renamed.
- `docs/handoffs`
  Latest handoff material from the most recent training runs.

## Included latest pipeline

- `HDTF` public fetch -> archive batches -> Google Drive upload
- `TalkVid` public fetch -> archive batches -> Google Drive upload
- archive ingest from Google Drive
- incremental preprocess
- `SyncNet` training
- teacher comparison against the official reference SyncNet
- generator training with watchdog
- generator teacher comparison workflow
- local processed-faceclip export pipeline
- generator validation and CoreML export

## Intentionally excluded for now

Some older or ambiguous helper scripts from the source repo were not copied on
purpose. They should be reviewed one by one before being added:

- older `MPS` launchers
- older full-training wrappers superseded by the latest mixed pipeline
- legacy smoke-only helpers
- historical backup/restore utilities
- one-off archive download helpers superseded by the current pipeline

## First setup

1. Put the official checkpoint under:
   `models/official_syncnet/checkpoints/lipsync_expert.pth`
2. Install Python deps used by the scripts.
3. Use the workflows in `training/workflows/*` as the entrypoints.
