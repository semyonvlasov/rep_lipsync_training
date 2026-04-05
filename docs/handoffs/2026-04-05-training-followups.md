# Training / Fetch Follow-Ups As Of 2026-04-05

## Scope

This handoff records the latest pipeline changes after the 2026-04-04 SyncNet handoff.

It does **not** replace the current best SyncNet handoff.

Current best remains:
- `docs/handoffs/2026-04-04-syncnet-best-checkpoints.md`

## Current Best Convention

Do not change `current best` from this note alone.

The canonical `mirror` teacher remains:
- `training/output/training_cuda3090_syncnet_mirror_medium_20260404/syncnet/syncnet_best_off_eval.pth`

For `mirror` generator runs, default to that `best_off_eval` checkpoint unless an experiment explicitly says otherwise.

## Incremental Dataset Merge

`merge_faceclip_archives_from_gdrive.py` now works incrementally by default.

Behavior:
- archives already marked as `merged` or `merged_cleaned` in the merge manifest are skipped
- full reread is only requested explicitly via `--reload-all`

The default TalkVid processed root was also normalized:
- use `data/talkvid/processed`
- do not treat `processed_medium` as the canonical root anymore

Important nuance:
- historical data may still have a compatibility symlink
- training and merge code should target `data/talkvid/processed`

## Shared Lazy Cache

Generator and SyncNet should use the same dataset `_lazy_cache`.

Recent fix:
- removed the separate generator cache pattern like `data/_lazy_cache_generator_*`
- generator config now uses the shared per-dataset caches under:
  - `training/data/hdtf/processed/_lazy_cache`
  - `training/data/talkvid/processed/_lazy_cache`

Reason:
- the separate generator cache was duplicating materialization already done by SyncNet
- on `5080` this caused unnecessary rematerialization and misleading throughput

Operational note:
- on `5080`, the separate generator cache was deleted and generator was restarted on the shared cache
- on `3090`, the old generator cache was removed before the later SyncNet rerun

## 5080 Bootstrap / CUDA Compatibility

Bootstrap now checks whether installed `torch` is actually compatible with the GPU architecture.

Added:
- `training/scripts/ensure_torch_cuda_compat.py`

Expected bootstrap behavior:
- verify CUDA is available
- verify the needed SM architecture is present in `torch`
- run a real CUDA op
- if incompatible, upgrade to a compatible `torch/torchvision/torchaudio` build

Reason:
- `RTX 5080` initially failed with `no kernel image is available for execution on the device`
- this caused widespread `sync_alignment` failures and a collapsed dataset

## Generator Training Changes

`train_generator.py` received three important operational changes:

1. `ETA` now uses the most recent progress window instead of the whole run history
2. benchmark-on-eval is integrated into the training loop
3. benchmark artifacts are written next to the generator checkpoints

Benchmark behavior:
- on eval, save fresh `generator_latest.pth`
- run the sample benchmark
- write outputs under:
  - `generator_latest_bench`
  - `generator_best_off_eval_bench`

Current limitation:
- benchmark is skipped if benchmark sample assets are not present on that host

## Sync Alignment Robustness

`train_syncnet.py` / dataset init no longer hard-fail on certain broken lazy samples during sync alignment/materialization.

Instead:
- failed sync-alignment entries are marked and skipped
- the whole run does not abort because one clip has no valid candidate window

## Faceclip Monitor / Reaper

Remote monitoring was tightened up:
- `monitor_faceclip_remotes.py`
  - writes incremental status updates
  - replaces stale registry entries by remote name
  - preserves cached last-known status instead of flashing false `no response`
- `reap_idle_faceclip_remotes.py`
  - avoids repeated skip-log spam for already-dead instances
- `training/run_faceclip_idle_reaper.sh`
  - local `caffeinate` launcher for overnight cleanup

## TalkVid Fetch Changes

Recent local fetch fixes:
- canonical background launch recipe is now fixed and documented
- fetch can bootstrap cookies from browser once and then continue via `cookies.txt`
- conservative fetch mode can run with `jobs=1`
- per-batch size enforcement now uses actual batch directory size, not only bytes downloaded in the current invocation
- resume/next-batch naming was fixed so sealing `batch_003N` moves correctly to `batch_003(N+1)`
- removed videos are now classified as `video_unavailable` and whole `video_key` gets blocked after first failure
- optional cookie rotation was added:
  - switch between two cookie variants every `500` successful downloads

Source of truth for the local fetch recipe remains:
- `docs/handoffs/2026-03-23-talkvid-youtube-fetch.md`

## 3090 Init-From-Drive SyncNet Rerun

There was a later `3090` SyncNet run:
- `training/output/training_cuda3090_syncnet_mirror_medium_init_from_drive_20260404`

Important:
- do **not** treat it as the new `current best`

Reason:
- it was not apples-to-apples with the 2026-04-04 canonical handoff run

Specifically:
- the init checkpoint came from a `5080` run:
  - `config_output_dir = output/training_rtx5080_syncnet_mirror_medium_20260404`
- the split was much larger than the handoff run

Split comparison:
- handoff run:
  - `total=20896`
  - `train=18848`
  - `val=2048`
- init-from-drive rerun:
  - `total=48306`
  - `train=46258`
  - `val=2048`

Observed effect:
- the rerun's `best_off_eval` landed at `epoch=0`, `step=0`
- that result reflects reevaluation on a different split, not a clean same-split improvement over the canonical handoff best

Conclusion:
- keep `2026-04-04-syncnet-best-checkpoints.md` as the source of truth for `current best`
- do not promote the init-from-drive rerun without a controlled apples-to-apples comparison

## 5080 Generator Teacher Convention

One bad intermediate state occurred:
- the `5080` generator was briefly pointed at a non-canonical `5080` SyncNet best-off checkpoint via a misleading path

This was corrected.

Current intended rule:
- use the canonical handoff teacher from `2026-04-04-syncnet-best-checkpoints.md`
- if an override is needed, use the launcher override explicitly:
  - `SYNCNET_CKPT=...`

Related launcher:
- `training/scripts/launch_generator_best_off_syncnet_from_scratch_20260404.sh`

## Recommended Next-Chat Assumptions

Use these assumptions by default:
- `current best` SyncNet is still the one recorded on `2026-04-04`
- generator and SyncNet should share the same dataset `_lazy_cache`
- `data/talkvid/processed` is the canonical processed TalkVid root
- merge is incremental by default
- `mirror` generator teacher should default to canonical `best_off_eval`, not an ad hoc latest/best from a fresh side experiment
