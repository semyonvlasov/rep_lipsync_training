# TalkVid YouTube Fetch Handoff

Updated: `2026-03-23 13:11:07 CET`

Post-handoff update: `2026-03-23 14:00:05 CET`

Post-handoff update: `2026-03-23 14:28:11 CET`

Post-handoff update: `2026-03-23 14:55:49 CET`

Post-handoff update: `2026-03-23 15:18:38 CET`

Post-handoff update: `2026-03-23 15:56:10 CET`

Post-handoff corrective update: `2026-03-23 16:07:26 CET`

Post-handoff update: `2026-03-23 16:14:04 CET`

Post-handoff update: `2026-03-23 17:09:15 CET`

Post-handoff update: `2026-03-24 14:25:18 CET`

- Current process tracker is maintained locally in:
  - `training/output/process_tracker.md`
  - it is the best single place to check active `local` and `remote` processes, PIDs, logs, and monitor commands

- Current live state:
  - local YouTube fetch is alive again and should stay alive:
    - process: `TalkVid local fetch cycle (batch_0021)`
    - artifacts: `training/data/talkvid_local/batches/batch_0021`
    - log: `training/output/talkvid_local_cycle/cycle.log`
  - remote main pipeline is running as:
    - output: `output/full_v15_cuda3090_hdtf_talkvid_20260324_resume3`
    - key launcher pids at start: `436059`, `436061`
    - log: `output/full_v15_cuda3090_hdtf_talkvid_20260324_resume3/pipeline.log`
  - remote sidecar SyncNet run is also running in parallel on a frozen snapshot:
    - output: `output/syncnet_sidecar_snapshot_20260324_medium`
    - key pids at start: `446947`, `446948`, `446950`, `446958`
    - log: `output/syncnet_sidecar_snapshot_20260324_medium/pipeline.log`

- Remote TalkVid ingest/state changes completed before `resume3`:
  - `talkvid_raw_batch_0021.tar`, `0022.tar`, `0023.tar` were downloaded directly on the remote box and staged into `data/talkvid_remote/raw`
  - after staging those three archives, remote raw inventory became:
    - `dataset=talkvid_raw speakers=36805 root=data/talkvid_remote/raw`
  - important nuance:
    - earlier `resume2` processed clips from batches `0011..0020` even though those archives were not visible under `gdrive_downloads_smoke_v3`
    - this is because those clips had already been staged into `data/talkvid_remote/raw` by an earlier import; preprocess reads `raw/`, not the archive cache

- Main remote pipeline logic was updated and is now running in the intended mode:
  - `TALKVID_HEAD_FILTER_MODE` default is now `off`
  - this means TalkVid clips are no longer dropped early by head-pose preset during preprocess
  - after preprocess, the pipeline now runs a post-sort step:
    - `training/scripts/sort_talkvid_processed_by_quality.py`
  - training then uses speaker allowlists built from those quality tiers rather than hard-dropping clips up front
  - allowlist support was wired into:
    - `training/scripts/build_syncnet_holdout_split.py`
    - `training/scripts/compare_syncnet_teachers.py`
    - `training/scripts/train_with_watchdog.py`

- Current TalkVid quality sort results on remote processed data:
  - output dir: `output/talkvid_quality_20260324`
  - counts:
    - `confident=2962`
    - `medium=5621`
    - `unconfident=1780`
    - `rejected=970`
    - `usable_total=10363`
  - `rejected` here means a processed sample has `bad_sample=true` in `bbox.json` or invalid metadata
  - dominant `bad_reasons` among these `rejected` processed samples:
    - `short_segment`: `951`
    - `low_detection_coverage`: `34`
    - `bbox_center_jump`: `15`
    - `bbox_scale_jump`: `10`
    - `face_near_edge`: `2`
  - important nuance:
    - this `rejected` bucket is not the same thing as hard process failures like `no_stable_face`
    - it is a post-sort label over samples that already have `frames.npy + mel.npy + bbox.json`

- Current processed dataset size estimates:
  - live processed corpus on remote at latest check:
    - `HDTF`: `2.21 h`
    - `TalkVid`: `16.49 h`
    - total: `18.71 h`
  - projected total if the current staged remote raw corpus finishes processing at roughly the current average clip duration:
    - around `49.25 h` upper bound
    - a more realistic expectation discussed in chat was still comfortably `40+ h`

- `resume3` TalkVid pending count explanation:
  - example log line:
    - `Pending raw videos: 23680 (failed_skipped=1792)`
  - at that moment, the count came from:
    - `raw_mp4 = 36805`
    - `processed_complete = 11333`
    - `failed_skipped = 1792`
    - formula: `36805 - 11333 - 1792 = 23680`
  - later recounts were lower simply because `resume3` kept processing clips while the recount ran

- Frozen SyncNet sidecar snapshot details:
  - current sidecar uses `min_medium`, not all usable TalkVid tiers
  - that means it includes:
    - `confident`
    - `medium`
  - and excludes:
    - `unconfident`
    - `rejected`
  - frozen snapshot size:
    - combined snapshot: `14.31 h`
    - train split: `14.02 h`
    - holdout: `0.01 h`
  - speaker counts for that frozen snapshot:
    - combined: `8927`
    - train: `8865`
    - holdout: `10`

- Important dataset-format clarification from chat:
  - processed storage is clip-level, not pre-materialized 5-frame windows
  - each processed sample directory stores:
    - `frames.npy`: full trimmed clip as face crops, shape `(N, H, W, 3)`, `uint8`, BGR
    - `mel.npy`: full clip mel spectrogram, shape `(80, T_audio)`, `float32`
    - `bbox.json`: metadata such as `fps`, `n_frames`, `mel_frames`, `bad_sample`, `bad_reasons`, `quality`, trim info, backend/device info
    - optional `preview_contact.jpg`
  - `N` is the number of kept video frames in the processed clip
  - `H` and `W` are usually the face crop size such as `256`
  - `3` is the color channel dimension
  - `80` in `mel.npy` is the number of mel bins
  - `T_audio` is the number of mel time steps across the whole clip
  - one training window is assembled on the fly from these full arrays at load time

- Important training-format clarification from chat:
  - one SyncNet sample is a window of `T=5` consecutive video frames plus aligned audio
  - the dataset does not store those windows separately on disk
  - instead, `__getitem__` samples a random start position and slices a 5-frame window out of `frames.npy`
  - this is why `len(dataset)` is based on total frame counts rather than the number of precomputed 5-frame windows
  - current sidecar log example:
    - `[Dataset] Total: 1267866 frames across 8865 speakers`
    - `SyncNet training: 1267866 samples, 158483 batches/epoch`
  - with `batch_size=8`, `158483 = floor(1267866 / 8)`
  - this `samples` count is really “how many random sampling opportunities per epoch”, not “how many unique 5-frame windows are stored on disk”

- Important mel/frame alignment clarification from chat:
  - audio uses:
    - `sample_rate=16000`
    - `hop_size=200`
    - therefore `80` mel columns per second
  - video target rate is `25 fps`
  - so one video frame spans about `80 / 25 = 3.2` mel columns
  - one 5-frame window spans `5 / 25 = 0.2 s`, which corresponds to `16` mel columns
  - this is why `mel_step_size=16` matches the 5-frame SyncNet window
  - current loader behavior for frame-to-mel alignment is:
    - `mel_start = int(frame_idx * 3.2)`
    - i.e. floor quantization, not snapping starts to multiples of 5 frames
  - consequence:
    - there can be a small sub-frame quantization error up to about one mel step (`~12.5 ms`)
    - but positive SyncNet samples still use the same random start for audio and video, so there is no deliberate drift in positive pairs
    - negative SyncNet samples intentionally offset audio to create async examples
  - an idea discussed in chat, but not implemented yet:
    - experiment with start positions snapped to 5-frame boundaries or with `round(...)` instead of `floor(...)` for slightly cleaner alignment

- Important future TODO from chat: possible clean audio-grid experiment
  - a potentially cleaner alternative alignment regime would be:
    - keep `fps=25`
    - switch audio `hop_size` from `200` to `160`
    - this would produce `100` mel columns per second
    - that gives exactly `4` mel columns per video frame
    - then a `5`-frame SyncNet window would map exactly to `20` mel columns
  - this would remove the current fractional `3.2 mel/frame` alignment and the floor-quantization issue
  - however, it is not a small parameter tweak:
    - it would require a new mel format
    - `mel_step_size` would need to move from `16` to `20`
    - generator audio encoder assumptions would need to be updated
    - processed audio would need to be regenerated
    - comparisons to the current reference-style official teacher path would no longer be one-to-one compatible
  - conclusion from chat:
    - do **not** do this now in the main pipeline
    - only consider it as a separate experimental branch **after** the local SyncNet already starts beating the official teacher on the current compatible pipeline

- HDTF clip re-upload to Google Drive was completed successfully in small archives:
  - `hdtf_clips_batch_0000.tar` .. `hdtf_clips_batch_0007.tar` are now present in the shared Drive folder
  - local tar batches under `training/data/hdtf/archives_clips_batches_20260323/` were deleted after successful upload; only `batches_manifest.jsonl` and `uploaded_manifest.jsonl` remain
  - the original local source clips under `training/data/hdtf/clips/` were not deleted
  - the old monolithic `hdtf_clips_20260321.tar` still exists on Drive alongside the new batch archives
- Google Drive folder structure was verified directly via `rclone lsf` (no download):
  - `hdtf_clips_20260321.tar`
  - `talkvid_raw_batch_0000.tar` .. `talkvid_raw_batch_0006.tar`
  - no subdirectories and no processed archives are present right now
- Added safer HDTF re-upload support in smaller units:
  - `training/scripts/package_media_batches.py`
  - `training/scripts/upload_archives_no_cleanup.py`
  - intended use is to split `training/data/hdtf/clips` into `hdtf_clips_batch_000N.tar` archives of about `1 GB` each and upload them without deleting the local clips
- Real smoke test on current HDTF clips:
  - batch `0000`: `49` files, `0.979 GB`
  - batch `0001`: `38` files, `0.920 GB`
  - this confirms the new packer behaves as expected for the current HDTF corpus
- The existing remote staging helper already accepts these future HDTF batch names because any archive classified as `hdtf` media is staged into `training/data/hdtf/clips`.

- Important correction to the remote night-run bootstrap:
  - the earlier assumption "Drive folder contains processed TalkVid" was wrong for the current workflow
  - TalkVid uploads to Google Drive are currently raw tar batches produced by `training/scripts/package_raw_batches.py` and uploaded by `training/scripts/upload_batches_and_cleanup.py`
  - those archives contain flat `*.mp4 + *.json` clip pairs, not `frames.npy/mel.npy` speaker directories
- The remote bootstrap was updated accordingly:
  - new staging helper: `training/scripts/download_training_inputs_public_gdrive.py`
  - it can stage:
    - `talkvid_raw` -> `training/data/talkvid_remote/raw`
    - `hdtf_clips` -> `training/data/hdtf/clips`
    - `hdtf_raw` -> `training/data/hdtf/raw`
    - processed sets, if they do happen to exist in the folder
  - classification is based on archive/folder names plus observed file structure
- `training/run_full_pipeline_hdtf_talkvid.sh` no longer assumes Drive already contains processed data:
  - if staged HDTF processed exists, it uses it directly
  - else if staged HDTF clips exist, it runs `preprocess_dataset.py`
  - else if staged HDTF raw exists, it runs `process_hdtf_incremental.py`
  - if staged TalkVid processed exists, it uses it directly
  - else if staged TalkVid raw exists, it runs `process_talkvid_incremental.py` with the requested head-filter preset before training
- The launcher now writes an effective config at runtime so `talkvid_root` matches the selected preset root instead of being hardcoded.
- Local validation after the correction:
  - `download_training_inputs_public_gdrive.py` passed a synthetic smoke test for:
    - `talkvid_raw_batch_0000.tar`
    - `hdtf_clips_20260321.tar`
    - `hdtf_processed.tar`
  - `bash -n` passed again for the updated run/watch scripts

- Added remote mixed-training bootstrap for tonight's HDTF + TalkVid run:
  - `training/configs/lipsync_cuda3090_hdtf_talkvid.yaml`
  - `training/run_full_pipeline_hdtf_talkvid.sh`
  - `training/watch_vast_3090_sync_and_launch_hdtf_talkvid_20260323.sh`
- Important architecture change for training:
  - the training stack now supports a separate `talkvid_root` alongside `hdtf_root`
  - shared helper: `training/scripts/dataset_roots.py`
  - wired into `train_generator.py`, `train_syncnet.py`, `check_audio_sensitivity.py`, and `training/configs/default.yaml`
- Remote dataset ingest is now scripted from the public Google Drive folder shared in chat:
  - helper: `training/scripts/download_preprocessed_public_gdrive.py`
  - default folder URL: `https://drive.google.com/drive/folders/1v06momk8fR-eqw79Z93zczBx_InWsCS9?usp=drive_link`
  - expects public-folder download via `gdown`
  - classifies archives/folders by name and places speakers into:
    - `training/data/hdtf/processed`
    - `training/data/talkvid/processed`
    - `training/data/talkvid/processed_soft`
    - `training/data/talkvid/processed_medium`
    - `training/data/talkvid/processed_strict`
  - default required set for the night run is `hdtf,talkvid_medium`
- Added a no-data code sync path for remote deploy:
  - `training/sync_to_vast_3090_nodata_20260323.sh`
  - unlike the earlier sync script, it excludes all local `training/data/` so we do not re-upload local HDTF clips / TalkVid raw to the remote box
- Local validation completed:
  - `py_compile` passed for updated Python scripts
  - `bash -n` passed for the new remote launch scripts
  - `download_preprocessed_public_gdrive.py` passed a local smoke test against synthetic `hdtf_processed.tar` and `talkvid_medium_processed.tar`

- TalkVid dataset generation now supports head-pose quality presets without changing fetch behavior:
  - fetch remains unrestricted so the raw corpus stays maximally reusable for later experiments
  - `training/scripts/process_talkvid_incremental.py` now accepts `--head-filter-mode {off,soft,medium,strict}`
  - filtering happens at generation time from raw clip sidecars (`head_detail.scores`)
- Current preset thresholds:
  - `soft`: `avg_orientation >= 90`, `min_orientation >= 82`
  - `medium`: `avg_orientation >= 92`, `min_orientation >= 84`
  - `strict`: `avg_orientation >= 94`, `min_orientation >= 88`
- Verified on the current TalkVid metadata snapshot:
  - `soft`: `135239 / 181752` clips (`74.4%`)
  - `medium`: `114717 / 181752` clips (`63.1%`)
  - `strict`: `66165 / 181752` clips (`36.4%`)
- Recommended workflow for comparisons:
  - keep one shared `raw/`
  - generate separate processed outputs per preset, e.g. `processed_soft`, `processed_medium`, `processed_strict`

- Processing pipeline was aligned with the intended format:
  - recommended path is now `raw -> normalized 25fps/16kHz clips -> processed frames+mel`
  - runtime fps normalization in `training/data/dataset.py` is now treated as a legacy fallback for older processed sets
- Added reusable clip normalization helper:
  - `training/scripts/transcode_video.py`
  - supports explicit `--video-encoder` selection for the normalization stage
  - currently supported choices: `auto`, `libx264`, `h264_videotoolbox`, `h264_nvenc`
  - `auto` now probes hardware encoder usability and falls back to `libx264` if the advertised hardware path is not actually usable
- Added generic incremental processing path for TalkVid:
  - `training/scripts/process_talkvid_incremental.py`
  - intended usage per batch:
    - `--raw-dir training/data/talkvid_local/batches/batch_NNNN/raw`
    - `--clips-dir training/data/talkvid_local/batches/batch_NNNN/clips_25fps`
    - `--processed-dir training/data/talkvid_local/processed`
- Refactored HDTF processing scripts onto the same normalization stage:
  - `training/scripts/process_hdtf_incremental.py`
  - `training/scripts/process_hdtf_pending_once.py`
  - `training/scripts/download_hdtf.py`
- Important local GPU nuance:
  - this macOS host advertises `h264_videotoolbox` in ffmpeg, but a live probe failed with `cannot create compression session: -12908`
  - because of that, `--video-encoder auto` currently resolves to `libx264` here instead of forcing a broken hardware path
- Smoke test completed successfully with the project Python environment:
  - raw clip: `training/data/talkvid_local/batches/batch_0006/raw/videovideo0AvHEwYHsCo-scene57-scene1.mp4`
  - normalized output was verified as `25 fps`, `16 kHz`, mono
  - processed sample produced `frames.npy`, `mel.npy`, `bbox.json`

- Local orchestration was switched from a shared-`raw/` live pack loop to per-run batch directories under `training/data/talkvid_local/batches/`.
- The previous legacy partial `training/data/talkvid_local/raw/` was sealed as `talkvid_raw_batch_0005.tar` and moved under `batches/batch_0005/raw/` for cleanup tracking.
- Batch numbering was aligned with archive numbering:
  - sealed legacy partial: `batch_0005`
  - current live fetch run: `batch_0006`
- Current live state during this update:
  - launcher pid: `40116`
  - uploader pid: `40133`
  - downloader pid: `40138`
  - current upload: `talkvid_raw_batch_0005.tar`
  - current fetch root: `training/data/talkvid_local/batches/batch_0006`
  - `batch_0006/raw` had `13` clips at `2026-03-23 14:28:11 CET`
- Important behavior change:
  - upload now starts in the background immediately
  - fetch starts or resumes immediately in the current `batch_n` directory without waiting for upload to finish
  - cleanup for newly sealed batches uses the manifest-recorded `raw_dir` / `batch_root`, so deleting a finished batch cannot touch files from the next batch directory

- The live cycle is still healthy. `cycle.log` is producing fresh `OK` entries through `2026-03-23 14:00:05 CET`.
- Current live state during this update:
  - launcher pid: `98647`
  - downloader pid: `98663`
  - raw partial batch state: `ready_files=1448`, `size_gb=0.888`
- A follow-up code patch was applied for the next launch:
  - `training/scripts/download_talkvid.py` now records a short failure `detail` in the manifest/log for `yt_dlp_error`-style failures
  - `training/scripts/download_talkvid.py` now supports cooldown/resume after `rate_limited` via `--rate-limit-cooldown-seconds` and `--max-rate-limit-cooldowns`
  - `training/run_local_talkvid_batch_cycle.sh` now wires those settings from `TALKVID_RATE_LIMIT_COOLDOWN_SECONDS` and `TALKVID_MAX_RATE_LIMIT_COOLDOWNS`
- Post-14:16 local behavior change:
  - `training/run_local_talkvid_batch_cycle.sh` now acts as a continuous local orchestrator instead of a single-run supervisor
  - default `TALKVID_TARGET_ADDITIONAL_GB` is now `0.0`, so fetch no longer stops every 1 GB just because a batch was packaged/uploaded
  - if `download_talkvid.py` exits with `rc=10` because an explicit per-run target was reached, the launcher now immediately starts the next fetch run instead of treating that as the end of the whole cycle
- Important nuance: the currently running downloader pid `98663` already has the old Python code loaded in memory. The new cooldown/diagnostic behavior will apply on the next launch, not retroactively to the already-running process.

## Scope

This handoff covers the local `TalkVid` fetch/package/upload loop in this repo, especially:
- keeping fetch alive while package/upload runs
- parallel `yt-dlp` fetches
- better logging with timestamps
- distinguishing YouTube rate limiting from real unavailable videos
- resuming the run with cookies exported from Safari

## Current status

The pipeline is currently running and healthy.

Active runtime:
- `caffeinate` session id: `42509`
- fetch pid from the launcher log: `98663`
- current downloader command:

```bash
python3 training/scripts/download_talkvid.py \
  --output /Users/semenvlasov/Documents/repos/lipsync_test/training/data/talkvid_local \
  --variant with_captions \
  --max-height 720 \
  --min-duration 4.0 \
  --max-duration 6.5 \
  --min-width 720 \
  --min-height 720 \
  --min-dover 8.0 \
  --min-cotracker 0.90 \
  --target-additional-gb 1.0 \
  --min-free-gb 5.0 \
  --timeout 300 \
  --skip-manifest /Users/semenvlasov/Documents/repos/lipsync_test/training/data/talkvid_local/download_manifest.jsonl \
  --skip-manifest /Users/semenvlasov/Documents/repos/lipsync_test/training/data/talkvid_local/archives/batches_manifest.jsonl \
  --skip-manifest /Users/semenvlasov/Documents/repos/lipsync_test/training/data/talkvid_local/archives/uploaded_manifest.jsonl \
  --cookies-file /tmp/talkvid_local_cookies.txt \
  --jobs 4
```

Recent health signal:
- fresh `OK` downloads are appearing in `training/output/talkvid_local_cycle/cycle.log`
- latest successful clips at handoff time were around `2026-03-23 13:11:03 CET`
- package/upload loop is alive and currently leaving the partial tail unsealed until it reaches the batch threshold

## Important files

Main code:
- `training/scripts/download_talkvid.py`
- `training/scripts/package_raw_batches.py`
- `training/scripts/upload_batches_and_cleanup.py`
- `training/run_local_talkvid_batch_cycle.sh`

State and logs:
- `training/output/talkvid_local_cycle/cycle.log`
- `training/data/talkvid_local/download_manifest.jsonl`
- `training/data/talkvid_local/raw/`
- `training/data/talkvid_local/archives/`

Backup made before rollback of rate-limited failures:
- `training/output/talkvid_local_cycle/download_manifest_backup_pre_rate_limit_20260323_1250.jsonl`

Local cookies file currently in use:
- `/tmp/talkvid_local_cookies.txt`

Source Safari export used to build that file:
- `/Users/semenvlasov/Downloads/Cookies.binarycookies`

## What was changed

### 1. Fetch no longer pauses for upload

`training/run_local_talkvid_batch_cycle.sh` was reworked into a producer/consumer loop:
- downloader runs in the background
- package and upload loop runs in parallel
- final package/upload happens after fetch exits

### 2. Added parallel YouTube fetches

`training/scripts/download_talkvid.py` now supports:
- `--jobs`
- a `ThreadPoolExecutor`
- refill-on-first-completion behavior

This means it does not wait for a group of 4 to finish before starting the next one. It keeps up to `jobs` downloads in flight and submits a new one as soon as any slot frees up.

### 3. Added timestamped logging

Timestamp prefixes were added to:
- `training/run_local_talkvid_batch_cycle.sh`
- `training/scripts/download_talkvid.py`
- `training/scripts/package_raw_batches.py`
- `training/scripts/upload_batches_and_cleanup.py`

### 4. Fixed package/fetch race

`download_talkvid.py` now publishes a clip atomically:
- writes sidecar JSON first into a temp location
- publishes JSON
- publishes MP4 last

`package_raw_batches.py` now only packages ready `mp4 + json` pairs.

### 5. Prevented upload/resume from losing progress

The `target_additional_gb` accounting in `download_talkvid.py` now tracks bytes downloaded in the current run instead of recomputing progress from current raw-dir size after cleanup.

### 6. Source-level block for real dead videos

If a clip fails with:
- `video_unavailable`
- `private_video`

then future clips from the same source `video_key` are skipped in later scheduling.

This is intentionally not used for rate-limit errors.

### 7. Rate-limit detection added

`download_talkvid.py` now explicitly distinguishes YouTube session throttling:
- if `yt-dlp` stderr contains `rate-limited by YouTube`
- it records reason `rate_limited`
- it stops submitting new fetches for the current run
- it exits with `rc=21`

This was added after verifying that several supposed `video_unavailable` cases were actually YouTube session rate limits.

### 8. Failed clips are no longer treated as completed

Resume logic was fixed so `status=fail` entries in manifests do not count as completed clips.

## Rate-limit investigation and rollback

What happened:
- increasing to `6` jobs caused a wave of failures
- at first they looked like `video_unavailable`
- direct `yt-dlp` probing showed the real message was YouTube session rate limiting

Action taken:
- stopped the run
- rolled back the recent `status=fail, reason=video_unavailable` entries that belonged to the throttled period
- kept a backup manifest before the rewrite
- reduced default parallelism back to `4`

Important nuance:
- real `private_video` entries were not rolled back
- rate-limited failures should now be labeled `rate_limited` instead of `video_unavailable`

## Cookies flow

Safari browser-cookie access through live browser APIs was blocked by macOS permissions in this environment, so the working path was:

1. export/copy Safari `Cookies.binarycookies`
2. convert it locally with Python `browser_cookie3`
3. save as Netscape cookies file at `/tmp/talkvid_local_cookies.txt`
4. run `yt-dlp` using `--cookies-file`

The converted cookies file was validated with a live command before restarting the run.

## Current observed log snapshot

At handoff time, `cycle.log` shows healthy progress such as:
- `2026-03-23 13:10:57 CET [TalkVid] [243] ... OK`
- `2026-03-23 13:11:02 CET [TalkVid] [244] ... OK`
- `2026-03-23 13:11:03 CET [TalkVid] [247] ... OK`

The manifest tail also shows fresh `status=ok` entries for the same period.

## Resume / operator commands

Watch the live log:

```bash
tail -f /Users/semenvlasov/Documents/repos/lipsync_test/training/output/talkvid_local_cycle/cycle.log
```

Check the active fetch pid from the log:

```bash
rg -n "fetch pid=" /Users/semenvlasov/Documents/repos/lipsync_test/training/output/talkvid_local_cycle/cycle.log | tail
```

Check whether the current downloader pid is still alive:

```bash
ps -p 98663 -o pid,ppid,%cpu,%mem,etime,state,command
```

Restart the loop manually with the current cookie file:

```bash
caffeinate -dimsu /bin/zsh /Users/semenvlasov/Documents/repos/lipsync_test/training/run_local_talkvid_batch_cycle.sh
```

If cookies need to be rebuilt from a copied Safari export:

```bash
python3 - <<'PY'
import browser_cookie3
from http.cookiejar import MozillaCookieJar
src = '/Users/semenvlasov/Downloads/Cookies.binarycookies'
out = '/tmp/talkvid_local_cookies.txt'
jar = browser_cookie3.safari(cookie_file=src)
mj = MozillaCookieJar(out)
for cookie in jar:
    mj.set_cookie(cookie)
mj.save(ignore_discard=True, ignore_expires=True)
print(out, len(list(mj)))
PY
```

Validate the cookies file with `yt-dlp`:

```bash
yt-dlp --cookies /tmp/talkvid_local_cookies.txt --skip-download --no-playlist --print '%(id)s|%(title)s' 'https://www.youtube.com/watch?v=FtbbWeoL2vQ'
```

## Known limitations

- Rate-limit detection depends on `yt-dlp` exposing recognizable stderr text. If YouTube changes the message, it may still surface as `yt_dlp_error` or even `video_unavailable`.
- Source blocking is only for truly fatal source states, not for rate limiting.
- With `jobs=4`, up to 4 in-flight clips from a newly bad source can still fail before the source block kicks in.
- The worktree is dirty in many unrelated areas. Do not revert unrelated user changes while continuing this task.

## Suggested next steps

- Keep watching whether the Safari-derived cookies stay healthy at `jobs=4`
- If rate limiting returns, add automatic backoff or cooldown handling for `rate_limited`
- Optionally cap in-flight clips per `video_key` to `1` to reduce waste on newly dead sources

## Prompt for the next account/chat

Use this as the first message in the new chat:

```text
Continue the TalkVid fetch work in /Users/semenvlasov/Documents/repos/lipsync_test.
First read /Users/semenvlasov/Documents/repos/lipsync_test/docs/handoffs/2026-03-23-talkvid-youtube-fetch.md.
Then inspect /Users/semenvlasov/Documents/repos/lipsync_test/training/output/talkvid_local_cycle/cycle.log and continue from the current runtime state without reverting unrelated changes.
```
