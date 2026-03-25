# Corp Account Transfer Handoff

Updated: `2026-03-23 13:13:15 CET`

Related handoffs:
- [TalkVid fetch details](./2026-03-23-talkvid-youtube-fetch.md)
- Remote recovery notes live alongside the compact backup bundle at `training/backup/compact_remote_state_20260322/RECOVERY_20260322.md`.

## Scope

This handoff is the high-level transfer point for continuing the lipsync work on a corporate account.

It covers:
- current best training artifacts from the last 3090 cloud run
- how to restore that state onto a fresh cloud machine
- current local `TalkVid` fetch/package/upload runtime
- recent preprocess changes for faster cloud preprocessing
- the main known limitations and the safest next steps

## Current status

### 1. Remote training machine

The previous Vast 3090 machine was intentionally destroyed after compact backup.

What was preserved:
- best generator checkpoints and logs
- best local SyncNet checkpoints and logs
- generator teacher-ablation summaries

Compact backup root:
- `training/backup/compact_remote_state_20260322`

Restore script:
- `training/backup/compact_remote_state_20260322/restore_compact_remote_state_20260322.sh`

### 2. Best current model choices

Current winners from the last cloud run:
- generator winner: `training/backup/compact_remote_state_20260322/generator_full/generator_epoch008.pth`
- late generator checkpoints kept for comparison:
  - `training/backup/compact_remote_state_20260322/generator_full/generator_epoch018.pth`
  - `training/backup/compact_remote_state_20260322/generator_full/generator_epoch019.pth`
- best local SyncNet candidate:
  - `training/backup/compact_remote_state_20260322/syncnet_full_minus10/syncnet_epoch011.pth`

Important conclusion from ablations:
- official `official SyncNet` still beats the local SyncNet downstream as the teacher for generator training
- local SyncNet improved, but was not yet the better teacher for generator training

### 3. Local TalkVid fetch runtime

The local `TalkVid` fetch is currently running and healthy.

Active runtime snapshot:
- launcher pid: `98647`
- `caffeinate` pid: `98648`
- downloader pid: `98663`

Recent health signal:
- fresh `OK` lines are appearing in `training/output/talkvid_local_cycle/cycle.log`
- latest visible successful clips at handoff time were around `2026-03-23 13:13 CET`
- package/upload loop is alive and currently leaves the small partial tail unsealed until it reaches the batch threshold

## What changed recently

### 1. Preprocess was upgraded for cloud use

Files:
- `training/scripts/preprocess_dataset.py`
- `training/scripts/process_hdtf_incremental.py`

Changes:
- added multi-worker incremental preprocessing over videos
- added detector backends:
  - `opencv`
  - `sfd`
- added detector device selection:
  - `auto`
  - `cpu`
  - `cuda`
  - `mps`
- added `ffmpeg_threads` control
- added thread-local detector caching
- added a safety guard so `sfd` does not run multi-threaded in the local thread-fallback path

Practical guidance now written in code comments:
- local CPU-only iteration: prefer `opencv + workers=4`
- remote CUDA box:
  - fast path: `opencv + process workers`
  - quality path: `sfd + cuda + 1 worker`

Important nuance:
- do not blindly run `sfd + cuda` across many process workers on one GPU; each process would otherwise load its own detector model on the GPU

### 2. TalkVid downloader handles external interrupts more cleanly

File:
- `training/scripts/download_talkvid.py`

Change:
- if the process receives an external `KeyboardInterrupt`, it now exits cleanly without a huge traceback
- already downloaded raw clips are preserved for resume
- wrapper behavior remains compatible with package/upload after interrupted download stages

### 3. Local TalkVid fetch pipeline remains the active data-growth path

Main files:
- `training/scripts/download_talkvid.py`
- `training/scripts/package_raw_batches.py`
- `training/scripts/upload_batches_and_cleanup.py`
- `training/run_local_talkvid_batch_cycle.sh`

Detailed fetch-specific notes are in:
- [2026-03-23-talkvid-youtube-fetch.md](./2026-03-23-talkvid-youtube-fetch.md)

## Important files

### Recovery and preserved cloud artifacts

- `training/backup/compact_remote_state_20260322/RECOVERY_20260322.md`
- `training/backup/compact_remote_state_20260322/restore_compact_remote_state_20260322.sh`
- `training/backup/compact_remote_state_20260322/generator_full/`
- `training/backup/compact_remote_state_20260322/syncnet_full_minus10/`
- `training/backup/compact_remote_state_20260322/ablation3_official/`
- `training/backup/compact_remote_state_20260322/ablation3_local09/`
- `training/backup/compact_remote_state_20260322/ablation3_local11/`

### Current local TalkVid state

- `training/output/talkvid_local_cycle/cycle.log`
- `training/data/talkvid_local/download_manifest.jsonl`
- `training/data/talkvid_local/raw/`
- `training/data/talkvid_local/archives/`
- `/tmp/talkvid_local_cookies.txt`

### Main preprocess code

- `training/scripts/preprocess_dataset.py`
- `training/scripts/process_hdtf_incremental.py`

## Recommended operator commands

Watch the local TalkVid fetch:

```bash
tail -f training/output/talkvid_local_cycle/cycle.log
```

Restore the preserved remote state onto a fresh machine:

```bash
training/backup/compact_remote_state_20260322/restore_compact_remote_state_20260322.sh <host> <port> /root
```

Manually restart the local TalkVid loop:

```bash
caffeinate -dimsu /bin/zsh training/run_local_talkvid_batch_cycle.sh
```

## Known risks and open questions

- official `official SyncNet` is still the safer teacher for generator training; do not switch to local SyncNet as default without a new downstream ablation
- local `SFD` preprocessing on CPU is not a meaningful benchmark for cloud throughput; its target mode is remote CUDA
- if preprocess is run on a new CUDA machine, compare:
  - `opencv + 4 process workers`
  - `sfd + cuda + 1 process worker`
  before choosing the default
- the local TalkVid fetch depends on the health of the current cookies file at `/tmp/talkvid_local_cookies.txt`
- the worktree may contain unrelated user changes; do not revert unrelated files while continuing this work

## Suggested next steps on the corp account

1. Read this handoff and the detailed TalkVid fetch handoff.
2. If a new cloud machine is available, restore the compact remote training state with the provided script.
3. Keep the local TalkVid fetch running as the ongoing source of new raw data.
4. On the next cloud machine, benchmark preprocess in the two practical modes:
   - `opencv/process x4`
   - `sfd/cuda x1`
5. Continue generator training with the official teacher unless a fresh downstream ablation proves otherwise.

## Prompt for the next account/chat

Use this as the first message in the new chat:

```text
Continue the lipsync work in this repository checkout.
First read docs/handoffs/2026-03-23-corp-account-transfer.md.
Then read docs/handoffs/2026-03-23-talkvid-youtube-fetch.md.
After that:
1. inspect the current TalkVid runtime in training/output/talkvid_local_cycle/cycle.log
2. inspect the compact remote backup in training/backup/compact_remote_state_20260322
3. continue without reverting unrelated user changes.
```
