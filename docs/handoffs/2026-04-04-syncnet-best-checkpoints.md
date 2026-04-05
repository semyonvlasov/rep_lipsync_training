# SyncNet Best Checkpoints As Of 2026-04-04

## Scope

This handoff fixes the answer to:
- which checkpoint is the best `official-like` SyncNet right now
- which checkpoint is the best on our pairwise eval
- where the uploaded artifacts and launch config live

## Run

Finished run:
- run dir:
  `training/output/training_cuda3090_syncnet_mirror_medium_20260404`
- saved config:
  `training/output/training_cuda3090_syncnet_mirror_medium_20260404/syncnet_mirror_cuda3090_medium_20260404.yaml`

Training shape:
- model: `mirror`
- `img_size=96`
- `mel_steps=16`
- `base_channels=32`
- `batch_size=64`
- `lr=1e-4`
- `T=5`
- official-like eval cadence:
  - `eval_interval_steps=12000`
  - `eval_batches=1400`
- our eval cadence:
  - `pairwise_eval_samples=200`
  - `eval_seed=20260329`
  - `pairwise_eval_seed=20260329`
- data:
  - `hdtf_root=data/hdtf/processed`
  - `talkvid_root=data/talkvid/processed`

## Winner

Best `official-like` checkpoint today:
- file:
  `syncnet_best_off_eval.pth`
- step:
  `96000`
- path:
  `training/output/training_cuda3090_syncnet_mirror_medium_20260404/syncnet/syncnet_best_off_eval.pth`

Official baseline from this run:
- `loss=0.8062`
- `acc=0.769`

Best official-like result from this run:
- `loss=0.4156`
- `acc=0.820`

Delta vs official baseline:
- `loss_delta=-0.3906`
- `acc_delta=+0.0510`

## Our Eval Winner

Best checkpoint on our pairwise eval:
- file:
  `syncnet_best_our_eval.pth`
- step:
  `192000`
- path:
  `training/output/training_cuda3090_syncnet_mirror_medium_20260404/syncnet/syncnet_best_our_eval.pth`

Official our-eval baseline:
- `pairwise_acc=0.895`
- `margin=0.4268`
- `shift_acc=0.885`
- `foreign_acc=0.905`

Best our-eval result from this run:
- `pairwise_acc=0.925`
- `margin=0.4595`
- `shift_acc=0.925`
- `foreign_acc=0.925`

Delta vs official our-eval baseline:
- `pairwise_acc_delta=+0.030`
- `margin_delta=+0.0328`
- `shift_acc_delta=+0.040`
- `foreign_acc_delta=+0.020`

## Upload

Uploaded to Google Drive checkpoints root:
- folder id:
  `1YCxup3Qop6iwG9uR-9UIy3zvXWVoggN3`
- run subfolder:
  `syncnet_mirror_medium_20260404`

Uploaded files:
- `syncnet_best_off_eval.pth`
- `syncnet_best_our_eval.pth`
- `syncnet_latest.pth`
- `launch_config.yaml`
- `launch_config.json`
- `split_summary.json`

Important convention:
- for any future question about the best `official-like` SyncNet as of this handoff, use `syncnet_best_off_eval.pth` from `syncnet_mirror_medium_20260404`
- do not confuse it with `syncnet_best_our_eval.pth`, which is the pairwise-eval winner

## Notes

- The dataset fix materially improved both the official baseline and our trained model.
- This run beat official on both:
  - official-like eval
  - our pairwise eval
- If a downstream pipeline wants the most conservative teacher aligned with official evaluation, use:
  - `syncnet_best_off_eval.pth`
- If a downstream experiment wants the strongest pairwise separation on our eval, use:
  - `syncnet_best_our_eval.pth`

## Next Chat Prompt

Use `rep_lipsync_training/docs/handoffs/2026-04-04-syncnet-best-checkpoints.md` as the source of truth for the current best SyncNet checkpoints. Treat `syncnet_best_off_eval.pth` from `syncnet_mirror_medium_20260404` as the best official-like model, and `syncnet_best_our_eval.pth` as the best on our pairwise eval.
