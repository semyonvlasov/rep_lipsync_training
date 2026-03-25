# Training Run Log

Git-tracked registry of training runs and uploaded Drive artifacts.

Artifacts root: https://drive.google.com/drive/folders/1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv?usp=sharing

## 2026-03-25 15:47:44 UTC `syncnet_epoch011_medium_b64_20260324`
- kind: `syncnet`
- local_output: `/root/lipsync_test/rep_lipsync_training/training/output/syncnet_best_epoch011_medium_b64_20260324`
- artifacts_root: [Google Drive](https://drive.google.com/drive/folders/1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv?usp=sharing)
- drive_subdir: `syncnet/syncnet_epoch011_medium_b64_20260324`
- manifest: [artifacts_upload_manifest.json](https://drive.google.com/open?id=14L88zRRpIRC9L23DKL0xNeyIGAsaGOPf)
- training_params:
- model: `img_size=96` `mel_steps=16` `base_channels=32` `predict_alpha=False`
- audio: `sr=16000` `hop=200` `n_mels=80` `n_fft=800`
- data: `fps=25` `crop=None` `workers=8` `materialize=None` `hdtf_root=data/hdtf/processed` `talkvid_root=None`
- syncnet: `epochs=12` `batch=64` `lr=0.0001` `T=5`
- benchmarks:
- selected_teacher: `winner=syncnet_epoch011` `kind=local` `pairwise_acc=0.8515625` `margin=0.36921833731594234`
- syncnet_compare: `holdout_total=10` `samples=256`
- teacher `official_wav2lip`: `pairwise_acc=0.7578125` `margin=0.19320448782391964` `shifted_acc=0.75` `foreign_acc=0.765625`
- teacher `syncnet_epoch011`: `pairwise_acc=0.8515625` `margin=0.36921833731594234` `shifted_acc=0.8359375` `foreign_acc=0.8671875`
- teacher `syncnet_full_minus10_epoch011`: `pairwise_acc=0.6953125` `margin=0.17301626406333526` `shifted_acc=0.67578125` `foreign_acc=0.71484375`
- checkpoints:
- [syncnet_epoch011.pth](https://drive.google.com/open?id=1n1Aa8BGcdbbUApcTHmk2lUg1axZM6eiD)
- top_reports:
- [pipeline.log](https://drive.google.com/open?id=1bZVA9QQKpVOHkV4wsFT-Eaw90W7Kl4t4)
- [syncnet_selected_teacher.json](https://drive.google.com/open?id=1duUqb4PSrG8aX6HMuMi2Fev8U5uR1SXs)
- [syncnet_selected_teacher_256_with_old.json](https://drive.google.com/open?id=1Ijc6dfam5VJb7SgyVwUc9DjOR803pnth)
- [syncnet_teacher_compare.json](https://drive.google.com/open?id=1oqIwnNl1QEbWWRTIGfgggt0dskGDEFcS)
- [syncnet_teacher_compare_256_with_old.json](https://drive.google.com/open?id=1gJmQc633okmF2MkGJePWSswgCLqzq02X)
