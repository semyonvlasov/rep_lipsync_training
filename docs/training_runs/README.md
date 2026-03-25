# Training Runs

This folder is the git-tracked registry for completed training runs.

- `log.md`
  Append-only markdown log with:
  Drive links, training parameters, and benchmark values.
- Per-run artifacts themselves do not live in git.
  They are uploaded to the shared Google Drive root:
  https://drive.google.com/drive/folders/1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv?usp=sharing

Recommended flow after each completed training run:

1. Upload the run output with:
   `make upload-training-artifacts ARTIFACTS_OUTPUT_DIR=training/output/<run_name> ARTIFACTS_CONFIG_PATH=training/configs/<config>.yaml`
2. Review the appended entry in `docs/training_runs/log.md`
3. Commit and push the updated git log
