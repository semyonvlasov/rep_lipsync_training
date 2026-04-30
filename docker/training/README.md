# Training Docker Image

Build and push the public runtime image:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/training/Dockerfile \
  -t semyonvlasov/lipsync-training:cuda12.8 \
  --push .
```

Preferred Vast AI flow: set the image when creating/renting the instance.
Find an offer first. By default the helper searches all hosts; add `--eu-only`
when a Europe-only shortlist is needed. `--extra-query` is passed into the
Vast search query, not filtered only after the first result page.

```bash
python3 training/scripts/search_vast_offers.py \
  --storage-gb 800 \
  --min-days 7 \
  --min-cuda 12.8 \
  --limit 20 \
  --extra-query 'gpu_name in [RTX_3090,RTX_4090]'

python3 training/scripts/search_vast_offers.py \
  --storage-gb 800 \
  --min-cuda 12.8 \
  --eu-only \
  --extra-query 'gpu_name in [RTX_3090,RTX_4090]'
```

```bash
vastai create instance <offer_id> \
  --image semyonvlasov/lipsync-training:cuda12.8 \
  --disk 800 \
  --ssh \
  --direct
```

In the Vast UI, use the same value in the Docker image field:
`semyonvlasov/lipsync-training:cuda12.8`.

After SSH connects to the instance/container, put `rclone.conf` on the host,
then run the image entrypoint directly so it prepares `/workspace` symlinks:

```bash
mkdir -p /root/.config/rclone
# copy or mount rclone.conf to /root/.config/rclone/rclone.conf
export RCLONE_CONFIG=/root/.config/rclone/rclone.conf

/opt/lipsync/docker/training/entrypoint.sh doctor
/opt/lipsync/docker/training/entrypoint.sh prepare --skip-existing
```

Alternative generic-host flow if the machine was rented with another image and
Docker is available inside the host:

```bash
docker run --gpus all --ipc=host --shm-size=32g --rm -it \
  -v /workspace/lipsync:/workspace \
  -v /root/.config/rclone/rclone.conf:/run/secrets/rclone.conf:ro \
  semyonvlasov/lipsync-training:cuda12.8 \
  prepare
```

Common commands:

```bash
lipsyncctl doctor --require-prepared
lipsyncctl merge-dataset --include-tier confident --include-tier medium
lipsyncctl prewarm-cache --max-items 512
lipsyncctl split-generator-dataset
lipsyncctl train-syncnet
lipsyncctl train-generator-gan
lipsyncctl benchmark --device cuda
```

After a full prewarm, export a portable dataset snapshot when you want to move
the materialized lazy cache and sync metadata to another instance. Keep the
training root stable at `/opt/lipsync/training` after import; lazy cache keys
include absolute source paths.

For Hugging Face uploads, provide auth through environment, not through command
arguments:

```bash
export HF_TOKEN=hf_...
hf auth whoami
```

From macOS, if the token is already in the clipboard, install it on the remote
instance without printing it:

```bash
pbpaste | LC_ALL=C tr -d '\n\r ' | ssh -p <port> root@<vast-host> '
  set -e
  install -d -m 700 /root/.cache/huggingface
  cat > /root/.cache/huggingface/token
  chmod 600 /root/.cache/huggingface/token
  hf auth whoami
'
```

Preferred export flow streams shards to a HF dataset repo. With
`--delete-uploaded-shards`, each local shard is removed only after a successful
upload, so peak disk usage is roughly the dataset plus one shard. The exporter
removes and recreates `--snapshot-dir` at startup; write nohup logs outside that
directory. The snapshot includes `prepared/`, `split/`, `merge_manifest.jsonl`,
`sync_alignment_manifest.jsonl`, `sample_index.jsonl`, `cache_index.jsonl`, and
tar shards. For immutable archives, put each snapshot into its own HF
branch/revision so later exports do not overwrite it. For the current active
dataset, upload to `main`.

```bash
lipsyncctl split-generator-dataset
lipsyncctl export-dataset-snapshot \
  --snapshot-dir output/dataset_snapshots/generator_tiltaware_YYYYMMDD \
  --shard-size-gb 20 \
  --hf-repo-id <namespace>/<dataset-repo> \
  --hf-create-repo \
  --hf-revision snapshot-YYYYMMDD \
  --hf-create-branch \
  --hf-private \
  --delete-uploaded-shards

hf download <namespace>/<dataset-repo> \
  --type dataset \
  --revision snapshot-YYYYMMDD \
  --local-dir /opt/lipsync/training/output/dataset_snapshots/generator_tiltaware_YYYYMMDD
lipsyncctl import-dataset-snapshot \
  --snapshot-dir output/dataset_snapshots/generator_tiltaware_YYYYMMDD
```

Current active dataset upload example:

```bash
ssh -p <port> root@<vast-host> '
  set -e
  cd /opt/lipsync/training
  mkdir -p output/dataset_snapshots
  nohup lipsyncctl export-dataset-snapshot \
    --snapshot-dir output/dataset_snapshots/generator_tiltaware_current_YYYYMMDD \
    --hf-repo-id semyonvlasov/x96_lipsync_tilt \
    --hf-revision main \
    --delete-uploaded-shards \
    > output/dataset_snapshots/generator_tiltaware_current_YYYYMMDD_export_hf.log 2>&1 &
'

ssh -p <port> root@<vast-host> \
  'tail -f /opt/lipsync/training/output/dataset_snapshots/generator_tiltaware_current_YYYYMMDD_export_hf.log'
```

If you intentionally want a full local snapshot first, omit `--hf-repo-id` from
`export-dataset-snapshot`, then upload the resulting folder separately:

```bash
lipsyncctl upload-dataset-snapshot-hf \
  --repo-id <namespace>/<dataset-repo> \
  --snapshot-dir output/dataset_snapshots/generator_tiltaware_YYYYMMDD \
  --create-repo \
  --revision snapshot-YYYYMMDD \
  --create-branch \
  --private
```

Private/current-best checkpoints are intentionally downloaded at runtime by
`lipsyncctl prepare` through `rclone backend copyid`; they are not baked into
the public DockerHub image.

The image clones `face_processing` from the public external repository at
build time into `/opt/face_processing`, exposes it through
`FACE_PROCESSING_REPO_ROOT`/`PYTHONPATH`, and does not vendor
`face_processing/` inside `/opt/lipsync`.

Troubleshooting:

If `prepare` fails with `command "copyid" failed: command not found`, the
container has an old apt-packaged rclone. Pull the latest image or update rclone
inside the current instance before rerunning `prepare`:

```bash
curl https://rclone.org/install.sh | bash
rclone backend help gdrive: | grep copyid
```
