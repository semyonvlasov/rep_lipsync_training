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
lipsyncctl train-syncnet
lipsyncctl train-generator-gan
lipsyncctl benchmark --device cuda
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
