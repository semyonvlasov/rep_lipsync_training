# Training Docker Image

Build and push the public runtime image:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/training/Dockerfile \
  -t <dockerhub-user>/rep-lipsync-training:cuda12.8 \
  --push .
```

Preferred Vast AI flow: set the image when creating/renting the instance.

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
  <dockerhub-user>/rep-lipsync-training:cuda12.8 \
  prepare
```

Common commands:

```bash
lipsyncctl doctor --require-prepared
lipsyncctl merge-dataset --include-tier confident --include-tier medium
lipsyncctl prewarm-cache --max-items 512
lipsyncctl train-syncnet
lipsyncctl train-generator
lipsyncctl train-generator-gan
lipsyncctl benchmark --device cuda
```

Private/current-best checkpoints are intentionally downloaded at runtime by
`lipsyncctl prepare` through `rclone backend copyid`; they are not baked into
the public DockerHub image.
