# Training Docker Image

Build and push the public runtime image:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/training/Dockerfile \
  -t <dockerhub-user>/rep-lipsync-training:cuda12.8 \
  --push .
```

Run on a Vast AI host with runtime secrets and persistent workspace mounts:

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
