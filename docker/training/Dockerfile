FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG RCLONE_RELEASE=1.73.3
ARG FACE_PROCESSING_GIT_URL=https://github.com/semyonvlasov/face_processing.git
ARG FACE_PROCESSING_REF=4223501956614ccd878e398830c8d5d9e7fcc317

ENV LIPSYNC_REPO_ROOT=/opt/lipsync \
    FACE_PROCESSING_REPO_ROOT=/opt/face_processing \
    PYTHONPATH=/opt/face_processing:/opt/lipsync:/opt/lipsync/training \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        make \
        python-is-python3 \
        python3 \
        python3-pip \
        python3-venv \
        rsync \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        "https://downloads.rclone.org/v${RCLONE_RELEASE}/rclone-v${RCLONE_RELEASE}-linux-amd64.zip" \
        -o /tmp/rclone.zip \
    && unzip -q /tmp/rclone.zip -d /tmp \
    && install -m 0755 "/tmp/rclone-v${RCLONE_RELEASE}-linux-amd64/rclone" /usr/local/bin/rclone \
    && rm -rf /tmp/rclone.zip "/tmp/rclone-v${RCLONE_RELEASE}-linux-amd64" \
    && rclone version

RUN git clone --filter=blob:none "${FACE_PROCESSING_GIT_URL}" "${FACE_PROCESSING_REPO_ROOT}" \
    && git -C "${FACE_PROCESSING_REPO_ROOT}" checkout --detach "${FACE_PROCESSING_REF}" \
    && test "$(git -C "${FACE_PROCESSING_REPO_ROOT}" rev-parse HEAD)" = "${FACE_PROCESSING_REF}"

WORKDIR /tmp/bootstrap
COPY training/requirements-server.txt /tmp/bootstrap/requirements-server.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
    && python3 -m pip install -r /tmp/bootstrap/requirements-server.txt

WORKDIR /opt/lipsync
COPY . /opt/lipsync

RUN chmod +x /opt/lipsync/docker/training/entrypoint.sh /opt/lipsync/training/scripts/lipsyncctl.py \
    && ln -sf /opt/lipsync/training/scripts/lipsyncctl.py /usr/local/bin/lipsyncctl \
    && mkdir -p /workspace/data /workspace/output /workspace/cache \
    && lipsyncctl list-artifacts >/dev/null

ENTRYPOINT ["/opt/lipsync/docker/training/entrypoint.sh"]
CMD ["doctor"]
