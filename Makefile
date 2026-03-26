SHELL := /bin/bash

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TRAINING_ROOT := $(REPO_ROOT)/training
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SERVER_PY_REQUIREMENTS := $(TRAINING_ROOT)/requirements-server.txt

SMOKE_LAZY_WORKFLOW ?= workflows/train/run_lazy_smoke_remote_20260325.sh
SYNCNET_CONFIG ?= configs/syncnet_cuda3090_medium.yaml
GENERATOR_CONFIG ?= configs/lipsync_cuda3090_hdtf_talkvid.yaml
PREWARM_CONFIG ?= configs/syncnet_cuda3090_medium.yaml
PREWARM_SPEAKER_LIST ?=
PREWARM_LOG_EVERY ?= 100
PREWARM_MAX_ITEMS ?=
SYSTEM_WATCH_DIR ?= $(TRAINING_ROOT)/output/system_observe
SYSTEM_WATCH_INTERVAL ?= 10
SYNCNET_TEACHER ?= ../../models/wav2lip/checkpoints/lipsync_expert.pth
OFFICIAL_SYNCNET_PATH ?= ../../models/wav2lip/checkpoints/lipsync_expert.pth
SYNCNET_RESUME ?=
GENERATOR_RESUME ?=
SPEAKER_LIST ?=
SYNCNET_OUTPUT_DIR ?=
SYNCNET_WATCH_PID ?=
GENERATOR_TEMPLATE_CONFIG ?= configs/lipsync_cuda3090_hdtf_talkvid.yaml
GENERATOR_BATCH_SIZE ?= 8
GENERATOR_EPOCHS ?=
GENERATOR_OUTPUT_DIR ?=
GENERATOR_LAZY_CACHE_ROOT ?=
COMPARE_SAMPLES ?= 200
COMPARE_SEED ?= 123
COMPARE_DEVICE ?= cuda
WATCH_POLL_SECONDS ?= 60
ARTIFACTS_OUTPUT_DIR ?=
ARTIFACTS_RUN_KIND ?= auto
ARTIFACTS_RUN_NAME ?=
ARTIFACTS_CONFIG_PATH ?=
ARTIFACTS_REMOTE ?= gdrive:
ARTIFACTS_DRIVE_ROOT_ID ?= 1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv
ARTIFACTS_LOG_PATH ?= docs/training_runs/log.md
ARTIFACTS_INCLUDE_MEDIA ?= 0
ARTIFACTS_DRY_RUN ?= 0
BENCH_FACE ?=
BENCH_AUDIO ?=
BENCH_CHECKPOINT ?=
BENCH_OUTFILE ?=
BENCH_DEVICE ?= auto
BENCH_DETECTOR_DEVICE ?= auto
BENCH_BATCH_SIZE ?= 32
BENCH_FACE_DET_BATCH_SIZE ?= 4
BENCH_PADS ?= 0 10 0 0
BENCH_RESIZE_FACTOR ?= 1
BENCH_STATIC ?= 0
BENCH_NOSMOOTH ?= 0
BENCH_S3FD_PATH ?=

.PHONY: help server-setup smoke-lazy train-syncnet train-generator prewarm-syncnet-cache observe-system watch-syncnet-generator bench-wav2lip upload-training-artifacts

help:
	@echo "Available targets:"
	@echo "  make server-setup    # install apt + pip deps for remote Linux/Vast training"
	@echo "  make smoke-lazy      # run the lazy dataset smoke workflow"
	@echo "  make train-syncnet   # run scripts/train_syncnet.py with \$$SYNCNET_CONFIG"
	@echo "  make train-generator # run scripts/train_generator.py with \$$GENERATOR_CONFIG"
	@echo "  make prewarm-syncnet-cache # pre-materialize lazy frames/mels into the configured cache root"
	@echo "  make observe-system  # start a background CPU/RAM/GPU/VRAM/network observer"
	@echo "  make watch-syncnet-generator # wait for SyncNet, benchmark all epochs vs official, then launch generator"
	@echo "  make bench-wav2lip   # run the official Wav2Lip benchmark path (SFD + 96x96 generator)"
	@echo "  make upload-training-artifacts # upload a finished run to Drive and append the git log"
	@echo ""
	@echo "Useful overrides:"
	@echo "  PYTHON=python3"
	@echo "  SYNCNET_CONFIG=configs/syncnet_cuda3090_medium.yaml"
	@echo "  GENERATOR_CONFIG=configs/lipsync_cuda3090_hdtf_talkvid.yaml"
	@echo "  SYNCNET_TEACHER=../../models/wav2lip/checkpoints/lipsync_expert.pth"
	@echo "  OFFICIAL_SYNCNET_PATH=../../models/wav2lip/checkpoints/lipsync_expert.pth"
	@echo "  SYNCNET_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  GENERATOR_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  SPEAKER_LIST=/abs/or/rel/speakers.txt"
	@echo "  SYNCNET_OUTPUT_DIR=output/<syncnet_run_dir>"
	@echo "  SYNCNET_WATCH_PID=<running_syncnet_pid>"
	@echo "  COMPARE_SAMPLES=200"
	@echo "  BENCH_FACE=/abs/path/face.mp4"
	@echo "  BENCH_AUDIO=/abs/path/audio.mp3"
	@echo "  BENCH_CHECKPOINT=/abs/path/Wav2Lip-SD-GAN.pt"
	@echo "  ARTIFACTS_OUTPUT_DIR=training/output/<run_name>"
	@echo "  ARTIFACTS_RUN_KIND=syncnet|generator|pipeline|smoke|auto"
	@echo "  ARTIFACTS_RUN_NAME=<drive_subdir_name>"
	@echo "  ARTIFACTS_CONFIG_PATH=training/configs/<config>.yaml"

server-setup:
	@if ! command -v apt-get >/dev/null 2>&1; then \
		echo "server-setup is intended for Ubuntu/Debian training servers with apt-get"; \
		exit 1; \
	fi
	apt-get update
	DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsndfile1 rsync rclone git make
	$(PIP) install --upgrade pip
	$(PIP) install --no-cache-dir -r $(SERVER_PY_REQUIREMENTS)
	@echo "server-setup complete"

smoke-lazy:
	cd $(TRAINING_ROOT) && bash $(SMOKE_LAZY_WORKFLOW)

train-syncnet:
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/train_syncnet.py \
		--config $(SYNCNET_CONFIG) \
		$(if $(SYNCNET_RESUME),--resume $(SYNCNET_RESUME),) \
		$(if $(SPEAKER_LIST),--speaker-list $(SPEAKER_LIST),)

train-generator:
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/train_generator.py \
		--config $(GENERATOR_CONFIG) \
		--syncnet $(SYNCNET_TEACHER) \
		$(if $(GENERATOR_RESUME),--resume $(GENERATOR_RESUME),) \
		$(if $(SPEAKER_LIST),--speaker-list $(SPEAKER_LIST),)

prewarm-syncnet-cache:
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/prewarm_lazy_cache.py \
		--config $(PREWARM_CONFIG) \
		--log-every $(PREWARM_LOG_EVERY) \
		$(if $(PREWARM_SPEAKER_LIST),--speaker-list $(PREWARM_SPEAKER_LIST),) \
		$(if $(PREWARM_MAX_ITEMS),--max-items $(PREWARM_MAX_ITEMS),)

observe-system:
	mkdir -p "$(SYSTEM_WATCH_DIR)"
	@pids="$$(pgrep -f '[s]cripts/system_watch.py' || true)"; \
	if [ -n "$$pids" ]; then \
		echo "$$pids" | xargs -r kill 2>/dev/null || true; \
		sleep 1; \
	fi
	@: > "$(SYSTEM_WATCH_DIR)/system_watch.log"
	cd $(TRAINING_ROOT) && nohup $(PYTHON) -u scripts/system_watch.py \
		--interval $(SYSTEM_WATCH_INTERVAL) \
		> "$(SYSTEM_WATCH_DIR)/system_watch.log" 2>&1 & echo $$! > "$(SYSTEM_WATCH_DIR)/system_watch.pid"
	@echo "started observe-system:"
	@echo "  pid file: $(SYSTEM_WATCH_DIR)/system_watch.pid"
	@echo "  live log:"
	@echo "    tail -n 17 -f $(SYSTEM_WATCH_DIR)/system_watch.log"
	@echo "  recommended remote observer:"
	@echo "    ssh -t -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p <PORT> root@<HOST> \\"
	@echo "      \"watch -c -n $(SYSTEM_WATCH_INTERVAL) 'tail -n 17 /root/lipsync_test/rep_lipsync_training/training/output/system_observe/system_watch.log'\""

watch-syncnet-generator:
	@if [ -z "$(SYNCNET_OUTPUT_DIR)" ]; then \
		echo "SYNCNET_OUTPUT_DIR is required, e.g. output/<syncnet_run_dir>"; \
		exit 1; \
	fi
	@if [ -z "$(SPEAKER_LIST)" ]; then \
		echo "SPEAKER_LIST is required, e.g. output/<syncnet_run_dir>/train_confident_medium_allowlist.txt"; \
		exit 1; \
	fi
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/watch_syncnet_then_train_generator.py \
		--syncnet-config $(SYNCNET_CONFIG) \
		--syncnet-output-dir $(SYNCNET_OUTPUT_DIR) \
		--speaker-snapshot $(SPEAKER_LIST) \
		--official-syncnet $(OFFICIAL_SYNCNET_PATH) \
		--generator-template-config $(GENERATOR_TEMPLATE_CONFIG) \
		--compare-samples $(COMPARE_SAMPLES) \
		--compare-seed $(COMPARE_SEED) \
		--compare-device $(COMPARE_DEVICE) \
		--generator-batch-size $(GENERATOR_BATCH_SIZE) \
		$(if $(SYNCNET_WATCH_PID),--syncnet-pid $(SYNCNET_WATCH_PID),) \
		$(if $(GENERATOR_EPOCHS),--generator-epochs $(GENERATOR_EPOCHS),) \
		$(if $(GENERATOR_OUTPUT_DIR),--generator-output-dir $(GENERATOR_OUTPUT_DIR),) \
		$(if $(GENERATOR_LAZY_CACHE_ROOT),--generator-lazy-cache-root $(GENERATOR_LAZY_CACHE_ROOT),) \
		--poll-seconds $(WATCH_POLL_SECONDS)

bench-wav2lip:
	@if [ -z "$(BENCH_FACE)" ]; then \
		echo "BENCH_FACE is required, e.g. /abs/path/portrait_avatar.mp4"; \
		exit 1; \
	fi
	@if [ -z "$(BENCH_AUDIO)" ]; then \
		echo "BENCH_AUDIO is required, e.g. /abs/path/short_4s.mp3"; \
		exit 1; \
	fi
	@if [ -z "$(BENCH_CHECKPOINT)" ]; then \
		echo "BENCH_CHECKPOINT is required, e.g. /abs/path/Wav2Lip-SD-GAN.pt"; \
		exit 1; \
	fi
	cd $(REPO_ROOT) && $(PYTHON) training/scripts/run_official_wav2lip_benchmark.py \
		--face "$(BENCH_FACE)" \
		--audio "$(BENCH_AUDIO)" \
		--checkpoint "$(BENCH_CHECKPOINT)" \
		--device "$(BENCH_DEVICE)" \
		--detector_device "$(BENCH_DETECTOR_DEVICE)" \
		--batch_size $(BENCH_BATCH_SIZE) \
		--face_det_batch_size $(BENCH_FACE_DET_BATCH_SIZE) \
		--pads $(BENCH_PADS) \
		--resize_factor $(BENCH_RESIZE_FACTOR) \
		$(if $(BENCH_OUTFILE),--outfile "$(BENCH_OUTFILE)",) \
		$(if $(BENCH_S3FD_PATH),--s3fd_path "$(BENCH_S3FD_PATH)",) \
		$(if $(filter 1 true yes,$(BENCH_STATIC)),--static,) \
		$(if $(filter 1 true yes,$(BENCH_NOSMOOTH)),--nosmooth,)

upload-training-artifacts:
	@if [ -z "$(ARTIFACTS_OUTPUT_DIR)" ]; then \
		echo "ARTIFACTS_OUTPUT_DIR is required, e.g. training/output/<run_name>"; \
		exit 1; \
	fi
	cd $(REPO_ROOT) && $(PYTHON) training/scripts/upload_training_artifacts.py \
		--source-dir "$(ARTIFACTS_OUTPUT_DIR)" \
		--run-kind "$(ARTIFACTS_RUN_KIND)" \
		--remote "$(ARTIFACTS_REMOTE)" \
		--drive-root-folder-id "$(ARTIFACTS_DRIVE_ROOT_ID)" \
		--log-path "$(ARTIFACTS_LOG_PATH)" \
		$(if $(ARTIFACTS_RUN_NAME),--run-name "$(ARTIFACTS_RUN_NAME)",) \
		$(if $(ARTIFACTS_CONFIG_PATH),--config-path "$(ARTIFACTS_CONFIG_PATH)",) \
		$(if $(filter 1 true yes,$(ARTIFACTS_INCLUDE_MEDIA)),--include-media,) \
		$(if $(filter 1 true yes,$(ARTIFACTS_DRY_RUN)),--dry-run,)
