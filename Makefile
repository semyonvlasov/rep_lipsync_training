SHELL := /bin/bash

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TRAINING_ROOT := $(REPO_ROOT)/training
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
REMOTE ?= root@ssh9.vast.ai
PORT ?= 22
REMOTE_ROOT ?= /root/lipsync_test/rep_lipsync_training
REMOTE_PYTHON ?= python3
OFFICIAL_SYNCNET_CKPT ?=
OFFICIAL_SYNCNET_URL ?= https://drive.google.com/open?id=1_RUm6ncXDX2e_Qoyj24Yrvr6CGaBtPY4
OFFICIAL_SYNCNET_SKIP_UPLOAD ?= 1
RCLONE_CONFIG_PATH ?= $(HOME)/.config/rclone/rclone.conf
REMOTE_RCLONE_CONFIG ?= /root/.config/rclone/rclone.conf
REMOTE_RCLONE_DIR ?= /root/.config/rclone
REMOTE_RCLONE_PARENT ?= /root/.config

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
DATASET_REMOTE ?= gdrive:
DATASET_FOLDER_ID ?= 1xx2IlfiAYC1AFf3xJwcjeTEsAK-Uqt8n
DATASET_ARCHIVE_GLOB ?= *faceclips*.tar
DATASET_MAX_ARCHIVES ?= 0
DATASET_MANIFEST_PATH ?= output/faceclip_merge/merge_manifest.jsonl
DATASET_DOWNLOAD_DIR ?= data/_imports/faceclip_merge_downloads
DATASET_EXTRACT_DIR ?= data/_imports/faceclip_merge_extracted
DATASET_HDTF_ROOT ?= data/hdtf/processed
DATASET_TALKVID_ROOT ?= data/talkvid/processed_medium
DATASET_IMPORT_SUBDIR ?= _lazy_imports
DATASET_INCLUDE_TIERS ?=
SYNCNET_TEACHER ?= ../../models/wav2lip/checkpoints/lipsync_expert.pth
OFFICIAL_SYNCNET_PATH ?= ../../models/wav2lip/checkpoints/lipsync_expert.pth
SYNCNET_RESUME ?=
GENERATOR_RESUME ?=
SPEAKER_LIST ?=
VAL_SPEAKER_LIST ?=
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
PUBLISH_CHECKPOINT ?=
PUBLISH_INFER_ONLY_OUT ?=
PUBLISH_RUN_NAME ?=
PUBLISH_REMOTE ?= gdrive:
PUBLISH_DRIVE_ROOT_ID ?= 1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv
PUBLISH_FACE_LIST ?=
PUBLISH_AUDIO ?=
PUBLISH_OUTPUT_DIR ?=
PUBLISH_DEVICE ?= cuda
PUBLISH_DETECTOR_DEVICE ?= cuda
PUBLISH_BATCH_SIZE ?= 32
PUBLISH_FACE_DET_BATCH_SIZE ?= 4
PUBLISH_S3FD_PATH ?=
PUBLISH_SKIP_UPLOAD ?= 0

.PHONY: help server-setup remote-sync-code remote-server-setup remote-rclone-config remote-prewarm-sfd remote-fetch-official-syncnet remote-observe-system remote-bootstrap remote-faceclip-bootstrap dataset remote-dataset smoke-lazy train-syncnet train-generator prewarm-syncnet-cache observe-system watch-syncnet-generator bench-wav2lip publish-checkpoint-benchmark upload-training-artifacts

help:
	@echo "Available targets:"
	@echo "  make server-setup    # install apt + pip deps for remote Linux/Vast training"
	@echo "  make remote-sync-code # upload repo training code + official SyncNet assets to a remote box"
	@echo "  make remote-server-setup # install apt + pip deps on a remote Linux/Vast box"
	@echo "  make remote-rclone-config # upload local rclone.conf so the remote can access Drive"
	@echo "  make remote-prewarm-sfd # instantiate the SFD detector once so the checkpoint is cached on the remote"
	@echo "  make remote-fetch-official-syncnet # download official SyncNet checkpoint on the remote from Drive"
	@echo "  make remote-observe-system # start the remote system observer"
	@echo "  make remote-bootstrap # sync code, install deps, and start the remote observer"
	@echo "  make remote-faceclip-bootstrap # prepare a remote box for raw->lazy faceclip processing (SFD + ffmpeg + rclone)"
	@echo "  make dataset         # fetch new processed faceclip archives and merge them into lazy dataset roots"
	@echo "  make remote-dataset  # run make dataset on the configured remote box"
	@echo "  make smoke-lazy      # run the lazy dataset smoke workflow"
	@echo "  make train-syncnet   # run scripts/train_syncnet.py with \$$SYNCNET_CONFIG"
	@echo "  make train-generator # run scripts/train_generator.py with \$$GENERATOR_CONFIG"
	@echo "  make prewarm-syncnet-cache # pre-materialize lazy frames/mels into the configured cache root"
	@echo "  make observe-system  # start a background CPU/RAM/GPU/VRAM/network observer"
	@echo "  make watch-syncnet-generator # wait for SyncNet, benchmark all epochs vs official, then launch generator"
	@echo "  make bench-wav2lip   # run the official Wav2Lip benchmark path (SFD + 96x96 generator)"
	@echo "  make publish-checkpoint-benchmark # upload a checkpoint to Drive and benchmark it on-server"
	@echo "  make upload-training-artifacts # upload a finished run to Drive and append the git log"
	@echo ""
	@echo "Useful overrides:"
	@echo "  PYTHON=python3"
	@echo "  REMOTE=root@ssh9.vast.ai"
	@echo "  PORT=24380"
	@echo "  REMOTE_ROOT=/root/lipsync_test/rep_lipsync_training"
	@echo "  OFFICIAL_SYNCNET_CKPT=/abs/path/lipsync_expert.pth"
	@echo "  OFFICIAL_SYNCNET_URL=https://drive.google.com/open?id=..."
	@echo "  OFFICIAL_SYNCNET_SKIP_UPLOAD=1"
	@echo "  DATASET_FOLDER_ID=1xx2IlfiAYC1AFf3xJwcjeTEsAK-Uqt8n"
	@echo "  DATASET_MANIFEST_PATH=output/faceclip_merge/merge_manifest.jsonl"
	@echo "  DATASET_INCLUDE_TIERS=\"confident medium\""
	@echo "  SYNCNET_CONFIG=configs/syncnet_cuda3090_medium.yaml"
	@echo "  GENERATOR_CONFIG=configs/lipsync_cuda3090_hdtf_talkvid.yaml"
	@echo "  SYNCNET_TEACHER=../../models/wav2lip/checkpoints/lipsync_expert.pth"
	@echo "  OFFICIAL_SYNCNET_PATH=../../models/wav2lip/checkpoints/lipsync_expert.pth"
	@echo "  SYNCNET_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  GENERATOR_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  SPEAKER_LIST=/abs/or/rel/speakers.txt"
	@echo "  VAL_SPEAKER_LIST=/abs/or/rel/val_speakers.txt"
	@echo "  SYNCNET_OUTPUT_DIR=output/<syncnet_run_dir>"
	@echo "  SYNCNET_WATCH_PID=<running_syncnet_pid>"
	@echo "  COMPARE_SAMPLES=200"
	@echo "  BENCH_FACE=/abs/path/face.mp4"
	@echo "  BENCH_AUDIO=/abs/path/audio.mp3"
	@echo "  BENCH_CHECKPOINT=/abs/path/Wav2Lip-SD-GAN.pt"
	@echo "  PUBLISH_CHECKPOINT=output/<run>/generator/generator_epoch000.pth"
	@echo "  PUBLISH_FACE_LIST=\"/abs/face1.mp4 /abs/face2.mp4\""
	@echo "  PUBLISH_AUDIO=/abs/path/short_4s.mp3"
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

remote-sync-code:
	PORT="$(PORT)" REMOTE="$(REMOTE)" REMOTE_ROOT="$(REMOTE_ROOT)" \
		OFFICIAL_SYNCNET_CKPT="$(OFFICIAL_SYNCNET_CKPT)" \
		OFFICIAL_SYNCNET_URL="$(OFFICIAL_SYNCNET_URL)" \
		OFFICIAL_SYNCNET_SKIP_UPLOAD="$(OFFICIAL_SYNCNET_SKIP_UPLOAD)" \
		bash "$(TRAINING_ROOT)/workflows/train/sync_remote_code.sh"
	@echo "remote-sync-code complete: $(REMOTE):$(REMOTE_ROOT)"

remote-server-setup:
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		if ! command -v apt-get >/dev/null 2>&1; then \
			echo 'remote-server-setup is intended for Ubuntu/Debian training servers with apt-get'; \
			exit 1; \
		fi; \
		apt-get update; \
		DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsndfile1 rsync rclone git make python3-pip; \
		$(REMOTE_PYTHON) -m pip install --upgrade pip; \
		$(REMOTE_PYTHON) -m pip install --no-cache-dir -r '$(REMOTE_ROOT)/training/requirements-server.txt'; \
	"
	@echo "remote-server-setup complete: $(REMOTE):$(REMOTE_ROOT)"

remote-rclone-config:
	@if [ ! -f "$(RCLONE_CONFIG_PATH)" ]; then \
		echo "remote-rclone-config: missing local rclone config at $(RCLONE_CONFIG_PATH)"; \
		exit 1; \
	fi
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		mkdir -p '$(REMOTE_RCLONE_DIR)'; \
		chmod 700 '$(REMOTE_RCLONE_PARENT)' '$(REMOTE_RCLONE_DIR)' 2>/dev/null || true; \
	"
	scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$(PORT)" \
		"$(RCLONE_CONFIG_PATH)" "$(REMOTE):$(REMOTE_RCLONE_CONFIG)"
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		chmod 600 '$(REMOTE_RCLONE_CONFIG)'; \
		rclone listremotes >/dev/null; \
	"
	@echo "remote-rclone-config complete: $(REMOTE):$(REMOTE_RCLONE_CONFIG)"

remote-prewarm-sfd:
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		cd '$(REMOTE_ROOT)/training'; \
		PYTHONPATH='$(REMOTE_ROOT)/models/official_syncnet:$(REMOTE_ROOT)/training:$(REMOTE_ROOT)' \
			$(REMOTE_PYTHON) -c \"from face_detection import FaceAlignment, LandmarksType; FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False, face_detector='sfd'); print('SFD checkpoint cached')\"; \
	"
	@echo "remote-prewarm-sfd complete: $(REMOTE):$(REMOTE_ROOT)"

remote-fetch-official-syncnet:
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		mkdir -p '$(REMOTE_ROOT)/models/official_syncnet/checkpoints'; \
		$(REMOTE_PYTHON) -m gdown --fuzzy '$(OFFICIAL_SYNCNET_URL)' -O '$(REMOTE_ROOT)/models/official_syncnet/checkpoints/lipsync_expert.pth'; \
	"
	@echo "remote-fetch-official-syncnet complete: $(REMOTE):$(REMOTE_ROOT)"

remote-observe-system:
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "bash -lc '\
		set -euo pipefail; \
		mkdir -p \"$(REMOTE_ROOT)/training/output/system_observe\"; \
		existing_pids=\$$(pgrep -f \"^$(REMOTE_PYTHON) -u scripts/system_watch.py --interval \" || true); \
		if [ -n \"\$$existing_pids\" ]; then \
			echo \"\$$existing_pids\" | xargs -r kill 2>/dev/null || true; \
			sleep 1; \
		fi; \
		if [ -f \"$(REMOTE_ROOT)/training/output/system_observe/system_watch.pid\" ]; then \
			old_pid=\$$(cat \"$(REMOTE_ROOT)/training/output/system_observe/system_watch.pid\" 2>/dev/null || true); \
			if [ -n \"\$$old_pid\" ] && ps -p \"\$$old_pid\" >/dev/null 2>&1; then \
				kill \"\$$old_pid\" 2>/dev/null || true; \
				sleep 1; \
			fi; \
		fi; \
		: > \"$(REMOTE_ROOT)/training/output/system_observe/system_watch.log\"; \
		cd \"$(REMOTE_ROOT)/training\"; \
		nohup $(REMOTE_PYTHON) -u scripts/system_watch.py --interval \"$(SYSTEM_WATCH_INTERVAL)\" \
			< /dev/null > \"$(REMOTE_ROOT)/training/output/system_observe/system_watch.log\" 2>&1 & \
		pid=\$$!; \
		echo \"\$$pid\" > \"$(REMOTE_ROOT)/training/output/system_observe/system_watch.pid\"; \
		sleep 1; \
		ps -p \"\$$pid\" >/dev/null \
	'"
	@echo "remote-observe-system complete: $(REMOTE):$(REMOTE_ROOT)"

remote-bootstrap: remote-sync-code remote-server-setup remote-rclone-config remote-fetch-official-syncnet remote-observe-system
	@echo "remote-bootstrap complete: $(REMOTE):$(REMOTE_ROOT)"

remote-faceclip-bootstrap: remote-sync-code remote-server-setup remote-rclone-config remote-prewarm-sfd
	-@$(MAKE) remote-observe-system REMOTE="$(REMOTE)" PORT="$(PORT)" REMOTE_ROOT="$(REMOTE_ROOT)" REMOTE_PYTHON="$(REMOTE_PYTHON)" SYSTEM_WATCH_INTERVAL="$(SYSTEM_WATCH_INTERVAL)"
	@echo "remote-faceclip-bootstrap complete: $(REMOTE):$(REMOTE_ROOT)"

dataset:
	@set -euo pipefail; \
	include_args=""; \
	for tier in $(DATASET_INCLUDE_TIERS); do \
		include_args="$$include_args --include-tier $$tier"; \
	done; \
	cd $(TRAINING_ROOT); \
	$(PYTHON) scripts/merge_faceclip_archives_from_gdrive.py \
		--remote "$(DATASET_REMOTE)" \
		--folder-id "$(DATASET_FOLDER_ID)" \
		--archive-glob "$(DATASET_ARCHIVE_GLOB)" \
		--max-archives $(DATASET_MAX_ARCHIVES) \
		--manifest-path "$(DATASET_MANIFEST_PATH)" \
		--download-dir "$(DATASET_DOWNLOAD_DIR)" \
		--extract-dir "$(DATASET_EXTRACT_DIR)" \
		--hdtf-root "$(DATASET_HDTF_ROOT)" \
		--talkvid-root "$(DATASET_TALKVID_ROOT)" \
		--import-subdir "$(DATASET_IMPORT_SUBDIR)" \
		$$include_args

remote-dataset:
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$(PORT)" "$(REMOTE)" "\
		set -euo pipefail; \
		cd '$(REMOTE_ROOT)'; \
		make dataset \
			PYTHON='$(REMOTE_PYTHON)' \
			DATASET_REMOTE='$(DATASET_REMOTE)' \
			DATASET_FOLDER_ID='$(DATASET_FOLDER_ID)' \
			DATASET_ARCHIVE_GLOB='$(DATASET_ARCHIVE_GLOB)' \
			DATASET_MAX_ARCHIVES='$(DATASET_MAX_ARCHIVES)' \
			DATASET_MANIFEST_PATH='$(DATASET_MANIFEST_PATH)' \
			DATASET_DOWNLOAD_DIR='$(DATASET_DOWNLOAD_DIR)' \
			DATASET_EXTRACT_DIR='$(DATASET_EXTRACT_DIR)' \
			DATASET_HDTF_ROOT='$(DATASET_HDTF_ROOT)' \
			DATASET_TALKVID_ROOT='$(DATASET_TALKVID_ROOT)' \
			DATASET_IMPORT_SUBDIR='$(DATASET_IMPORT_SUBDIR)' \
			DATASET_INCLUDE_TIERS='$(DATASET_INCLUDE_TIERS)'; \
	"
	@echo "remote-dataset complete: $(REMOTE):$(REMOTE_ROOT)"

smoke-lazy:
	cd $(TRAINING_ROOT) && bash $(SMOKE_LAZY_WORKFLOW)

train-syncnet:
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/train_syncnet.py \
		--config $(SYNCNET_CONFIG) \
		$(if $(SYNCNET_RESUME),--resume $(SYNCNET_RESUME),) \
		$(if $(SPEAKER_LIST),--speaker-list $(SPEAKER_LIST),) \
		$(if $(VAL_SPEAKER_LIST),--val-speaker-list $(VAL_SPEAKER_LIST),)

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
	@cd $(TRAINING_ROOT) && \
		nohup $(PYTHON) -u scripts/system_watch.py \
			--interval $(SYSTEM_WATCH_INTERVAL) \
			< /dev/null \
			> "$(SYSTEM_WATCH_DIR)/system_watch.log" 2>&1 & \
		pid=$$!; \
		disown $$pid; \
		echo $$pid > "$(SYSTEM_WATCH_DIR)/system_watch.pid"
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

publish-checkpoint-benchmark:
	@if [ -z "$(PUBLISH_CHECKPOINT)" ]; then \
		echo "PUBLISH_CHECKPOINT is required, e.g. output/<run>/generator/generator_epoch000.pth"; \
		exit 1; \
	fi
	@if [ -z "$(PUBLISH_AUDIO)" ]; then \
		echo "PUBLISH_AUDIO is required, e.g. /abs/path/short_4s.mp3"; \
		exit 1; \
	fi
	@if [ -z "$(PUBLISH_FACE_LIST)" ]; then \
		echo "PUBLISH_FACE_LIST is required, e.g. \"/abs/portrait_avatar.mp4 /abs/portrait_rama.mp4\""; \
		exit 1; \
	fi
	cd $(TRAINING_ROOT) && $(PYTHON) scripts/publish_checkpoint_benchmark.py \
		--checkpoint "$(PUBLISH_CHECKPOINT)" \
		--audio "$(PUBLISH_AUDIO)" \
		$(foreach f,$(PUBLISH_FACE_LIST),--face "$(f)") \
		--remote "$(PUBLISH_REMOTE)" \
		--drive-root-folder-id "$(PUBLISH_DRIVE_ROOT_ID)" \
		--device "$(PUBLISH_DEVICE)" \
		--detector-device "$(PUBLISH_DETECTOR_DEVICE)" \
		--batch-size $(PUBLISH_BATCH_SIZE) \
		--face-det-batch-size $(PUBLISH_FACE_DET_BATCH_SIZE) \
		$(if $(PUBLISH_INFER_ONLY_OUT),--infer-only-out "$(PUBLISH_INFER_ONLY_OUT)",) \
		$(if $(PUBLISH_RUN_NAME),--run-name "$(PUBLISH_RUN_NAME)",) \
		$(if $(PUBLISH_OUTPUT_DIR),--output-dir "$(PUBLISH_OUTPUT_DIR)",) \
		$(if $(PUBLISH_S3FD_PATH),--s3fd-path "$(PUBLISH_S3FD_PATH)",) \
		$(if $(filter 1 true yes,$(PUBLISH_SKIP_UPLOAD)),--skip-upload,)

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
