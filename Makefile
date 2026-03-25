SHELL := /bin/bash

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TRAINING_ROOT := $(REPO_ROOT)/training
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SERVER_PY_REQUIREMENTS := $(TRAINING_ROOT)/requirements-server.txt

SMOKE_LAZY_WORKFLOW ?= workflows/train/run_lazy_smoke_remote_20260325.sh
SYNCNET_HOP_ABLATION_WORKFLOW ?= workflows/train/run_syncnet_hop_ablation_20260325.sh
SYNCNET_CONFIG ?= configs/syncnet_cuda3090_medium.yaml
GENERATOR_CONFIG ?= configs/lipsync_cuda3090_hdtf_talkvid.yaml
SYNCNET_TEACHER ?= ../models/official_syncnet/checkpoints/lipsync_expert.pth
SYNCNET_RESUME ?=
GENERATOR_RESUME ?=
SPEAKER_LIST ?=
ARTIFACTS_OUTPUT_DIR ?=
ARTIFACTS_RUN_KIND ?= auto
ARTIFACTS_RUN_NAME ?=
ARTIFACTS_CONFIG_PATH ?=
ARTIFACTS_REMOTE ?= gdrive:
ARTIFACTS_DRIVE_ROOT_ID ?= 1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv
ARTIFACTS_LOG_PATH ?= docs/training_runs/log.md
ARTIFACTS_INCLUDE_MEDIA ?= 0
ARTIFACTS_DRY_RUN ?= 0

.PHONY: help server-setup smoke-lazy syncnet-hop-ablation train-syncnet train-generator upload-training-artifacts

help:
	@echo "Available targets:"
	@echo "  make server-setup    # install apt + pip deps for remote Linux/Vast training"
	@echo "  make smoke-lazy      # run the lazy dataset smoke workflow"
	@echo "  make syncnet-hop-ablation # run the hop160 vs hop200 SyncNet ablation workflow"
	@echo "  make train-syncnet   # run scripts/train_syncnet.py with \$$SYNCNET_CONFIG"
	@echo "  make train-generator # run scripts/train_generator.py with \$$GENERATOR_CONFIG"
	@echo "  make upload-training-artifacts # upload a finished run to Drive and append the git log"
	@echo ""
	@echo "Useful overrides:"
	@echo "  PYTHON=python3"
	@echo "  SYNCNET_CONFIG=configs/syncnet_cuda3090_medium.yaml"
	@echo "  GENERATOR_CONFIG=configs/lipsync_cuda3090_hdtf_talkvid.yaml"
	@echo "  SYNCNET_TEACHER=../models/official_syncnet/checkpoints/lipsync_expert.pth"
	@echo "  SYNCNET_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  GENERATOR_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  SPEAKER_LIST=/abs/or/rel/speakers.txt"
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

syncnet-hop-ablation:
	cd $(TRAINING_ROOT) && bash $(SYNCNET_HOP_ABLATION_WORKFLOW)

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
