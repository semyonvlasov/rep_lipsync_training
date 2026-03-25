SHELL := /bin/bash

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TRAINING_ROOT := $(REPO_ROOT)/training
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

SERVER_PY_REQUIREMENTS := $(TRAINING_ROOT)/requirements-server.txt

SMOKE_LAZY_WORKFLOW ?= workflows/train/run_lazy_smoke_remote_20260325.sh
SYNCNET_CONFIG ?= configs/syncnet_cuda3090_medium.yaml
GENERATOR_CONFIG ?= configs/lipsync_cuda3090_hdtf_talkvid.yaml
SYNCNET_TEACHER ?= ../models/official_syncnet/checkpoints/lipsync_expert.pth
SYNCNET_RESUME ?=
GENERATOR_RESUME ?=
SPEAKER_LIST ?=

.PHONY: help server-setup smoke-lazy train-syncnet train-generator

help:
	@echo "Available targets:"
	@echo "  make server-setup    # install apt + pip deps for remote Linux/Vast training"
	@echo "  make smoke-lazy      # run the lazy dataset smoke workflow"
	@echo "  make train-syncnet   # run scripts/train_syncnet.py with \$$SYNCNET_CONFIG"
	@echo "  make train-generator # run scripts/train_generator.py with \$$GENERATOR_CONFIG"
	@echo ""
	@echo "Useful overrides:"
	@echo "  PYTHON=python3"
	@echo "  SYNCNET_CONFIG=configs/syncnet_cuda3090_medium.yaml"
	@echo "  GENERATOR_CONFIG=configs/lipsync_cuda3090_hdtf_talkvid.yaml"
	@echo "  SYNCNET_TEACHER=../models/official_syncnet/checkpoints/lipsync_expert.pth"
	@echo "  SYNCNET_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  GENERATOR_RESUME=/abs/or/rel/checkpoint.pth"
	@echo "  SPEAKER_LIST=/abs/or/rel/speakers.txt"

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
