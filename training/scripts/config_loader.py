#!/usr/bin/env python3
"""
Shared YAML config loader with narrow defaults for data.sync_alignment only.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


SYNC_ALIGNMENT_DEFAULTS = {
    "outlier_trim_ratio": 0.2,
    "min_consensus_ratio": 0.7,
    "max_shift_mad": 1.5,
}


def _load_yaml_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping YAML at {path}, got {type(payload).__name__}")
    return payload


def apply_shared_sync_alignment_defaults(cfg: dict) -> dict:
    result = copy.deepcopy(cfg)
    data_cfg = result.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError(f"Expected 'data' to be a mapping, got {type(data_cfg).__name__}")
    sync_cfg = data_cfg.setdefault("sync_alignment", {})
    if not isinstance(sync_cfg, dict):
        raise TypeError(
            f"Expected 'data.sync_alignment' to be a mapping, got {type(sync_cfg).__name__}"
        )
    for key, value in SYNC_ALIGNMENT_DEFAULTS.items():
        sync_cfg.setdefault(key, value)
    return result


def load_config(config_path: str | Path) -> dict:
    cfg = _load_yaml_file(Path(config_path).resolve())
    return apply_shared_sync_alignment_defaults(cfg)
