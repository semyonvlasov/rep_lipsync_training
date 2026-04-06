#!/usr/bin/env python3
"""Compatibility shim for shared process video transcode helpers."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.process.common.transcode_video import *  # noqa: F401,F403
