#!/usr/bin/env python3
"""Compatibility shim for the shared raw/local faceclip export engine."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.process.common.export_faceclip_batch import *  # noqa: F401,F403
from dataset_prepare.process.common.export_faceclip_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
