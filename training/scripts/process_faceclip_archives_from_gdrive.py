#!/usr/bin/env python3
"""
Compatibility wrapper for the rebuilt raw->lazy faceclip Drive processing loop.

The old entrypoint name is kept so existing shell wrappers can continue to call
it, but the implementation now lives in
`process_raw_archives_to_lazy_faceclips_gdrive.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).with_name("process_raw_archives_to_lazy_faceclips_gdrive.py")
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
