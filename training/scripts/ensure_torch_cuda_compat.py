#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


def _gpu_present() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _probe_torch() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["torch_import_ok"] = False
        info["error"] = repr(exc)
        return info

    info["torch_import_ok"] = True
    info["torch_version"] = getattr(torch, "__version__", "<unknown>")

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["cuda_available"] = False
        info["cuda_available_error"] = repr(exc)
        return info

    if not info["cuda_available"]:
        return info

    try:
        info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["device_name_error"] = repr(exc)

    required_arch = None
    try:
        major, minor = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{major}{minor}"
        info["device_capability"] = [major, minor]
        info["required_arch"] = required_arch
    except Exception as exc:
        info["device_capability_error"] = repr(exc)

    try:
        arch_list = list(torch.cuda.get_arch_list())
        info["arch_list"] = arch_list
        if required_arch is not None:
            info["arch_supported"] = required_arch in arch_list
    except Exception as exc:
        info["arch_list_error"] = repr(exc)

    try:
        value = (torch.zeros((1,), device="cuda") + 1).sum().item()
        info["simple_cuda_ok"] = True
        info["simple_cuda_value"] = float(value)
    except Exception as exc:
        info["simple_cuda_ok"] = False
        info["simple_cuda_error"] = str(exc)

    return info


def _needs_repair(gpu_present: bool, probe: dict[str, Any]) -> tuple[bool, str]:
    if not gpu_present:
        return False, "no_gpu_detected"
    if not probe.get("torch_import_ok"):
        return True, "torch_import_failed"
    if not probe.get("cuda_available"):
        return True, "torch_cuda_unavailable"
    if probe.get("arch_supported") is False:
        return True, f"missing_arch:{probe.get('required_arch')}"
    if probe.get("simple_cuda_ok") is False:
        return True, "simple_cuda_failed"
    return False, "compatible"


def _install_cuda_torch(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"torch=={args.torch_version}",
        f"torchvision=={args.torchvision_version}",
        f"torchaudio=={args.torchaudio_version}",
        "--index-url",
        args.index_url,
    ]
    print(
        "[ensure-torch-cuda] upgrading torch packages:",
        " ".join(cmd),
        flush=True,
    )
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate torch/CUDA compatibility and auto-upgrade when needed."
    )
    parser.add_argument("--torch-version", default="2.10.0")
    parser.add_argument("--torchvision-version", default="0.25.0")
    parser.add_argument("--torchaudio-version", default="2.10.0")
    parser.add_argument(
        "--index-url",
        default="https://download.pytorch.org/whl/cu128",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print raw probe JSON after validation.",
    )
    args = parser.parse_args()

    gpu_present = _gpu_present()
    print(f"[ensure-torch-cuda] gpu_present={gpu_present}", flush=True)
    initial = _probe_torch()
    print(
        "[ensure-torch-cuda] initial_probe="
        + json.dumps(initial, sort_keys=True, ensure_ascii=True),
        flush=True,
    )
    repair_needed, reason = _needs_repair(gpu_present, initial)
    print(f"[ensure-torch-cuda] status={reason}", flush=True)

    if repair_needed:
        _install_cuda_torch(args)
        final = _probe_torch()
        print(
            "[ensure-torch-cuda] final_probe="
            + json.dumps(final, sort_keys=True, ensure_ascii=True),
            flush=True,
        )
        still_broken, final_reason = _needs_repair(gpu_present, final)
        if still_broken:
            print(
                f"[ensure-torch-cuda] ERROR: still incompatible after repair ({final_reason})",
                file=sys.stderr,
                flush=True,
            )
            return 1
        if args.print_json:
            print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=True))
        print("[ensure-torch-cuda] compatibility repaired", flush=True)
        return 0

    if args.print_json:
        print(json.dumps(initial, indent=2, sort_keys=True, ensure_ascii=True))
    print("[ensure-torch-cuda] compatibility ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
