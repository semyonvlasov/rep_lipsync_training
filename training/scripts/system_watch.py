#!/usr/bin/env python3
"""
Lightweight system observer for remote training boxes.

Prints a human-readable multi-line snapshot every N seconds with:
- CPU utilization
- RAM usage
- GPU utilization / VRAM / power / temperature
- Network RX/TX throughput

Typical usage:
    python3 -u scripts/system_watch.py --interval 10
"""

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone

BOLD = "\033[1m"
RESET = "\033[0m"


def detect_net_dev():
    for cand in ("eth0", "ens5", "enp1s0"):
        if os.path.exists(f"/sys/class/net/{cand}/statistics/rx_bytes"):
            return cand
    devs = [d for d in os.listdir("/sys/class/net") if d != "lo"]
    return devs[0] if devs else "lo"


def read_cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]
    total = sum(vals)
    return idle, total


def read_mem():
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.strip().split()[0])
    total = vals.get("MemTotal", 0) / 1024.0
    avail = vals.get("MemAvailable", 0) / 1024.0
    used = total - avail
    return used, total


def read_net_bytes(dev):
    with open(f"/sys/class/net/{dev}/statistics/rx_bytes") as f:
        rx = int(f.read().strip())
    with open(f"/sys/class/net/{dev}/statistics/tx_bytes") as f:
        tx = int(f.read().strip())
    return rx, tx


def read_gpu():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        util, mem_used, mem_total, power, temp = [x.strip() for x in out.split(",")]
        return {
            "gpu_util": float(util),
            "vram_used_mb": float(mem_used),
            "vram_total_mb": float(mem_total),
            "gpu_power_w": float(power),
            "gpu_temp_c": float(temp),
        }
    except Exception:
        return {
            "gpu_util": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
        }


def read_disk_space(path="/"):
    stat = os.statvfs(path)
    total = (stat.f_blocks * stat.f_frsize) / (1024.0 ** 3)
    free = (stat.f_bavail * stat.f_frsize) / (1024.0 ** 3)
    used = total - free
    return used, total, free


def iter_physical_block_devices():
    devices = []
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if name.startswith(("loop", "ram", "fd", "sr", "md", "dm-")):
                continue
            if name.startswith("nvme") and "p" in name:
                continue
            if name.startswith("sd") and any(ch.isdigit() for ch in name[2:]):
                continue
            if name.startswith("vd") and any(ch.isdigit() for ch in name[2:]):
                continue
            if name.startswith("xvd") and any(ch.isdigit() for ch in name[3:]):
                continue
            devices.append(name)
    return devices


def read_disk_bytes():
    wanted = set(iter_physical_block_devices())
    read_bytes = 0
    write_bytes = 0
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if name not in wanted:
                continue
            sectors_read = int(parts[5])
            sectors_written = int(parts[9])
            read_bytes += sectors_read * 512
            write_bytes += sectors_written * 512
    return read_bytes, write_bytes


def fmt_num(value, suffix=""):
    if value is None:
        return "-"
    return f"{value:.1f}{suffix}"


def bold(text):
    return f"{BOLD}{text}{RESET}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10, help="Snapshot interval in seconds")
    args = parser.parse_args()

    interval = max(1, int(args.interval))
    net_dev = detect_net_dev()

    prev_idle, prev_total = read_cpu_times()
    prev_rx, prev_tx = read_net_bytes(net_dev)
    prev_disk_read, prev_disk_write = read_disk_bytes()
    prev_t = time.time()

    print(f"{bold('Observer')} net={net_dev} interval={interval}s", flush=True)
    while True:
        time.sleep(interval)
        now = time.time()
        idle, total = read_cpu_times()
        rx, tx = read_net_bytes(net_dev)
        disk_read, disk_write = read_disk_bytes()
        dt = max(now - prev_t, 1e-6)

        cpu_util = 100.0 * (1.0 - ((idle - prev_idle) / max(total - prev_total, 1)))
        ram_used, ram_total = read_mem()
        disk_used_gb, disk_total_gb, disk_free_gb = read_disk_space("/")
        rx_mbps = ((rx - prev_rx) * 8.0) / dt / 1_000_000.0
        tx_mbps = ((tx - prev_tx) * 8.0) / dt / 1_000_000.0
        disk_read_mbps = (disk_read - prev_disk_read) / dt / (1024.0 ** 2)
        disk_write_mbps = (disk_write - prev_disk_write) / dt / (1024.0 ** 2)
        gpu = read_gpu()

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{ts}]", flush=True)
        print(bold("CPU / RAM"), flush=True)
        print(f"cpu: {cpu_util:.1f}%", flush=True)
        print(f"ram: {ram_used:.1f} / {ram_total:.1f} MB", flush=True)
        print(bold("Disk"), flush=True)
        print(f"space: {disk_used_gb:.1f} / {disk_total_gb:.1f} GB (free {disk_free_gb:.1f} GB)", flush=True)
        print(f"read: {disk_read_mbps:.2f} MB/s", flush=True)
        print(f"write: {disk_write_mbps:.2f} MB/s", flush=True)
        print(bold("GPU / VRAM"), flush=True)
        print(f"gpu: {fmt_num(gpu['gpu_util'], '%')}", flush=True)
        print(
            f"vram: {fmt_num(gpu['vram_used_mb'])} / {fmt_num(gpu['vram_total_mb'])} MB",
            flush=True,
        )
        print(f"power: {fmt_num(gpu['gpu_power_w'], ' W')}", flush=True)
        print(f"temp: {fmt_num(gpu['gpu_temp_c'], ' C')}", flush=True)
        print(bold(f"Network ({net_dev})"), flush=True)
        print(f"rx: {rx_mbps:.3f} Mbps", flush=True)
        print(f"tx: {tx_mbps:.3f} Mbps", flush=True)
        print("", flush=True)

        prev_idle, prev_total = idle, total
        prev_rx, prev_tx = rx, tx
        prev_disk_read, prev_disk_write = disk_read, disk_write
        prev_t = now


if __name__ == "__main__":
    main()
