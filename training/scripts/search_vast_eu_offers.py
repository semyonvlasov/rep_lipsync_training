#!/usr/bin/env python3
"""
List cheap reliable Vast AI offers for training.

Example:
  python3 training/scripts/search_vast_eu_offers.py --storage-gb 800 --limit 20
  python3 training/scripts/search_vast_eu_offers.py --storage-gb 800 --eu-only
  python3 training/scripts/search_vast_eu_offers.py --storage-gb 800 --country DE --country NL

The script asks Vast to price the requested local storage amount, then prints
base GPU price, storage price, and total hourly price separately.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from typing import Any


DEFAULT_EUROPE_COUNTRY_CODES = (
    "AT",
    "BE",
    "BG",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IS",
    "IT",
    "LI",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
)


def number(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_vast_search(args: argparse.Namespace) -> list[dict[str, Any]]:
    query_parts = [
        "rentable=true",
        "verified=true",
        "rented=false",
        f"reliability>={args.min_reliability}",
        f"disk_space>={args.storage_gb}",
    ]
    if args.country_filter:
        query_parts.append(f"geolocation in [{','.join(args.country_filter)}]")
    if args.min_gpus is not None:
        query_parts.append(f"num_gpus>={args.min_gpus}")
    if args.max_gpus is not None:
        query_parts.append(f"num_gpus<={args.max_gpus}")
    if args.min_gpu_ram_gb is not None:
        query_parts.append(f"gpu_ram>={args.min_gpu_ram_gb}")
    if args.min_cuda is not None:
        query_parts.append(f"cuda_vers>={args.min_cuda}")
    for extra in args.extra_query:
        extra = extra.strip()
        if extra:
            query_parts.append(extra)

    query = " ".join(query_parts)
    cmd = [
        args.vastai_bin,
        "search",
        "offers",
        "--raw",
        query,
        "--storage",
        str(args.storage_gb),
        "--limit",
        str(args.search_limit),
        "-o",
        "dph",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=args.timeout)
    if proc.returncode != 0:
        sys.stderr.write(redact_secrets(proc.stderr or proc.stdout))
        raise SystemExit(proc.returncode)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"vastai returned non-JSON output: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit("vastai returned JSON, but not a list of offers")
    return [item for item in payload if isinstance(item, dict)]


def redact_secrets(text: str) -> str:
    return re.sub(r"(api_key=)[A-Za-z0-9._-]+", r"\1<redacted>", text)


def country_code(offer: dict[str, Any]) -> str:
    geo = str(offer.get("geolocation") or "")
    if "," in geo:
        return geo.rsplit(",", 1)[-1].strip()
    return geo.strip()


def max_days_until_expiry(offer: dict[str, Any], now: float) -> float:
    end_date = number(offer.get("end_date"))
    if math.isfinite(end_date) and end_date > 0:
        return max(0.0, (end_date - now) / 86400.0)

    duration = number(offer.get("duration"))
    if not math.isfinite(duration) or duration <= 0:
        return 0.0
    # Raw Vast JSON currently returns duration in seconds, while CLI help/table
    # labels the comparable value as days. Keep both representations tolerated.
    return duration / 86400.0 if duration > 3660 else duration


def price_parts(offer: dict[str, Any], storage_gb: float) -> tuple[float, float, float]:
    search = offer.get("search") if isinstance(offer.get("search"), dict) else {}
    base = number(offer.get("dph_base"), number(search.get("gpuCostPerHour"), 0.0))
    storage = number(
        offer.get("storage_total_cost"),
        number(search.get("diskHour"), math.nan),
    )
    if not math.isfinite(storage):
        storage_cost_month = number(offer.get("storage_cost"), 0.0)
        storage = storage_gb * storage_cost_month / (30.0 * 24.0)
    total = number(offer.get("dph_total"), number(search.get("totalHour"), base + storage))
    return base, storage, total


def gpu_ram_gb(offer: dict[str, Any]) -> float:
    raw = number(offer.get("gpu_ram"), 0.0)
    return raw / 1024.0 if raw > 512 else raw


def cpu_ram_gb(offer: dict[str, Any]) -> float:
    raw = number(offer.get("cpu_ram"), 0.0)
    return raw / 1024.0 if raw > 512 else raw


def filter_and_enrich(args: argparse.Namespace, offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = time.time()
    min_days = float(args.min_days)
    rows: list[dict[str, Any]] = []
    for offer in offers:
        days = max_days_until_expiry(offer, now)
        if days < min_days:
            continue
        reliability = number(offer.get("reliability"), 0.0)
        if reliability < args.min_reliability:
            continue
        disk_space = number(offer.get("disk_space"), 0.0)
        if disk_space < args.storage_gb:
            continue
        if args.min_gpu_ram_gb is not None and gpu_ram_gb(offer) < args.min_gpu_ram_gb:
            continue
        code = country_code(offer)
        if args.country_filter and code not in args.country_filter:
            continue
        base, storage, total = price_parts(offer, args.storage_gb)
        row = dict(offer)
        row["_country_code"] = code
        row["_max_days"] = days
        row["_price_base"] = base
        row["_price_storage"] = storage
        row["_price_total"] = total
        rows.append(row)

    rows.sort(key=lambda item: (item["_price_total"], -number(item.get("reliability"), 0.0)))
    return rows[: args.limit]


def fmt_money(value: float) -> str:
    return f"${value:.4f}" if math.isfinite(value) else "-"


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "-"


def offer_to_json(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "machine_id": row.get("machine_id"),
        "host_id": row.get("host_id"),
        "country": row.get("_country_code"),
        "geolocation": row.get("geolocation"),
        "gpu": row.get("gpu_name"),
        "num_gpus": row.get("num_gpus"),
        "gpu_ram_gb": round(gpu_ram_gb(row), 1),
        "cpu_ram_gb": round(cpu_ram_gb(row), 1),
        "disk_space_gb": row.get("disk_space"),
        "max_days": round(row["_max_days"], 1),
        "reliability": row.get("reliability"),
        "base_dph": row["_price_base"],
        "storage_dph": row["_price_storage"],
        "total_dph": row["_price_total"],
        "storage_cost_gb_month": row.get("storage_cost"),
        "cuda": row.get("cuda_max_good") or row.get("cuda_vers"),
        "driver": row.get("driver_version"),
        "inet_up_mbps": row.get("inet_up"),
        "inet_down_mbps": row.get("inet_down"),
        "direct_port_count": row.get("direct_port_count"),
    }


def print_json(rows: list[dict[str, Any]]) -> None:
    print(json.dumps([offer_to_json(row) for row in rows], ensure_ascii=False, indent=2))


def print_table(rows: list[dict[str, Any]], storage_gb: float) -> None:
    headers = [
        "#",
        "id",
        "country",
        "gpu",
        "gpus",
        "vram",
        "disk",
        "max_days",
        "rel",
        "base/hr",
        f"{storage_gb:g}GB/hr",
        "total/hr",
        "net up/down",
        "driver",
    ]
    table: list[list[str]] = [headers]
    for idx, row in enumerate(rows, start=1):
        table.append(
            [
                str(idx),
                str(row.get("id", "-")),
                str(row.get("_country_code", "-")),
                str(row.get("gpu_name", "-")).replace(" ", "_"),
                str(row.get("num_gpus", "-")),
                f"{fmt_float(gpu_ram_gb(row), 1)}G",
                f"{fmt_float(number(row.get('disk_space')), 0)}G",
                fmt_float(row["_max_days"], 1),
                f"{number(row.get('reliability'), 0.0) * 100:.2f}%",
                fmt_money(row["_price_base"]),
                fmt_money(row["_price_storage"]),
                fmt_money(row["_price_total"]),
                f"{fmt_float(number(row.get('inet_up')), 0)}/{fmt_float(number(row.get('inet_down')), 0)}",
                str(row.get("driver_version") or "-"),
            ]
        )
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]
    for row_idx, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row))
        print(line.rstrip())
        if row_idx == 0:
            print("  ".join("-" * width for width in widths).rstrip())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List cheapest reliable Vast AI offers with explicit storage pricing.",
    )
    parser.add_argument("--storage-gb", type=float, required=True, help="Requested local storage in GiB/GB for Vast pricing")
    parser.add_argument("--limit", type=int, default=20, help="Number of filtered offers to print")
    parser.add_argument("--search-limit", type=int, default=200, help="Number of raw Vast offers to request before local filtering")
    parser.add_argument("--min-days", type=float, default=7.0, help="Minimum days until offer expiration/end_date")
    parser.add_argument("--min-reliability", type=float, default=0.98, help="Minimum Vast reliability score, e.g. 0.99")
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--min-gpu-ram-gb", type=float, default=None)
    parser.add_argument("--min-cuda", type=float, default=None)
    parser.add_argument(
        "--eu-only",
        action="store_true",
        help="Restrict offers to the built-in Europe allowlist: EU/EEA + UK + CH + LI.",
    )
    parser.add_argument(
        "--country",
        action="append",
        default=[],
        help="Allowed country code. Can be repeated. Default: all countries unless --eu-only is set.",
    )
    parser.add_argument(
        "--extra-query",
        action="append",
        default=[],
        help='Extra Vast query fragment, e.g. --extra-query "gpu_name in [RTX_3090,RTX_4090]"',
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table")
    parser.add_argument("--vastai-bin", default="vastai")
    parser.add_argument("--timeout", type=int, default=60)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.storage_gb <= 0:
        raise SystemExit("--storage-gb must be positive")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.search_limit < args.limit:
        raise SystemExit("--search-limit must be >= --limit")
    explicit_countries = tuple(code.strip().upper() for code in args.country if code.strip())
    args.country_filter = explicit_countries or (DEFAULT_EUROPE_COUNTRY_CODES if args.eu_only else ())

    offers = run_vast_search(args)
    rows = filter_and_enrich(args, offers)
    if args.json:
        print_json(rows)
    else:
        print_table(rows, args.storage_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
