"""Shared YAML config loading and stage-launch helpers for dataset_prepare."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import json
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime only
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


class ConfigError(RuntimeError):
    """Raised when a stage config is missing required structure."""


_MISSING = object()


@dataclass(frozen=True)
class StagePaths:
    launcher_path: Path
    script_dir: Path
    stage_root: Path
    dataset_prepare_root: Path
    repo_root: Path


def discover_stage_paths(launcher_file: str) -> StagePaths:
    launcher_path = Path(launcher_file).resolve()
    script_dir = launcher_path.parent
    stage_root = script_dir.parent
    dataset_prepare_root = stage_root.parents[1]
    repo_root = dataset_prepare_root.parent
    return StagePaths(
        launcher_path=launcher_path,
        script_dir=script_dir,
        stage_root=stage_root,
        dataset_prepare_root=dataset_prepare_root,
        repo_root=repo_root,
    )


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        ruby_bin = shutil.which("ruby")
        if not ruby_bin:
            raise ConfigError(
                "PyYAML is not installed for the active interpreter and ruby is not available "
                f"to parse YAML configs: {path}"
            ) from _YAML_IMPORT_ERROR

        proc = subprocess.run(
            [
                ruby_bin,
                "-rjson",
                "-ryaml",
                "-e",
                "payload = YAML.load_file(ARGV[0]) || {}; print JSON.dump(payload)",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise ConfigError(
                "failed to parse YAML config via ruby fallback: "
                f"{path}: {detail or f'rc={proc.returncode}'}"
            ) from _YAML_IMPORT_ERROR
        try:
            loaded = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ConfigError(f"ruby YAML fallback produced invalid JSON for {path}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return loaded


def _resolve_child_config_path(parent_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent_path.parent / path
    return path.resolve()


def _deep_merge_mappings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_config_tree(path: Path, *, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"config extends cycle detected: {cycle}")

    loaded = _read_yaml_mapping(path)
    extends_value = loaded.pop("extends", None)
    if extends_value is None:
        return loaded

    if isinstance(extends_value, str):
        extend_paths = [extends_value]
    elif isinstance(extends_value, list) and all(isinstance(item, str) for item in extends_value):
        extend_paths = list(extends_value)
    else:
        raise ConfigError(f"config key extends must be a string or list of strings: {path}")

    merged: dict[str, Any] = {}
    for extend_path in extend_paths:
        base_path = _resolve_child_config_path(path, extend_path)
        if not base_path.is_file():
            raise ConfigError(f"missing extended config: {base_path}")
        merged = _deep_merge_mappings(
            merged,
            _load_yaml_config_tree(base_path, stack=(*stack, path)),
        )
    return _deep_merge_mappings(merged, loaded)


def load_stage_config(config_path: str | Path, expected_stage: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise ConfigError(f"missing config: {path}")

    loaded = _load_yaml_config_tree(path)

    stage_name = get_str(loaded, "stage")
    if stage_name != expected_stage:
        raise ConfigError(
            f"config stage mismatch: expected {expected_stage!r}, got {stage_name!r} in {path}"
        )

    return path, loaded


def load_yaml_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_config_path(config_path)
    if not path.is_file():
        raise ConfigError(f"missing config: {path}")
    return path, _load_yaml_config_tree(path)


def get_value(config: dict[str, Any], *keys: str, default: Any = _MISSING) -> Any:
    node: Any = config
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            if default is not _MISSING:
                return default
            raise ConfigError(f"missing config key: {'.'.join(walked)}")
        node = node[key]
    return node


def get_mapping(config: dict[str, Any], *keys: str, default: Any = _MISSING) -> dict[str, Any]:
    value = get_value(config, *keys, default=default)
    if value is default:
        return value
    if not isinstance(value, dict):
        raise ConfigError(f"config key {'.'.join(keys)} must be a mapping")
    return value


def get_str(
    config: dict[str, Any],
    *keys: str,
    default: Any = _MISSING,
    allow_empty: bool = False,
) -> str:
    value = get_value(config, *keys, default=default)
    if value is default:
        return value
    if value is None:
        value = ""
    elif not isinstance(value, (str, int, float, bool)):
        raise ConfigError(f"config key {'.'.join(keys)} must be a scalar string-like value")

    text = str(value)
    if not allow_empty and text == "":
        raise ConfigError(f"config key {'.'.join(keys)} must not be empty")
    return text


def get_int(config: dict[str, Any], *keys: str, default: Any = _MISSING) -> int:
    value = get_value(config, *keys, default=default)
    if value is default:
        return value
    if isinstance(value, bool):
        raise ConfigError(f"config key {'.'.join(keys)} must be an integer, not a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"config key {'.'.join(keys)} must be an integer") from exc
    raise ConfigError(f"config key {'.'.join(keys)} must be an integer")


def get_float(config: dict[str, Any], *keys: str, default: Any = _MISSING) -> float:
    value = get_value(config, *keys, default=default)
    if value is default:
        return value
    if isinstance(value, bool):
        raise ConfigError(f"config key {'.'.join(keys)} must be a float, not a bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"config key {'.'.join(keys)} must be a float") from exc
    raise ConfigError(f"config key {'.'.join(keys)} must be a float")


def get_bool(config: dict[str, Any], *keys: str, default: Any = _MISSING) -> bool:
    value = get_value(config, *keys, default=default)
    if value is default:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ConfigError(f"config key {'.'.join(keys)} must be a boolean")


def resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None or value == "":
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str, log_fp: Any | None = None) -> None:
    line = f"{timestamp()} {message}"
    print(line, flush=True)
    if log_fp is not None:
        print(line, file=log_fp, flush=True)


def run_command(cmd: list[str], *, log_fp: Any | None = None) -> None:
    log(f"[run] {format_cmd(cmd)}", log_fp=log_fp)
    subprocess.run(cmd, check=True)


def open_stage_log(log_folder: Path | None, filename: str) -> Any | None:
    if log_folder is None:
        return None
    log_folder.mkdir(parents=True, exist_ok=True)
    return (log_folder / filename).open("a", encoding="utf-8", buffering=1)


def exit_with_config_error(exc: ConfigError) -> int:
    print(f"config error: {exc}", file=sys.stderr)
    return 2
