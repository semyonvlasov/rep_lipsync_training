#!/usr/bin/env python3
"""
Process completed local fetch batches into lazy faceclip archives, upload them,
and resume safely after failures via an active batch state manifest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.process.common.pipeline_utils import (
    append_failure_event,
    append_state_event,
    build_faceclip_export_cmd,
    cleanup_paths,
    count_exported_samples,
    load_json,
    load_latest_state,
    load_state_manifest,
    load_summary,
    log,
    pack_dir_to_tar,
    rclone_lsf,
    remove_state_manifest,
    run_logged,
    timestamp,
    update_state_manifest,
    write_state_manifest,
)


STATE_MANIFEST_BASENAME = "active_batch_state.json"
TERMINAL_STAGES = {"uploaded_cleaned", "cleaned_no_output", "skip_already_uploaded"}


def processed_archive_name(batch_name: str, prefix: str) -> str:
    return f"{prefix}{batch_name}.tar"


def iter_completed_batches(batches_dir: Path, batch_glob: str, complete_flag_name: str):
    for batch_root in sorted(batches_dir.glob(batch_glob)):
        if not batch_root.is_dir():
            continue
        if not (batch_root / complete_flag_name).exists():
            continue
        yield batch_root


def build_batch_state(
    *,
    batch_name: str,
    batch_root: Path,
    raw_dir: Path,
    source_archive: str,
    output_name: str,
    dataset_kind: str,
    export_root: Path,
    normalize_root: Path,
    processed_tar: Path,
    summary_path: Path,
    fetch_meta: dict,
    stage: str,
) -> dict:
    return {
        "version": 1,
        "created_ts": timestamp(),
        "updated_ts": timestamp(),
        "stage": stage,
        "batch_name": batch_name,
        "batch_root": str(batch_root),
        "raw_dir": str(raw_dir),
        "source_archive": source_archive,
        "processed_archive": output_name,
        "dataset_kind": dataset_kind,
        "export_root": str(export_root),
        "normalize_root": str(normalize_root),
        "processed_tar": str(processed_tar),
        "summary_path": str(summary_path),
        "fetch_meta": fetch_meta,
    }


def derive_resume_stage(previous_state: dict, export_root: Path, processed_tar: Path, summary_path: Path) -> str:
    previous_stage = str(previous_state.get("stage") or "")
    if previous_stage in TERMINAL_STAGES:
        return previous_stage
    if previous_stage == "uploaded":
        return "uploaded"
    if previous_stage == "upload_started":
        return "packaged" if processed_tar.exists() else "processed"
    if previous_stage == "packaged":
        return "packaged" if processed_tar.exists() else "processed"
    if previous_stage == "processed":
        if summary_path.exists() and count_exported_samples(export_root) > 0:
            return "processed"
    if previous_stage == "process_started":
        return "process_started"
    return "selected"


def process_batch_state(
    *,
    args,
    state: dict,
    state_path: Path,
    manifest_path: Path,
    export_script: Path,
    dest_archives: set[str],
) -> None:
    batch_name = state["batch_name"]
    output_name = state["processed_archive"]
    source_archive = state["source_archive"]
    batch_root = Path(state["batch_root"])
    raw_dir = Path(state["raw_dir"])
    batch_export_root = Path(state["export_root"])
    batch_normalized_root = Path(state["normalize_root"])
    processed_tar = Path(state["processed_tar"])
    summary_path = Path(state["summary_path"])
    stage = str(state.get("stage", "selected"))

    if output_name in dest_archives:
        append_state_event(
            manifest_path,
            state,
            "skip_already_uploaded",
            remote_name=output_name,
        )
        cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
        remove_state_manifest(state_path)
        return

    if stage in {"selected", "process_started"}:
        if not raw_dir.exists():
            raise FileNotFoundError(f"missing_raw_dir: {raw_dir}")

        cleanup_paths([batch_export_root, batch_normalized_root, processed_tar])
        log(f"[LocalFaceclipCycle] exporting {batch_name} -> {output_name}")
        state = update_state_manifest(state_path, state, "process_started")
        run_logged(
            build_faceclip_export_cmd(
                config_path=Path(args.process_config),
                python_bin=args.python_bin,
                export_script=export_script,
                input_dir=raw_dir,
                output_dir=batch_export_root,
                normalized_dir=batch_normalized_root,
                source_archive=source_archive,
                dataset_kind=args.dataset_kind,
            ),
            prefix="[LocalFaceclipCycle:export]",
        )

        summary = load_summary(summary_path)
        exported_samples = count_exported_samples(batch_export_root)
        append_state_event(
            manifest_path,
            state,
            "processed",
            batch_root=str(batch_root),
            export_root=str(batch_export_root),
            normalized_root=str(batch_normalized_root),
            summary=summary,
            exported_samples=exported_samples,
        )
        state = update_state_manifest(
            state_path,
            state,
            "processed",
            exported_samples=exported_samples,
        )
        stage = "processed"

    if stage == "processed":
        summary = load_summary(summary_path)
        exported_samples = count_exported_samples(batch_export_root)
        if exported_samples == 0:
            log(f"[LocalFaceclipCycle] no usable samples in {batch_name}; cleaning local staging")
            cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
            append_state_event(manifest_path, state, "cleaned_no_output", summary=summary)
            remove_state_manifest(state_path)
            return

        if batch_normalized_root.exists():
            log(f"[LocalFaceclipCycle] cleaning normalized staging for {batch_name}")
            cleanup_paths([batch_normalized_root])

        log(f"[LocalFaceclipCycle] packaging {output_name}")
        pack_dir_to_tar(batch_export_root, processed_tar)
        append_state_event(
            manifest_path,
            state,
            "packaged",
            processed_tar=str(processed_tar),
            bytes=processed_tar.stat().st_size,
            summary=summary,
        )
        state = update_state_manifest(
            state_path,
            state,
            "packaged",
            processed_tar_bytes=processed_tar.stat().st_size,
        )
        stage = "packaged"

    if stage in {"packaged", "upload_started", "uploaded"}:
        if not processed_tar.exists():
            exported_samples = count_exported_samples(batch_export_root)
            if exported_samples > 0:
                pack_dir_to_tar(batch_export_root, processed_tar)
                append_state_event(
                    manifest_path,
                    state,
                    "resume_repackaged",
                    processed_tar=str(processed_tar),
                )
                state = update_state_manifest(
                    state_path,
                    state,
                    "packaged",
                    processed_tar_bytes=processed_tar.stat().st_size,
                )
                stage = "packaged"
            else:
                state = update_state_manifest(state_path, state, "selected")
                return process_batch_state(
                    args=args,
                    state=state,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    export_script=export_script,
                    dest_archives=dest_archives,
                )

        summary = load_summary(summary_path)
        if stage != "uploaded":
            log(f"[LocalFaceclipCycle] uploading {output_name}")
            state = update_state_manifest(state_path, state, "upload_started")
            run_logged(
                [
                    "rclone",
                    "copyto",
                    "--drive-root-folder-id",
                    args.dest_folder_id,
                    str(processed_tar),
                    f"{args.remote}{output_name}",
                ],
                prefix="[LocalFaceclipCycle:rclone-upload]",
            )
            dest_archives.add(output_name)
            append_state_event(
                manifest_path,
                state,
                "uploaded",
                remote_name=output_name,
                summary=summary,
            )
            state = update_state_manifest(state_path, state, "uploaded")

        cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
        append_state_event(
            manifest_path,
            state,
            "uploaded_cleaned",
            remote_name=output_name,
            summary=summary,
        )
        remove_state_manifest(state_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--dest-folder-id", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--dataset-kind", default="talkvid")
    parser.add_argument("--source-archive-prefix", default="talkvid_raw_")
    parser.add_argument("--processed-archive-prefix", default="talkvid_faceclips_")
    parser.add_argument("--batch-glob", default="batch_*")
    parser.add_argument("--complete-flag-name", default="fetch_complete.json")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--producer-done-flag", default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0=all available")
    parser.add_argument("--process-config", required=True)
    args = parser.parse_args()

    batches_dir = Path(args.batches_dir)
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest_path)
    state_path = manifest_path.with_name(STATE_MANIFEST_BASENAME)
    producer_done_flag = Path(args.producer_done_flag) if args.producer_done_flag else None

    normalized_root = data_root / "normalized"
    export_root = data_root / "processed_work"
    archives_root = data_root / "archives"
    normalized_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)
    export_script = Path(__file__).with_name("export_faceclip_batch.py").resolve()

    try:
        dest_archives = set(rclone_lsf(args.remote, args.dest_folder_id))
    except Exception:
        dest_archives = set()

    log(f"[LocalFaceclipCycle] dataset_kind={args.dataset_kind}")
    log(f"[LocalFaceclipCycle] batches_dir={batches_dir}")
    log(f"[LocalFaceclipCycle] data_root={data_root}")
    log(f"[LocalFaceclipCycle] process_config={args.process_config}")
    log(f"[LocalFaceclipCycle] state_manifest={state_path}")
    log(f"[LocalFaceclipCycle] dest_archives={len(dest_archives)}")

    processed_batches = 0
    while True:
        active_state = load_state_manifest(state_path)
        if active_state is not None:
            log(
                f"[LocalFaceclipCycle] resuming {active_state.get('batch_name')} "
                f"stage={active_state.get('stage')}"
            )
            try:
                process_batch_state(
                    args=args,
                    state=active_state,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    export_script=export_script,
                    dest_archives=dest_archives,
                )
                processed_batches += 1
                continue
            except Exception as exc:
                append_failure_event(manifest_path, active_state, "resume_failed", exc)
                log(
                    f"[LocalFaceclipCycle] resume failed "
                    f"{active_state.get('batch_name')}: {type(exc).__name__}: {exc}"
                )
                raise

        latest_state = load_latest_state(manifest_path, "batch_name")
        pending = []
        for batch_root in iter_completed_batches(
            batches_dir,
            args.batch_glob,
            args.complete_flag_name,
        ):
            batch_name = batch_root.name
            output_name = processed_archive_name(batch_name, args.processed_archive_prefix)
            previous_state = latest_state.get(batch_name, {})
            final_stage = str(previous_state.get("stage") or "")
            if output_name in dest_archives or final_stage in TERMINAL_STAGES:
                continue
            pending.append((batch_root, batch_name, output_name, previous_state))

        if not pending:
            if args.follow and not (producer_done_flag and producer_done_flag.exists()):
                time.sleep(max(1, args.poll_seconds))
                continue
            break

        batch_root, batch_name, output_name, previous_state = pending[0]
        raw_dir = batch_root / "raw"
        fetch_meta = load_json(batch_root / args.complete_flag_name) or {}
        source_archive = f"{args.source_archive_prefix}{batch_name}.tar"
        work_name = output_name[:-4]
        batch_export_root = export_root / work_name
        batch_normalized_root = normalized_root / work_name
        processed_tar = archives_root / output_name
        summary_path = batch_export_root / "summary.json"
        initial_stage = derive_resume_stage(
            previous_state,
            batch_export_root,
            processed_tar,
            summary_path,
        )
        state = build_batch_state(
            batch_name=batch_name,
            batch_root=batch_root,
            raw_dir=raw_dir,
            source_archive=source_archive,
            output_name=output_name,
            dataset_kind=args.dataset_kind,
            export_root=batch_export_root,
            normalize_root=batch_normalized_root,
            processed_tar=processed_tar,
            summary_path=summary_path,
            fetch_meta=fetch_meta,
            stage=initial_stage,
        )
        write_state_manifest(state_path, state)

        if initial_stage == "selected":
            append_state_event(
                manifest_path,
                state,
                "selected",
                batch_root=str(batch_root),
                raw_dir=str(raw_dir),
                fetch_meta=fetch_meta,
            )
        else:
            append_state_event(
                manifest_path,
                state,
                "resume_selected",
                batch_root=str(batch_root),
                raw_dir=str(raw_dir),
                fetch_meta=fetch_meta,
                previous_stage=str(previous_state.get("stage") or ""),
                resume_stage=initial_stage,
            )

        try:
            process_batch_state(
                args=args,
                state=state,
                state_path=state_path,
                manifest_path=manifest_path,
                export_script=export_script,
                dest_archives=dest_archives,
            )
            processed_batches += 1
        except Exception as exc:
            append_failure_event(manifest_path, state, "failed", exc)
            log(f"[LocalFaceclipCycle] failed {batch_name}: {type(exc).__name__}: {exc}")
            raise

        if args.max_batches > 0 and processed_batches >= args.max_batches:
            break

    log(f"[LocalFaceclipCycle] done processed_batches={processed_batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
