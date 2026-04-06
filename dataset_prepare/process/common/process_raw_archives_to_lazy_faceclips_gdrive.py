#!/usr/bin/env python3
"""
Sequentially rebuild lazy faceclip archives from raw HDTF/TalkVid clip batches on
Google Drive.

Workflow per archive:
  1. list the first pending `*.tar` archive in the source folder
  2. atomically claim it on Drive by renaming to `*.tar.processed`
  3. download the claimed archive locally
  4. extract it
  5. export lazy faceclips with official-style framing:
       - SFD detector
       - detect every frame
       - official temporal smoothing with T=5
       - face-only crop at 288x288
  6. pack the processed faceclips into `hdtf_faceclips_*.tar` /
     `talkvid_faceclips_*.tar`
  7. upload the processed archive to the destination folder
  8. clean local staging and continue

The loop exits when there are no more non-`.processed` `*.tar` archives left
in the source Drive folder.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import tarfile
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.process.common.pipeline_utils import (
    append_failure_event,
    append_jsonl,
    append_state_event,
    build_faceclip_export_cmd,
    cleanup_paths,
    count_exported_samples,
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


DEFAULT_SOURCE_FOLDER_ID = "1v06momk8fR-eqw79Z93zczBx_InWsCS9"
DEFAULT_DEST_FOLDER_ID = "1xx2IlfiAYC1AFf3xJwcjeTEsAK-Uqt8n"
STATE_MANIFEST_BASENAME = "active_archive_state.json"


def try_claim_remote_archive(remote: str, folder_id: str, archive_name: str) -> tuple[bool, str]:
    claimed_name = archive_name + ".processed"
    cmd = [
        "rclone",
        "moveto",
        "--drive-root-folder-id",
        folder_id,
        f"{remote}{archive_name}",
        f"{remote}{claimed_name}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, claimed_name
    detail = (proc.stderr or proc.stdout or "").strip()
    detail = " ".join(detail.split()) if detail else f"rc={proc.returncode}"
    log(f"[RawFaceclipCycle:claim] skip {archive_name}: {detail}")
    return False, detail[-320:]


def guess_archive_kind(name: str) -> tuple[str, bool]:
    lower = name.lower()
    if "talkvid" in lower and "raw" in lower:
        return "talkvid", False
    if "hdtf" in lower and "clips" in lower:
        return "hdtf", True
    if "hdtf" in lower and "raw" in lower:
        return "hdtf", False
    if "clips" in lower and "raw" not in lower:
        return "hdtf", True
    return "talkvid", False


def processed_archive_name(source_archive: str) -> str:
    name = source_archive
    if name.endswith(".tar"):
        stem = name[:-4]
    else:
        stem = name
    stem = stem.replace("talkvid_raw_", "talkvid_faceclips_")
    stem = stem.replace("hdtf_clips_", "hdtf_faceclips_")
    stem = stem.replace("hdtf_raw_", "hdtf_faceclips_")
    if stem == name[:-4]:
        stem = stem + "_faceclips"
    return stem + ".tar"


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)


def pack_dir_to_tar(input_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(input_dir, arcname=input_dir.name)
def list_pending_source_archives(remote: str, folder_id: str, archive_glob: str) -> list[str]:
    names = rclone_lsf(remote, folder_id)
    return [
        name
        for name in names
        if name.endswith(".tar")
        and not name.endswith(".tar.processed")
        and fnmatch.fnmatch(name, archive_glob)
    ]
def build_archive_state(
    *,
    archive_name: str,
    claimed_name: str,
    output_name: str,
    dataset_kind: str,
    input_is_normalized: bool,
    local_tar: Path,
    extract_root: Path,
    normalize_root: Path,
    export_root: Path,
    processed_tar: Path,
    summary_path: Path,
) -> dict:
    return {
        "version": 1,
        "created_ts": timestamp(),
        "updated_ts": timestamp(),
        "stage": "claimed",
        "source_archive": archive_name,
        "claimed_archive": claimed_name,
        "processed_archive": output_name,
        "dataset_kind": dataset_kind,
        "input_is_normalized": bool(input_is_normalized),
        "local_tar": str(local_tar),
        "extract_root": str(extract_root),
        "normalize_root": str(normalize_root),
        "export_root": str(export_root),
        "processed_tar": str(processed_tar),
        "summary_path": str(summary_path),
    }
def build_export_cmd(args, export_script: Path, state: dict) -> list[str]:
    return build_faceclip_export_cmd(
        python_bin=args.python_bin,
        export_script=export_script,
        input_dir=Path(state["extract_root"]),
        output_dir=Path(state["export_root"]),
        normalized_dir=Path(state["normalize_root"]),
        source_archive=state["source_archive"],
        dataset_kind=state["dataset_kind"],
        size=args.size,
        fps=args.fps,
        max_frames=args.max_frames,
        detect_every=args.detect_every,
        smooth_window=args.smooth_window,
        detector_backend=args.detector_backend,
        detector_device=args.detector_device,
        detector_batch_size=args.detector_batch_size,
        resize_device=args.resize_device,
        ffmpeg_bin=args.ffmpeg_bin or "",
        ffmpeg_threads=args.ffmpeg_threads,
        ffmpeg_timeout=args.ffmpeg_timeout,
        video_encoder=args.video_encoder,
        normalized_video_bitrate=args.normalized_video_bitrate,
        video_bitrate=args.video_bitrate,
        smoothing_style=args.smoothing_style,
        framing_style=args.framing_style,
        min_detector_score=args.min_detector_score,
        input_is_normalized=bool(state.get("input_is_normalized")),
    )


def process_archive_state(
    *,
    args,
    state: dict,
    state_path: Path,
    manifest_path: Path,
    export_script: Path,
    dest_archives: set[str],
) -> None:
    archive_name = state["source_archive"]
    claimed_name = state["claimed_archive"]
    output_name = state["processed_archive"]
    local_tar = Path(state["local_tar"])
    extract_root = Path(state["extract_root"])
    normalize_root = Path(state["normalize_root"])
    export_root = Path(state["export_root"])
    processed_tar = Path(state["processed_tar"])
    summary_path = Path(state["summary_path"])

    stage = str(state.get("stage", "claimed"))

    if output_name in dest_archives and stage == "claimed":
        append_state_event(manifest_path, state, "skip_already_uploaded")
        cleanup_paths([local_tar, extract_root, normalize_root, export_root, processed_tar])
        remove_state_manifest(state_path)
        return

    if stage in {"claimed", "download_started"}:
        # Partial downloads are not resumable; start the claimed archive from a
        # clean local slate and fetch it again.
        cleanup_paths([local_tar, extract_root, normalize_root, export_root, processed_tar])
        log(f"[RawFaceclipCycle] downloading {claimed_name}")
        state = update_state_manifest(state_path, state, "download_started")
        run_logged(
            [
                "rclone",
                "copyto",
                "--drive-root-folder-id",
                args.source_folder_id,
                f"{args.remote}{claimed_name}",
                str(local_tar),
            ],
            prefix="[RawFaceclipCycle:rclone-download]",
        )
        append_state_event(
            manifest_path,
            state,
            "downloaded",
            local_tar=str(local_tar),
            bytes=local_tar.stat().st_size,
        )
        state = update_state_manifest(
            state_path,
            state,
            "downloaded",
            local_tar_bytes=local_tar.stat().st_size,
        )
        stage = "downloaded"

    if stage == "downloaded":
        cleanup_paths([extract_root, normalize_root, export_root, processed_tar])
        if not local_tar.exists():
            state = update_state_manifest(state_path, state, "claimed")
            return process_archive_state(
                args=args,
                state=state,
                state_path=state_path,
                manifest_path=manifest_path,
                export_script=export_script,
                dest_archives=dest_archives,
            )
        log(f"[RawFaceclipCycle] extracting {archive_name}")
        extract_tar(local_tar, extract_root)
        append_state_event(manifest_path, state, "extracted", extract_root=str(extract_root))
        state = update_state_manifest(state_path, state, "extracted")
        stage = "extracted"

    if stage in {"extracted", "process_started"}:
        if not extract_root.exists():
            if local_tar.exists():
                extract_tar(local_tar, extract_root)
                append_state_event(manifest_path, state, "resume_reextracted", extract_root=str(extract_root))
            else:
                state = update_state_manifest(state_path, state, "claimed")
                return process_archive_state(
                    args=args,
                    state=state,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    export_script=export_script,
                    dest_archives=dest_archives,
                )

        log(f"[RawFaceclipCycle] exporting {archive_name} -> {output_name}")
        state = update_state_manifest(state_path, state, "process_started")
        run_logged(
            build_export_cmd(args, export_script, state),
            prefix="[RawFaceclipCycle:export]",
        )

        summary = load_summary(summary_path)
        exported_samples = count_exported_samples(export_root)
        append_state_event(
            manifest_path,
            state,
            "processed",
            export_root=str(export_root),
            normalized_root=str(normalize_root),
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
        exported_samples = count_exported_samples(export_root)
        if exported_samples == 0:
            cleanup_paths([local_tar, extract_root, normalize_root, export_root, processed_tar])
            append_state_event(manifest_path, state, "cleaned_no_output", summary=summary)
            remove_state_manifest(state_path)
            return

        log(f"[RawFaceclipCycle] packaging {output_name}")
        pack_dir_to_tar(export_root, processed_tar)
        append_state_event(
            manifest_path,
            state,
            "packaged",
            processed_tar=str(processed_tar),
            bytes=processed_tar.stat().st_size,
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
            exported_samples = count_exported_samples(export_root)
            if exported_samples > 0:
                pack_dir_to_tar(export_root, processed_tar)
                append_state_event(manifest_path, state, "resume_repackaged", processed_tar=str(processed_tar))
            else:
                state = update_state_manifest(state_path, state, "extracted")
                return process_archive_state(
                    args=args,
                    state=state,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    export_script=export_script,
                    dest_archives=dest_archives,
                )

        if stage != "uploaded":
            log(f"[RawFaceclipCycle] uploading {output_name}")
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
                prefix="[RawFaceclipCycle:rclone-upload]",
            )
            dest_archives.add(output_name)
            summary = load_summary(summary_path)
            append_state_event(
                manifest_path,
                state,
                "uploaded",
                remote_name=output_name,
                summary=summary,
            )
            state = update_state_manifest(state_path, state, "uploaded")

        cleanup_paths([local_tar, extract_root, normalize_root, export_root, processed_tar])
        append_state_event(manifest_path, state, "uploaded_cleaned", remote_name=output_name)
        remove_state_manifest(state_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-folder-id", default=DEFAULT_SOURCE_FOLDER_ID)
    parser.add_argument("--dest-folder-id", default=DEFAULT_DEST_FOLDER_ID)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--data-root", default="data/dataset_prepare/process/raw_faceclips_x288_cycle")
    parser.add_argument(
        "--manifest-path",
        default="data/dataset_prepare/process/raw_faceclips_x288_cycle/archive_manifest.jsonl",
    )
    parser.add_argument("--archive-glob", default="*.tar")
    parser.add_argument("--max-archives", type=int, default=0, help="0=all pending")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--smoothing-style",
        choices=["legacy_centered", "official_inference", "none"],
        default="official_inference",
    )
    parser.add_argument(
        "--framing-style",
        choices=["legacy_square", "official_inference"],
        default="official_inference",
    )
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="sfd")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--min-detector-score", type=float, default=0.99999)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=4)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--normalized-video-bitrate", default="")
    parser.add_argument("--video-bitrate", default="600k")
    parser.add_argument("--keep-failed-artifacts", action="store_true")
    args = parser.parse_args()
    args.normalized_video_bitrate = args.normalized_video_bitrate or args.video_bitrate

    export_script = Path(__file__).with_name("export_faceclip_batch.py").resolve()
    root = Path(args.data_root)
    downloads_dir = root / "downloads"
    extracted_dir = root / "extracted"
    normalized_dir = root / "normalized"
    processed_dir = root / "processed_work"
    archives_dir = root / "archives"
    manifest_path = Path(args.manifest_path)
    state_path = manifest_path.with_name(STATE_MANIFEST_BASENAME)

    for path in (downloads_dir, extracted_dir, normalized_dir, processed_dir, archives_dir):
        path.mkdir(parents=True, exist_ok=True)

    try:
        dest_archives = set(rclone_lsf(args.remote, args.dest_folder_id))
    except Exception:
        dest_archives = set()

    log(f"[RawFaceclipCycle] source_folder_id={args.source_folder_id}")
    log(f"[RawFaceclipCycle] dest_folder_id={args.dest_folder_id}")
    log(
        f"[RawFaceclipCycle] export_defaults size={args.size} detector={args.detector_backend} "
        f"detect_every={args.detect_every} smoothing={args.smoothing_style} "
        f"smooth_window={args.smooth_window} framing={args.framing_style}"
    )
    log(f"[RawFaceclipCycle] state_manifest={state_path}")

    processed_count = 0
    while True:
        if args.max_archives and processed_count >= args.max_archives:
            log(f"[RawFaceclipCycle] reached max_archives={args.max_archives}")
            break

        active_state = load_state_manifest(state_path)
        if active_state is not None:
            log(
                f"[RawFaceclipCycle] resuming {active_state.get('source_archive')} "
                f"stage={active_state.get('stage')}"
            )
            try:
                process_archive_state(
                    args=args,
                    state=active_state,
                    state_path=state_path,
                    manifest_path=manifest_path,
                    export_script=export_script,
                    dest_archives=dest_archives,
                )
                processed_count += 1
                continue
            except Exception as exc:
                append_failure_event(manifest_path, active_state, "resume_failed", exc)
                log(
                    f"[RawFaceclipCycle] resume failed "
                    f"{active_state.get('source_archive')}: {type(exc).__name__}: {exc}"
                )
                raise

        pending = list_pending_source_archives(args.remote, args.source_folder_id, args.archive_glob)
        if not pending:
            break

        archive_name = pending[0]
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "stage": "claim_started",
            },
        )
        claimed_ok, claim_value = try_claim_remote_archive(args.remote, args.source_folder_id, archive_name)
        if not claimed_ok:
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "stage": "claim_failed",
                    "detail": claim_value,
                },
            )
            continue

        claimed_name = claim_value
        dataset_kind, input_is_normalized = guess_archive_kind(archive_name)
        output_name = processed_archive_name(archive_name)
        local_tar = downloads_dir / archive_name
        archive_stem = archive_name[:-4] if archive_name.endswith(".tar") else archive_name
        extract_root = extracted_dir / archive_stem
        export_root = processed_dir / archive_stem
        normalize_root = normalized_dir / archive_stem
        processed_tar = archives_dir / output_name
        summary_path = export_root / "summary.json"
        state = build_archive_state(
            archive_name=archive_name,
            claimed_name=claimed_name,
            output_name=output_name,
            dataset_kind=dataset_kind,
            input_is_normalized=input_is_normalized,
            local_tar=local_tar,
            extract_root=extract_root,
            normalize_root=normalize_root,
            export_root=export_root,
            processed_tar=processed_tar,
            summary_path=summary_path,
        )
        write_state_manifest(state_path, state)

        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "claimed_archive": claimed_name,
                "processed_archive": output_name,
                "dataset_kind": dataset_kind,
                "stage": "claimed",
            },
        )

        try:
            process_archive_state(
                args=args,
                state=state,
                state_path=state_path,
                manifest_path=manifest_path,
                export_script=export_script,
                dest_archives=dest_archives,
            )
            processed_count += 1
        except Exception as exc:
            append_failure_event(manifest_path, state, "failed", exc)
            log(f"[RawFaceclipCycle] failed {archive_name}: {type(exc).__name__}: {exc}")
            raise

    log(f"[RawFaceclipCycle] done processed_archives={processed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
