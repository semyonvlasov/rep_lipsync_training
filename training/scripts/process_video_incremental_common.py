#!/usr/bin/env python3
"""
Common helpers for incremental raw -> normalized -> processed video pipelines.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import json
import os
import time

from data.audio import AudioProcessor
from scripts.preprocess_dataset import process_video, resolve_detector_device
from scripts.transcode_video import media_file_is_valid, normalize_video_clip

_AUDIO_PROC = None
FAIL_RESULTS = {"transcode_fail", "process_fail", "worker_fail"}


def list_raw_videos(raw_dir):
    items = []
    if not os.path.isdir(raw_dir):
        return items
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".mp4"):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        items.append((mtime, fname))
    items.sort(key=lambda item: (item[0], item[1]))
    return items


def failed_manifest_path(processed_dir):
    return os.path.join(processed_dir, "_failed_samples.jsonl")


def load_failed_names(processed_dir):
    failed = set()
    manifest_path = failed_manifest_path(processed_dir)
    if not os.path.exists(manifest_path):
        return failed

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if name:
                failed.add(str(name))
    return failed


def append_failed_sample(processed_dir, name, result, message, raw_path):
    os.makedirs(processed_dir, exist_ok=True)
    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "result": result,
        "message": message,
        "raw_path": raw_path,
    }
    with open(failed_manifest_path(processed_dir), "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def processed_exists(processed_dir, name):
    speaker_dir = os.path.join(processed_dir, name)
    frames_path = os.path.join(speaker_dir, "frames.npy")
    mel_path = os.path.join(speaker_dir, "mel.npy")
    meta_path = os.path.join(speaker_dir, "bbox.json")
    if not (
        os.path.exists(frames_path)
        and os.path.exists(mel_path)
        and os.path.exists(meta_path)
    ):
        return False
    try:
        with open(meta_path) as f:
            json.load(f)
    except Exception:
        return False
    return True


def clip_exists(clips_dir, name, ffmpeg_bin=None):
    clip_path = os.path.join(clips_dir, f"{name}.mp4")
    return media_file_is_valid(clip_path, ffmpeg_bin)


def build_audio_processor():
    audio_cfg = {
        "sample_rate": 16000,
        "n_fft": 800,
        "hop_size": 200,
        "win_size": 800,
        "n_mels": 80,
        "fmin": 55,
        "fmax": 7600,
        "preemphasis": 0.97,
    }
    return AudioProcessor(audio_cfg)


def get_audio_processor():
    global _AUDIO_PROC
    if _AUDIO_PROC is None:
        _AUDIO_PROC = build_audio_processor()
    return _AUDIO_PROC


def process_one(raw_path, clips_dir, processed_dir, audio_proc, args):
    name = os.path.splitext(os.path.basename(raw_path))[0]
    clip_path = os.path.join(clips_dir, f"{name}.mp4")

    if processed_exists(processed_dir, name):
        return "skip_processed", f"{name}: already in processed"

    if not clip_exists(clips_dir, name, args.ffmpeg_bin):
        ok, detail, selected_encoder = normalize_video_clip(
            raw_path,
            clip_path,
            fps=args.fps,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_threads=args.ffmpeg_threads,
            video_encoder=args.video_encoder,
            video_bitrate=args.video_bitrate,
            timeout=args.ffmpeg_timeout,
        )
        if not ok:
            return (
                "transcode_fail",
                f"{name}: FAIL transcode encoder={selected_encoder} detail={detail}",
            )

    _, status, elapsed = process_video(
        clip_path,
        processed_dir,
        args.size,
        audio_proc,
        max_frames=args.max_frames,
        detect_every=args.detect_every,
        smooth_window=args.smooth_window,
        smoothing_style=getattr(args, "smoothing_style", "official_inference"),
        framing_style=getattr(args, "framing_style", "official_inference"),
        inference_pads=tuple(getattr(args, "inference_pads", (0, 10, 0, 0))),
        save_preview=not args.no_preview,
        overwrite=False,
        ffmpeg_bin=args.ffmpeg_bin,
        ffmpeg_threads=args.ffmpeg_threads,
        detector_backend=args.detector_backend,
        detector_device=args.detector_device,
        detector_batch_size=args.detector_batch_size,
        min_detector_score=getattr(args, "min_detector_score", 0.0),
        resize_device=args.resize_device,
    )
    if status == "skip":
        return "skip_processed", f"{name}: already in processed"
    if status.startswith("ok"):
        return "ok", f"{name}: {status} ({elapsed:.1f}s)"
    return "process_fail", f"{name}: FAIL {status} ({elapsed:.1f}s)"


def _worker_args_from_dict(args_dict):
    class WorkerArgs:
        pass

    args = WorkerArgs()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args


def process_one_worker(raw_path, clips_dir, processed_dir, args_dict):
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    args = _worker_args_from_dict(args_dict)
    audio_proc = get_audio_processor()
    return process_one(raw_path, clips_dir, processed_dir, audio_proc, args)


def process_one_thread(raw_path, clips_dir, processed_dir, args_dict):
    args = _worker_args_from_dict(args_dict)
    audio_proc = get_audio_processor()
    return process_one(raw_path, clips_dir, processed_dir, audio_proc, args)


def build_args_dict(args):
    return {
        "fps": args.fps,
        "ffmpeg_bin": args.ffmpeg_bin,
        "ffmpeg_threads": args.ffmpeg_threads,
        "ffmpeg_timeout": args.ffmpeg_timeout,
        "video_encoder": args.video_encoder,
        "video_bitrate": args.video_bitrate,
        "size": args.size,
        "max_frames": args.max_frames,
        "detect_every": args.detect_every,
        "smooth_window": args.smooth_window,
        "smoothing_style": getattr(args, "smoothing_style", "official_inference"),
        "framing_style": getattr(args, "framing_style", "official_inference"),
        "inference_pads": tuple(getattr(args, "inference_pads", (0, 10, 0, 0))),
        "no_preview": args.no_preview,
        "detector_backend": args.detector_backend,
        "detector_device": args.detector_device,
        "detector_batch_size": args.detector_batch_size,
        "min_detector_score": getattr(args, "min_detector_score", 0.0),
        "resize_device": args.resize_device,
    }


def run_executor(executor, worker_fn, candidates, args, args_dict, log_prefix):
    futures = {}
    next_submit = 0
    total = len(candidates)
    failed_names = set() if getattr(args, "retry_failed", False) else load_failed_names(args.processed_dir)

    while next_submit < total or futures:
        while next_submit < total and len(futures) < args.workers:
            raw_path = candidates[next_submit]
            try:
                added_ts = os.path.getmtime(raw_path)
                added_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(added_ts))
            except OSError:
                added_str = "unknown"
            future = executor.submit(
                worker_fn,
                raw_path,
                args.clips_dir,
                args.processed_dir,
                args_dict,
            )
            futures[future] = (next_submit + 1, total, raw_path, added_str)
            next_submit += 1

        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            submit_idx, total, raw_path, added_str = futures.pop(future)
            name = os.path.splitext(os.path.basename(raw_path))[0]
            print(
                f"[{log_prefix}] [{submit_idx}/{total}] {name} "
                f"(raw_mtime={added_str})",
                flush=True,
            )
            try:
                result, message = future.result()
            except Exception as exc:
                result = "worker_fail"
                message = f"{name}: FAIL worker_exc {type(exc).__name__}: {exc}"
            if result in FAIL_RESULTS and name not in failed_names:
                append_failed_sample(args.processed_dir, name, result, message, raw_path)
                failed_names.add(name)
            print(f"[{log_prefix}] {result}: {message}", flush=True)


def run_candidates(candidates, args, log_prefix):
    failed_names = set() if getattr(args, "retry_failed", False) else load_failed_names(args.processed_dir)

    if args.workers <= 1:
        audio_proc = get_audio_processor()
        for idx, raw_path in enumerate(candidates, start=1):
            name = os.path.splitext(os.path.basename(raw_path))[0]
            try:
                added_ts = os.path.getmtime(raw_path)
                added_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(added_ts))
            except OSError:
                added_str = "unknown"
            print(
                f"[{log_prefix}] [{idx}/{len(candidates)}] {name} "
                f"(raw_mtime={added_str})",
                flush=True,
            )
            result, message = process_one(raw_path, args.clips_dir, args.processed_dir, audio_proc, args)
            if result in FAIL_RESULTS and name not in failed_names:
                append_failed_sample(args.processed_dir, name, result, message, raw_path)
                failed_names.add(name)
            print(f"[{log_prefix}] {result}: {message}", flush=True)
        return

    args_dict = build_args_dict(args)

    worker_backend = args.worker_backend
    thread_workers = args.workers
    if worker_backend in ("auto", "process"):
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                if worker_backend == "auto":
                    print(f"[{log_prefix}] worker_backend=process", flush=True)
                run_executor(executor, process_one_worker, candidates, args, args_dict, log_prefix)
                return
        except PermissionError as exc:
            if worker_backend == "process":
                raise
            print(
                f"[{log_prefix}] worker_backend=process unavailable "
                f"({type(exc).__name__}: {exc}); falling back to threads",
                flush=True,
            )

    if args.detector_backend == "sfd" and thread_workers > 1:
        print(
            f"[{log_prefix}] detector=sfd with thread backend is memory-heavy; "
            "forcing thread_workers=1",
            flush=True,
        )
        thread_workers = 1

    print(f"[{log_prefix}] worker_backend=thread", flush=True)
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        run_executor(executor, process_one_thread, candidates, args, args_dict, log_prefix)


def run_incremental_loop(args, log_prefix="Incremental"):
    idle_cycles = 0
    while True:
        raw_items = list_raw_videos(args.raw_dir)
        failed_names = set() if getattr(args, "retry_failed", False) else load_failed_names(args.processed_dir)
        candidates = [
            os.path.join(args.raw_dir, fname)
            for _, fname in raw_items
            if not processed_exists(args.processed_dir, os.path.splitext(fname)[0])
            and os.path.splitext(fname)[0] not in failed_names
        ]

        if not candidates:
            idle_cycles += 1
            print(
                f"[{log_prefix}] No pending raw videos "
                f"(idle_cycle={idle_cycles}, follow={args.follow}, failed_skipped={len(failed_names)})",
                flush=True,
            )
            if not args.follow:
                break
            if args.idle_exit_cycles > 0 and idle_cycles >= args.idle_exit_cycles:
                print(f"[{log_prefix}] Idle exit threshold reached, stopping", flush=True)
                break
            time.sleep(max(args.poll_seconds, 1))
            continue

        idle_cycles = 0
        print(
            f"[{log_prefix}] Pending raw videos: {len(candidates)} "
            f"(failed_skipped={len(failed_names)})",
            flush=True,
        )
        run_candidates(candidates, args, log_prefix)

        if not args.follow:
            break

    print(f"[{log_prefix}] Done", flush=True)


def print_startup_banner(args, log_prefix="Incremental"):
    print(f"[{log_prefix}] raw={args.raw_dir}")
    print(f"[{log_prefix}] clips={args.clips_dir}")
    print(f"[{log_prefix}] processed={args.processed_dir}")
    print(f"[{log_prefix}] ffmpeg={args.ffmpeg_bin}")
    print(f"[{log_prefix}] ffmpeg_threads={args.ffmpeg_threads}")
    print(f"[{log_prefix}] ffmpeg_timeout={args.ffmpeg_timeout}")
    print(f"[{log_prefix}] video_encoder={args.video_encoder}")
    print(f"[{log_prefix}] video_bitrate={args.video_bitrate}")
    print(f"[{log_prefix}] workers={args.workers}")
    print(f"[{log_prefix}] worker_backend={args.worker_backend}")
    print(
        f"[{log_prefix}] detector={args.detector_backend}"
        f" device={resolve_detector_device(args.detector_backend, args.detector_device)}"
    )
    print(f"[{log_prefix}] detector_batch_size={args.detector_batch_size}")
    print(f"[{log_prefix}] resize_device={args.resize_device}")
    print(f"[{log_prefix}] retry_failed={getattr(args, 'retry_failed', False)}")
    print(f"[{log_prefix}] failed_manifest={failed_manifest_path(args.processed_dir)}")
    print(f"[{log_prefix}] order=oldest_first_by_mtime")
