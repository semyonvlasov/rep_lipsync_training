#!/usr/bin/env python3
"""
Download TalkVid clips from YouTube into a raw/ directory.

Uses the official TalkVid JSON metadata hosted on Hugging Face.
This script is download-only and intended to feed the existing incremental
transcode/preprocess pipeline later.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


TALKVID_METADATA_URLS = {
    "with_captions": "https://huggingface.co/datasets/FreedomIntelligence/TalkVid/resolve/main/data/filtered_video_clips_with_captions.json?download=true",
    "raw_filtered": "https://huggingface.co/datasets/FreedomIntelligence/TalkVid/resolve/main/data/filtered_video_clips.json?download=true",
}

SOURCE_FATAL_REASONS = {"video_unavailable", "private_video"}
RATE_LIMIT_MARKERS = (
    "rate-limited by youtube",
    "current session has been rate-limited by youtube",
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def decode_process_output(blob: object) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", "replace")
    return str(blob)


def summarize_failure_output(stdout: object, stderr: object, max_chars: int = 240) -> Optional[str]:
    combined = "\n".join(
        part for part in (decode_process_output(stderr), decode_process_output(stdout)) if part
    ).strip()
    if not combined:
        return None

    lines = [" ".join(line.split()) for line in combined.splitlines() if line.strip()]
    if not lines:
        return None

    for line in reversed(lines):
        lowered = line.lower()
        if (
            "error" in lowered
            or "unable to" in lowered
            or "unavailable" in lowered
            or "private video" in lowered
            or "rate-limit" in lowered
            or "confirm you're not a bot" in lowered
            or "confirm you’re not a bot" in lowered
        ):
            detail = line
            break
    else:
        detail = lines[-1]

    if len(detail) > max_chars:
        return detail[: max_chars - 3] + "..."
    return detail


def get_dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                total += os.path.getsize(fpath)
            except OSError:
                pass
    return total


def get_free_bytes(path: str) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def ensure_metadata(output_dir: str, variant: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, f"talkvid_{variant}.json")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    url = TALKVID_METADATA_URLS[variant]
    part_path = f"{local_path}.part"
    try:
        if os.path.exists(part_path):
            os.remove(part_path)
    except OSError:
        pass
    log(f"[TalkVid] Downloading metadata: {url}")
    with urlopen(url, timeout=120) as response, open(part_path, "wb") as out_f:
        shutil.copyfileobj(response, out_f)
    os.replace(part_path, local_path)
    return local_path


def iter_clips(json_path: str) -> Iterable[dict]:
    with open(json_path) as f:
        items = json.load(f)
    for item in items:
        if isinstance(item, dict):
            yield item


def load_completed_ids(manifest_paths: list[str]) -> set[str]:
    done: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path:
            continue
        path = Path(manifest_path)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                clip = obj.get("clip_id")
                if clip and obj.get("status") != "fail":
                    done.add(str(clip))
                for fname in obj.get("files", []):
                    base = os.path.basename(str(fname))
                    if base.endswith(".mp4"):
                        done.add(base[:-4])
    return done


def load_blocked_video_keys(manifest_paths: list[str]) -> set[str]:
    blocked: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path:
            continue
        path = Path(manifest_path)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("status") != "fail":
                    continue
                if obj.get("reason") not in SOURCE_FATAL_REASONS:
                    continue
                video_key = video_key_from_url(obj.get("video_link"))
                if video_key:
                    blocked.add(video_key)
    return blocked


def clip_id(item: dict) -> str:
    return str(item.get("id") or "").replace("/", "_")


def output_path_for(raw_dir: str, item: dict) -> str:
    return os.path.join(raw_dir, f"{clip_id(item)}.mp4")


def metadata_path_for(raw_dir: str, item: dict) -> str:
    return os.path.join(raw_dir, f"{clip_id(item)}.json")


def get_video_link(item: dict) -> Optional[str]:
    info = item.get("info") or {}
    return info.get("Video Link")


def video_key_from_url(video_url: Optional[str]) -> Optional[str]:
    if not video_url:
        return None

    try:
        parsed = urlparse(video_url)
    except Exception:
        return video_url

    host = (parsed.netloc or "").lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    if host in {"youtu.be", "www.youtu.be"}:
        return path_parts[0] if path_parts else video_url

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        query_video = parse_qs(parsed.query).get("v")
        if query_video and query_video[0]:
            return query_video[0]
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
            return path_parts[1]

    return video_url


def video_key_for_item(item: dict) -> Optional[str]:
    return video_key_from_url(get_video_link(item))


def clip_duration(item: dict) -> float:
    try:
        return float(item.get("end-time")) - float(item.get("start-time"))
    except Exception:
        return 0.0


def clip_allowed(item: dict, args: argparse.Namespace) -> bool:
    duration = clip_duration(item)
    if duration < args.min_duration or duration > args.max_duration:
        return False
    if float(item.get("height") or 0) < args.min_height:
        return False
    if float(item.get("width") or 0) < args.min_width:
        return False
    if float(item.get("dover_scores") or 0.0) < args.min_dover:
        return False
    if float(item.get("cotracker_ratio") or 0.0) < args.min_cotracker:
        return False
    return bool(get_video_link(item))


def write_sidecar(path: str, item: dict) -> None:
    with open(path, "w") as f:
        json.dump(item, f, ensure_ascii=True, indent=2)
        f.write("\n")


def append_manifest(manifest_path: str, payload: dict) -> None:
    with open(manifest_path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def cleanup_clip_outputs(output_path: str) -> None:
    sidecar_path = str(Path(output_path).with_suffix(".json"))
    for path in (output_path, sidecar_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def download_clip(
    item: dict,
    output_path: str,
    max_height: int,
    timeout: int,
    cookies_file: Optional[str],
    cookies_from_browser: Optional[str],
) -> tuple[bool, str, int, Optional[str]]:
    if os.path.exists(output_path):
        try:
            return True, "exists", os.path.getsize(output_path), None
        except OSError:
            return True, "exists", 0, None

    video_url = get_video_link(item)
    if not video_url:
        return False, "missing_video_link", 0, None

    try:
        start = float(item.get("start-time"))
        end = float(item.get("end-time"))
    except Exception:
        return False, "bad_time", 0, None

    tmp_dir = tempfile.mkdtemp(prefix="talkvid_dl_")
    tmp_template = os.path.join(tmp_dir, "clip.%(ext)s")
    try:
        cmd = [
            "yt-dlp",
            "--remote-components", "ejs:github",
            "-f",
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[height<={max_height}]",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{start:.3f}-{end:.3f}",
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--no-warnings",
            "--restrict-filenames",
            "--output", tmp_template,
            video_url,
        ]
        if cookies_file:
            cmd[1:1] = ["--cookies", cookies_file]
        elif cookies_from_browser:
            cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=timeout,
        )

        produced = None
        for fname in sorted(os.listdir(tmp_dir)):
            fpath = os.path.join(tmp_dir, fname)
            if os.path.isfile(fpath):
                produced = fpath
                break
        if not produced:
            return False, "no_output", 0, None

        final_sidecar_path = str(Path(output_path).with_suffix(".json"))
        sidecar_tmp_path = os.path.join(tmp_dir, "clip.json")
        write_sidecar(sidecar_tmp_path, item)

        # Publish the sidecar first and the mp4 last so packers that only look
        # at ready mp4+json pairs never see a half-written clip.
        os.replace(sidecar_tmp_path, final_sidecar_path)
        try:
            os.replace(produced, output_path)
        except Exception:
            cleanup_clip_outputs(output_path)
            raise
        try:
            size_bytes = os.path.getsize(output_path)
        except OSError:
            size_bytes = 0
        return True, "ok", size_bytes, None
    except subprocess.TimeoutExpired:
        return False, "timeout", 0, None
    except subprocess.CalledProcessError as exc:
        stderr = decode_process_output(exc.stderr).strip()
        stdout = decode_process_output(exc.stdout).strip()
        combined = "\n".join(part for part in (stderr, stdout) if part).strip()
        lowered = combined.lower()
        detail = summarize_failure_output(stdout, stderr)
        if any(marker in lowered for marker in RATE_LIMIT_MARKERS):
            return False, "rate_limited", 0, detail
        if "private video" in lowered:
            return False, "private_video", 0, detail
        if "video unavailable" in lowered or "this video is unavailable" in lowered:
            return False, "video_unavailable", 0, detail
        if "confirm you’re not a bot" in lowered or "confirm you're not a bot" in lowered:
            return False, "bot_check", 0, detail
        if "requested format is not available" in lowered:
            return False, "format_unavailable", 0, detail
        return False, "yt_dlp_error", 0, detail
    except OSError as exc:
        cleanup_clip_outputs(output_path)
        return False, "io_error", 0, str(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download TalkVid raw clips")
    parser.add_argument("--output", default="data/talkvid", help="Output root")
    parser.add_argument("--variant", choices=sorted(TALKVID_METADATA_URLS), default="with_captions")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=6.5)
    parser.add_argument("--min-width", type=int, default=720)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument("--min-dover", type=float, default=8.0)
    parser.add_argument("--min-cotracker", type=float, default=0.90)
    parser.add_argument("--max-clips", type=int, default=0, help="Max new clips to download (0=unlimited)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--target-additional-gb", type=float, default=0.0)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--manifest-name", default="download_manifest.jsonl")
    parser.add_argument(
        "--skip-manifest",
        action="append",
        default=[],
        help="JSONL manifest(s) containing completed clip ids or packaged filenames to skip",
    )
    parser.add_argument("--delay-seconds", type=int, default=0)
    parser.add_argument("--cookies-file", default=os.environ.get("YTDLP_COOKIES_FILE"))
    parser.add_argument("--cookies-from-browser", default=os.environ.get("YTDLP_COOKIES_FROM_BROWSER"))
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent yt-dlp jobs")
    parser.add_argument(
        "--rate-limit-cooldown-seconds",
        type=int,
        default=600,
        help="Cooldown before resuming after a YouTube rate limit (0=exit immediately)",
    )
    parser.add_argument(
        "--max-rate-limit-cooldowns",
        type=int,
        default=2,
        help="Maximum cooldown/resume attempts after rate limiting (0=disable retries)",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    if args.delay_seconds > 0:
        start_ts = time.time() + args.delay_seconds
        start_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(start_ts))
        log(f"[TalkVid] Delaying start for {args.delay_seconds}s until {start_str}")
        time.sleep(args.delay_seconds)

    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except Exception:
        log("[TalkVid] ERROR: yt-dlp is not installed")
        return 1

    output_root = args.output
    metadata_dir = os.path.join(output_root, "metadata")
    raw_dir = os.path.join(output_root, "raw")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    metadata_path = ensure_metadata(metadata_dir, args.variant)
    manifest_path = os.path.join(output_root, args.manifest_name)
    completed_ids = load_completed_ids(args.skip_manifest)
    blocked_video_keys = load_blocked_video_keys(args.skip_manifest)

    start_raw_bytes = get_dir_size_bytes(raw_dir)
    target_added_bytes = int(args.target_additional_gb * (1024 ** 3))
    min_free_bytes = int(args.min_free_gb * (1024 ** 3))
    jobs = max(1, args.jobs)

    log(f"[TalkVid] variant={args.variant}")
    log(f"[TalkVid] metadata={metadata_path}")
    log(f"[TalkVid] raw_dir={raw_dir}")
    log(f"[TalkVid] start_raw_gb={start_raw_bytes / (1024 ** 3):.2f}")
    log(f"[TalkVid] target_additional_gb={args.target_additional_gb:.2f}")
    log(f"[TalkVid] min_free_gb={args.min_free_gb:.2f}")
    log(f"[TalkVid] completed_ids={len(completed_ids)}")
    log(f"[TalkVid] blocked_video_keys={len(blocked_video_keys)}")
    log(f"[TalkVid] jobs={jobs}")
    log(f"[TalkVid] rate_limit_cooldown_seconds={args.rate_limit_cooldown_seconds}")
    log(f"[TalkVid] max_rate_limit_cooldowns={args.max_rate_limit_cooldowns}")
    if args.cookies_file:
        log(f"[TalkVid] cookies_file={args.cookies_file}")
    elif args.cookies_from_browser:
        log(f"[TalkVid] cookies_from_browser={args.cookies_from_browser}")

    attempted = 0
    downloaded = 0
    downloaded_bytes = 0
    skipped_existing = 0
    skipped_blocked_source = 0
    filtered_out = 0
    failures = 0
    interrupted = False
    stop_submitting = False
    bot_check_hit = False
    rate_limited_hit = False
    rate_limit_exit = False
    rate_limit_events = 0
    rate_limit_cooldowns_used = 0
    target_reached_hit = False
    bot_check_notice_sent = False
    rate_limit_notice_sent = False

    def log_result(
        attempt_index: int,
        item: dict,
        ok: bool,
        reason: str,
        size_bytes: int,
        detail: Optional[str],
    ) -> None:
        nonlocal downloaded, downloaded_bytes, failures, bot_check_hit
        nonlocal rate_limited_hit, stop_submitting, rate_limit_events
        nonlocal bot_check_notice_sent, rate_limit_notice_sent

        item_id = clip_id(item)
        out_path = output_path_for(raw_dir, item)
        info = item.get("info") or {}
        duration = clip_duration(item)
        video_key = video_key_for_item(item)
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "clip_id": item_id,
            "status": "ok" if ok else "fail",
            "reason": reason,
            "output_path": out_path,
            "start": item.get("start-time"),
            "end": item.get("end-time"),
            "video_link": get_video_link(item),
            "video_key": video_key,
        }
        if detail:
            payload["detail"] = detail
        append_manifest(manifest_path, payload)

        if ok:
            downloaded += 1
            downloaded_bytes += size_bytes
            size_mb = size_bytes / (1024 ** 2)
            log(
                f"[TalkVid] [{attempt_index}] {item_id} "
                f"dur={duration:.2f}s lang={info.get('Language')} "
                f"dover={float(item.get('dover_scores') or 0.0):.2f} "
                f"cotracker={float(item.get('cotracker_ratio') or 0.0):.3f} "
                f"OK ({size_mb:.1f} MB)"
            )
            return

        failures += 1
        cleanup_clip_outputs(out_path)
        detail_suffix = f": {detail}" if detail else ""
        log(
            f"[TalkVid] [{attempt_index}] {item_id} "
            f"dur={duration:.2f}s lang={info.get('Language')} "
            f"dover={float(item.get('dover_scores') or 0.0):.2f} "
            f"cotracker={float(item.get('cotracker_ratio') or 0.0):.3f} "
            f"FAIL ({reason}{detail_suffix})"
        )
        if reason == "bot_check":
            bot_check_hit = True
            stop_submitting = True
            bot_check_notice_sent = False
        if reason == "rate_limited":
            rate_limited_hit = True
            stop_submitting = True
            rate_limit_events += 1
            rate_limit_notice_sent = False
        if reason in SOURCE_FATAL_REASONS and video_key and video_key not in blocked_video_keys:
            blocked_video_keys.add(video_key)
            log(f"[TalkVid] blocking source video_key={video_key} after {reason}")

    try:
        clip_iter = iter_clips(metadata_path)
        pending: dict = {}
        exhausted = False
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            while pending or not exhausted:
                while not exhausted and not stop_submitting and len(pending) < jobs:
                    if args.max_clips > 0 and downloaded + len(pending) >= args.max_clips:
                        stop_submitting = True
                        log(f"[TalkVid] Reached max new clips: {downloaded}")
                        break

                    if target_added_bytes > 0 and downloaded_bytes >= target_added_bytes:
                        target_reached_hit = True
                        stop_submitting = True
                        log(
                            f"[TalkVid] Reached target downloaded size this run: "
                            f"{downloaded_bytes / (1024 ** 3):.2f} GB"
                        )
                        break

                    free_bytes = get_free_bytes(output_root)
                    if free_bytes < min_free_bytes:
                        stop_submitting = True
                        log(
                            f"[TalkVid] Free space below threshold: "
                            f"{free_bytes / (1024 ** 3):.2f} GB < {args.min_free_gb:.2f} GB"
                        )
                        break

                    try:
                        item = next(clip_iter)
                    except StopIteration:
                        exhausted = True
                        break

                    item_id = clip_id(item)
                    video_key = video_key_for_item(item)
                    if not clip_allowed(item, args):
                        filtered_out += 1
                        continue

                    if item_id in completed_ids:
                        skipped_existing += 1
                        continue

                    if video_key and video_key in blocked_video_keys:
                        skipped_blocked_source += 1
                        continue

                    out_path = output_path_for(raw_dir, item)
                    if os.path.exists(out_path):
                        skipped_existing += 1
                        continue

                    attempted += 1
                    future = executor.submit(
                        download_clip,
                        item,
                        out_path,
                        args.max_height,
                        args.timeout,
                        args.cookies_file,
                        args.cookies_from_browser,
                    )
                    pending[future] = (attempted, item)

                if not pending:
                    if exhausted or stop_submitting:
                        break
                    continue

                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    attempt_index, item = pending.pop(future)
                    try:
                        ok, reason, size_bytes, detail = future.result()
                    except Exception as exc:
                        ok, reason, size_bytes, detail = False, "unexpected_error", 0, str(exc)
                    log_result(attempt_index, item, ok, reason, size_bytes, detail)

                if bot_check_hit and not bot_check_notice_sent:
                    log("[TalkVid] Bot check detected; stopping new submissions for this run")
                    bot_check_notice_sent = True
                if rate_limited_hit and not rate_limit_notice_sent:
                    log("[TalkVid] Rate limit detected; draining in-flight jobs before cooldown")
                    rate_limit_notice_sent = True
                if rate_limited_hit and not pending:
                    if exhausted:
                        log("[TalkVid] Rate limit hit at the tail of the queue; ending this run for resume")
                        rate_limit_exit = True
                        break
                    if (
                        args.rate_limit_cooldown_seconds <= 0
                        or rate_limit_cooldowns_used >= args.max_rate_limit_cooldowns
                    ):
                        log("[TalkVid] Rate limit retry budget exhausted; ending this run")
                        rate_limit_exit = True
                        break

                    rate_limit_cooldowns_used += 1
                    resume_ts = time.time() + args.rate_limit_cooldown_seconds
                    resume_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(resume_ts))
                    log(
                        "[TalkVid] Cooling down for "
                        f"{args.rate_limit_cooldown_seconds}s before resuming "
                        f"({rate_limit_cooldowns_used}/{args.max_rate_limit_cooldowns}) until {resume_str}"
                    )
                    time.sleep(args.rate_limit_cooldown_seconds)
                    rate_limited_hit = False
                    stop_submitting = False
                    rate_limit_notice_sent = False
    except KeyboardInterrupt:
        interrupted = True
        log("[TalkVid] Interrupted externally; keeping already downloaded files for resume")

    final_raw_bytes = get_dir_size_bytes(raw_dir)
    log(
        "[TalkVid] Done: "
        f"downloaded={downloaded} skipped_existing={skipped_existing} "
        f"skipped_blocked_source={skipped_blocked_source} "
        f"filtered_out={filtered_out} failures={failures} "
        f"rate_limit_events={rate_limit_events} "
        f"rate_limit_cooldowns_used={rate_limit_cooldowns_used}"
    )
    log(f"[TalkVid] downloaded this run: {downloaded_bytes / (1024 ** 3):.2f} GB")
    log(f"[TalkVid] raw size: {final_raw_bytes / (1024 ** 3):.2f} GB")
    log(f"[TalkVid] manifest: {manifest_path}")
    if rate_limit_exit:
        log("[TalkVid] exiting with rc=21 because the session is rate-limited")
        return 21
    if bot_check_hit:
        log("[TalkVid] exiting with rc=20 because a bot_check was detected")
        return 20
    if target_reached_hit:
        log("[TalkVid] exiting with rc=10 because the target size for this run was reached")
        return 10
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
