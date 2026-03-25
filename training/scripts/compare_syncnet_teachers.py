#!/usr/bin/env python3
"""
Compare multiple SyncNet teachers on holdout videos.

Supports:
  - materialized processed roots with frames.npy + mel.npy
  - lazy faceclip roots with mp4 + json, materialized on demand

Each local teacher is evaluated with its own checkpoint config, so changes like
`hop_size=160` and `mel_steps=20` are respected during comparison.
"""

import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
sys.path.insert(0, TRAINING_ROOT)
from data import LipSyncDataset


DEFAULT_OFFICIAL_AUDIO_CFG = {
    "sample_rate": 16000,
    "n_fft": 800,
    "hop_size": 200,
    "win_size": 800,
    "n_mels": 80,
    "fmin": 55,
    "fmax": 7600,
    "preemphasis": 0.97,
}


def load_snapshot(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_official_syncnet_class():
    official_models_dir = os.path.join(REPO_ROOT, "models", "official_syncnet", "models")

    def load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    conv_mod = load_module("official_syncnet_models.conv", os.path.join(official_models_dir, "conv.py"))
    with open(os.path.join(official_models_dir, "syncnet.py")) as f:
        syncnet_src = f.read()
    syncnet_src = syncnet_src.replace("from .conv import Conv2d", "")
    syncnet_ns = {"__builtins__": __builtins__}
    exec("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n", syncnet_ns)
    syncnet_ns["Conv2d"] = conv_mod.Conv2d
    exec(syncnet_src, syncnet_ns)
    return syncnet_ns["SyncNet_color"]


def load_local_syncnet_class():
    sys.path.insert(0, os.path.join(TRAINING_ROOT, "models"))
    from syncnet import SyncNet

    return SyncNet


def load_teacher_model(spec, device):
    if spec["kind"] == "official":
        SyncNet = load_official_syncnet_class()
        model = SyncNet().to(device)
        ck = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
        state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state_dict)
    else:
        SyncNet = load_local_syncnet_class()
        model = SyncNet(T=spec["T"]).to(device)
        ck = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_local_teacher_spec(checkpoint_path, default_T):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    spec = {
        "name": os.path.splitext(os.path.basename(checkpoint_path))[0],
        "kind": "local",
        "checkpoint": checkpoint_path,
        "audio_cfg": cfg.get("audio", DEFAULT_OFFICIAL_AUDIO_CFG),
        "mel_steps": int(cfg.get("model", {}).get("mel_steps", 16)),
        "img_size": int(cfg.get("model", {}).get("img_size", 96)),
        "T": int(cfg.get("syncnet", {}).get("T", default_T)),
    }
    return spec


def build_dataset_for_spec(spec, args, speaker_allowlist):
    return LipSyncDataset(
        roots=args.processed_root,
        img_size=spec["img_size"],
        mel_step_size=spec["mel_steps"],
        fps=args.fps,
        audio_cfg=spec["audio_cfg"],
        syncnet_T=spec["T"],
        mode="syncnet",
        cache_size=args.cache_size,
        skip_bad_samples=True,
        speaker_allowlist=speaker_allowlist,
        lazy_cache_root=args.lazy_cache_root,
        ffmpeg_bin=args.ffmpeg_bin,
        materialize_timeout=args.materialize_timeout,
        materialize_frames_size=args.materialize_frames_size or spec["img_size"],
        min_samples_per_speaker=0,
    )


def build_holdout_items(dataset, snapshot_names):
    items = []
    for key in dataset.speakers:
        entry = dataset._entries[key]
        if entry["name"] in snapshot_names:
            continue
        frame_count = dataset._entry_frame_count(entry)
        if frame_count < dataset.syncnet_T + 5:
            continue
        items.append(
            {
                "name": entry["name"],
                "key": key,
                "n_frames": frame_count,
            }
        )
    return items


def resize_face(frame, size=96):
    import cv2

    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size))
    return frame.astype(np.float32) / 255.0


def build_visual_window(frames, start, T, img_size=96):
    window = [resize_face(frames[start + t], img_size) for t in range(T)]
    lowers = [face[img_size // 2 :, :, :] for face in window]
    visual = np.concatenate([lower.transpose(2, 0, 1) for lower in lowers], axis=0)
    return visual.astype(np.float32, copy=False)


def build_official_audio(mel_chunks, start):
    return mel_chunks[start][np.newaxis, np.newaxis]


def build_local_audio(mel_chunks, start, T):
    mel_concat = np.concatenate([mel_chunks[start + t] for t in range(T)], axis=1)
    return mel_concat[np.newaxis, np.newaxis]


def score_teacher(spec, model, visual_np, pos_audio_np, neg_audio_np, device):
    visual = torch.from_numpy(visual_np).unsqueeze(0).to(device)
    pos_audio = torch.from_numpy(pos_audio_np).to(device)
    neg_audio = torch.from_numpy(neg_audio_np).to(device)

    with torch.inference_mode():
        if spec["kind"] == "official":
            pos_a, pos_v = model(pos_audio, visual)
            neg_a, neg_v = model(neg_audio, visual)
        else:
            pos_v, pos_a = model(visual, pos_audio)
            neg_v, neg_a = model(visual, neg_audio)
        pos_score = F.cosine_similarity(pos_a, pos_v).mean().item()
        neg_score = F.cosine_similarity(neg_a, neg_v).mean().item()
    return pos_score, neg_score


def build_sample_pairs(items, T, samples, seed):
    rng = random.Random(seed)
    records = []
    if len(items) < 2:
        raise RuntimeError("Need at least 2 valid holdout speakers for comparison")

    name_to_item = {item["name"]: item for item in items}

    for sample_idx in range(samples):
        item = rng.choice(items)
        n_frames = int(item["n_frames"])
        start = rng.randint(0, n_frames - T - 1)

        offset = rng.choice([-1, 1]) * rng.randint(5, max(5, n_frames // 2))
        shifted_start = (start + offset) % max(1, n_frames - T)

        other = rng.choice([x for x in items if x["name"] != item["name"]])
        other_n_frames = int(other["n_frames"])
        other_start = rng.randint(0, other_n_frames - T - 1)

        records.append(
            {
                "sample_idx": sample_idx,
                "speaker": item["name"],
                "start": start,
                "shifted_start": shifted_start,
                "other_speaker": other["name"],
                "other_start": other_start,
            }
        )

    return records, name_to_item


def evaluate_teachers(args):
    device = torch.device(args.device)
    snapshot_names = load_snapshot(args.speaker_snapshot)
    requested_holdout = load_allowlist(args.speaker_list)

    official_spec = {
        "name": "official_wav2lip",
        "kind": "official",
        "checkpoint": args.official_checkpoint,
        "audio_cfg": {
            "sample_rate": args.official_sample_rate,
            "n_fft": args.official_n_fft,
            "hop_size": args.official_hop_size,
            "win_size": args.official_win_size,
            "n_mels": args.official_n_mels,
            "fmin": args.official_fmin,
            "fmax": args.official_fmax,
            "preemphasis": args.official_preemphasis,
        },
        "mel_steps": int(args.official_mel_steps),
        "img_size": int(args.official_img_size),
        "T": int(args.T),
    }

    local_specs = [load_local_teacher_spec(path, default_T=args.T) for path in args.checkpoints]
    teacher_specs = [official_spec] + local_specs

    if any(spec["T"] != args.T for spec in teacher_specs):
        bad = {spec["name"]: spec["T"] for spec in teacher_specs if spec["T"] != args.T}
        raise RuntimeError(
            f"All teacher checkpoints must match --T={args.T} for a fair compare, got {bad}"
        )

    base_dataset = build_dataset_for_spec(official_spec, args, requested_holdout)
    holdout_items = build_holdout_items(base_dataset, snapshot_names)
    if requested_holdout is not None:
        requested_missing = sorted(requested_holdout - {item["name"] for item in holdout_items})
        if requested_missing:
            print(f"missing_requested_holdout={requested_missing[:20]}")
    if not holdout_items:
        raise RuntimeError("No valid holdout speakers found after excluding the training snapshot")

    sample_records, item_by_name = build_sample_pairs(holdout_items, args.T, args.samples, args.seed)
    holdout_names = sorted(item["name"] for item in holdout_items)

    datasets = {
        spec["name"]: build_dataset_for_spec(spec, args, holdout_names)
        for spec in teacher_specs
    }

    results = {
        "processed_roots": args.processed_root,
        "holdout_total": len(holdout_items),
        "samples": len(sample_records),
        "teachers": {},
        "sample_records": sample_records,
        "holdout_names": holdout_names,
    }

    for spec in teacher_specs:
        model = load_teacher_model(spec, device)
        dataset = datasets[spec["name"]]
        shifted_margins = []
        shifted_hits = 0
        foreign_margins = []
        foreign_hits = 0
        per_sample = []

        for record in sample_records:
            speaker_key = dataset._entry_name_to_key[record["speaker"]]
            other_key = dataset._entry_name_to_key[record["other_speaker"]]
            frames, mel_chunks, _ = dataset._load_speaker(speaker_key)
            _, other_mel_chunks, _ = dataset._load_speaker(other_key)

            visual = build_visual_window(frames, record["start"], spec["T"], img_size=spec["img_size"])
            if spec["kind"] == "official":
                pos_audio = build_official_audio(mel_chunks, record["start"])
                shifted_audio = build_official_audio(mel_chunks, record["shifted_start"])
                foreign_audio = build_official_audio(other_mel_chunks, record["other_start"])
            else:
                pos_audio = build_local_audio(mel_chunks, record["start"], spec["T"])
                shifted_audio = build_local_audio(mel_chunks, record["shifted_start"], spec["T"])
                foreign_audio = build_local_audio(other_mel_chunks, record["other_start"], spec["T"])

            pos_score, shifted_score = score_teacher(spec, model, visual, pos_audio, shifted_audio, device)
            _, foreign_score = score_teacher(spec, model, visual, pos_audio, foreign_audio, device)

            shifted_margin = pos_score - shifted_score
            foreign_margin = pos_score - foreign_score
            shifted_margins.append(shifted_margin)
            foreign_margins.append(foreign_margin)
            shifted_hits += int(shifted_margin > 0)
            foreign_hits += int(foreign_margin > 0)
            per_sample.append(
                {
                    "sample_idx": record["sample_idx"],
                    "speaker": record["speaker"],
                    "other_speaker": record["other_speaker"],
                    "pos_score": pos_score,
                    "shifted_score": shifted_score,
                    "foreign_score": foreign_score,
                    "shifted_margin": shifted_margin,
                    "foreign_margin": foreign_margin,
                }
            )

        results["teachers"][spec["name"]] = {
            "kind": spec["kind"],
            "checkpoint": spec["checkpoint"],
            "audio_cfg": spec["audio_cfg"],
            "mel_steps": spec["mel_steps"],
            "img_size": spec["img_size"],
            "T": spec["T"],
            "shifted_pairwise_acc": shifted_hits / max(1, len(sample_records)),
            "shifted_margin_mean": float(np.mean(shifted_margins)),
            "foreign_pairwise_acc": foreign_hits / max(1, len(sample_records)),
            "foreign_margin_mean": float(np.mean(foreign_margins)),
            "pairwise_acc_mean": float((shifted_hits + foreign_hits) / max(1, 2 * len(sample_records))),
            "margin_mean": float((np.mean(shifted_margins) + np.mean(foreign_margins)) / 2.0),
            "per_sample": per_sample,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", action="append", required=True)
    parser.add_argument("--speaker-snapshot", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker-list", default=None, help="Optional unseen holdout allowlist")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=16)
    parser.add_argument("--lazy-cache-root", default=None)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--materialize-timeout", type=int, default=600)
    parser.add_argument("--materialize-frames-size", default=None)
    parser.add_argument("--official-img-size", type=int, default=96)
    parser.add_argument("--img-size", dest="official_img_size", type=int, help="Backward-compatible alias for --official-img-size")
    parser.add_argument("--official-mel-steps", type=int, default=16)
    parser.add_argument("--official-sample-rate", type=int, default=16000)
    parser.add_argument("--official-n-fft", type=int, default=800)
    parser.add_argument("--official-hop-size", type=int, default=200)
    parser.add_argument("--official-win-size", type=int, default=800)
    parser.add_argument("--official-n-mels", type=int, default=80)
    parser.add_argument("--official-fmin", type=float, default=55)
    parser.add_argument("--official-fmax", type=float, default=7600)
    parser.add_argument("--official-preemphasis", type=float, default=0.97)
    args = parser.parse_args()

    results = evaluate_teachers(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"holdout_total={results['holdout_total']}")
    print(f"samples={results['samples']}")
    for name, metrics in results["teachers"].items():
        print(
            f"{name}: "
            f"kind={metrics['kind']}, "
            f"hop={metrics['audio_cfg']['hop_size']}, "
            f"mel_steps={metrics['mel_steps']}, "
            f"audio_width={metrics['mel_steps'] if metrics['kind'] == 'official' else metrics['mel_steps'] * metrics['T']}, "
            f"acc_mean={metrics['pairwise_acc_mean']:.3f}, "
            f"margin_mean={metrics['margin_mean']:.4f}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
