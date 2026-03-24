#!/usr/bin/env python3
"""
Compare multiple SyncNet teachers on holdout processed videos.

Holdout is defined as:
    processed speaker dirs - speaker snapshot used for local SyncNet training

Metrics are computed on the same randomly sampled windows for every teacher:
  1. positive vs shifted-audio negative within the same clip
  2. positive vs foreign-audio negative from a different holdout clip

The script reports threshold-free ranking quality:
  - pairwise accuracy: fraction of times pos_score > neg_score
  - mean margin: mean(pos_score - neg_score)
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)


def load_snapshot(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_meta(speaker_dir):
    meta_path = os.path.join(speaker_dir, "bbox.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def build_mel_chunks(mel, fps, mel_step_size=16):
    mel_idx_mult = 80.0 / float(fps)
    chunks = []
    n_frames = max(1, int(np.ceil(mel.shape[1] / mel_idx_mult)))
    for i in range(n_frames):
        start = int(i * mel_idx_mult)
        end = start + mel_step_size
        if mel.shape[1] < mel_step_size:
            chunk = np.pad(
                mel,
                ((0, 0), (0, mel_step_size - mel.shape[1])),
                mode="edge",
            )
        elif end > mel.shape[1]:
            chunk = mel[:, -mel_step_size:]
        else:
            chunk = mel[:, start:end]
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


class HoldoutSet:
    def __init__(self, processed_roots, snapshot_path, fps=25, cache_size=16, speaker_allowlist=None):
        self.processed_roots = processed_roots
        self.snapshot = load_snapshot(snapshot_path)
        self.speaker_allowlist = speaker_allowlist
        self.fps = fps
        self.cache = OrderedDict()
        self.cache_size = max(int(cache_size), 0)
        self.items = self._collect_items()

    def _collect_items(self):
        items = []
        seen = {}
        duplicates = []
        for processed_root in self.processed_roots:
            for name in sorted(os.listdir(processed_root)):
                speaker_dir = os.path.join(processed_root, name)
                if not os.path.isdir(speaker_dir):
                    continue
                if self.speaker_allowlist is not None and name not in self.speaker_allowlist:
                    continue
                if name in self.snapshot:
                    continue
                meta = load_meta(speaker_dir)
                if meta.get("bad_sample", False):
                    continue
                frames_path = os.path.join(speaker_dir, "frames.npy")
                mel_path = os.path.join(speaker_dir, "mel.npy")
                if not (os.path.exists(frames_path) and os.path.exists(mel_path)):
                    continue
                if name in seen:
                    duplicates.append((name, seen[name], processed_root))
                    continue
                seen[name] = processed_root
                frames = np.load(frames_path, mmap_mode="r")
                mel = np.load(mel_path, mmap_mode="r")
                mel_chunks = build_mel_chunks(mel, self.fps)
                n_frames = min(int(frames.shape[0]), len(mel_chunks))
                if n_frames < 12:
                    continue
                items.append({
                    "name": name,
                    "dir": speaker_dir,
                    "processed_root": processed_root,
                    "n_frames": n_frames,
                })
        if duplicates:
            detail = "\n".join(
                f"  - {name}: {root_a} vs {root_b}"
                for name, root_a, root_b in duplicates[:20]
            )
            raise RuntimeError(
                "Duplicate speaker names across processed roots would make holdout comparison ambiguous:\n"
                f"{detail}"
            )
        return items

    def _remember(self, key, value):
        self.cache[key] = value
        while self.cache_size and len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def load_item(self, item):
        key = item["dir"]
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        frames = np.load(os.path.join(item["dir"], "frames.npy"), mmap_mode="r")
        mel = np.load(os.path.join(item["dir"], "mel.npy"))
        mel_chunks = build_mel_chunks(mel, self.fps)
        n_frames = min(int(frames.shape[0]), len(mel_chunks))
        value = (frames, mel_chunks, n_frames)
        self._remember(key, value)
        return value


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


def load_teacher(spec, device, T):
    if spec["kind"] == "official":
        SyncNet = load_official_syncnet_class()
        model = SyncNet().to(device)
        ck = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
        state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state_dict)
    else:
        SyncNet = load_local_syncnet_class()
        model = SyncNet(T=T).to(device)
        ck = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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


def build_sample_pairs(holdout, T, samples, seed):
    rng = random.Random(seed)
    records = []
    if len(holdout.items) < 2:
        raise RuntimeError("Need at least 2 valid holdout speakers for comparison")

    for sample_idx in range(samples):
        item = rng.choice(holdout.items)
        frames, mel_chunks, n_frames = holdout.load_item(item)
        start = rng.randint(0, n_frames - T - 1)

        offset = rng.choice([-1, 1]) * rng.randint(5, max(5, n_frames // 2))
        shifted_start = (start + offset) % max(1, n_frames - T)

        other = rng.choice([x for x in holdout.items if x["name"] != item["name"]])
        _, other_mel_chunks, other_n_frames = holdout.load_item(other)
        other_start = rng.randint(0, other_n_frames - T - 1)

        records.append({
            "sample_idx": sample_idx,
            "speaker": item["name"],
            "start": start,
            "shifted_start": shifted_start,
            "other_speaker": other["name"],
            "other_start": other_start,
        })
    return records


def evaluate_teachers(args):
    device = torch.device(args.device)
    holdout = HoldoutSet(
        processed_roots=args.processed_root,
        snapshot_path=args.speaker_snapshot,
        fps=args.fps,
        cache_size=args.cache_size,
        speaker_allowlist=load_allowlist(args.speaker_list),
    )
    if not holdout.items:
        raise RuntimeError("No valid holdout speakers found after excluding the snapshot and bad samples")

    sample_records = build_sample_pairs(holdout, args.T, args.samples, args.seed)

    teacher_specs = [{"name": "official_syncnet", "kind": "official", "checkpoint": args.official_checkpoint}]
    for ckpt in args.checkpoints:
        teacher_specs.append({
            "name": os.path.splitext(os.path.basename(ckpt))[0],
            "kind": "local",
            "checkpoint": ckpt,
        })

    results = {
        "processed_roots": args.processed_root,
        "holdout_total": len(holdout.items),
        "samples": len(sample_records),
        "teachers": {},
        "sample_records": sample_records,
    }

    for spec in teacher_specs:
        model = load_teacher(spec, device, args.T)
        shifted_margins = []
        shifted_hits = 0
        foreign_margins = []
        foreign_hits = 0

        per_sample = []
        for record in sample_records:
            item = next(x for x in holdout.items if x["name"] == record["speaker"])
            other = next(x for x in holdout.items if x["name"] == record["other_speaker"])
            frames, mel_chunks, _ = holdout.load_item(item)
            _, other_mel_chunks, _ = holdout.load_item(other)

            visual = build_visual_window(frames, record["start"], args.T, img_size=args.img_size)
            if spec["kind"] == "official":
                pos_audio = build_official_audio(mel_chunks, record["start"])
                shifted_audio = build_official_audio(mel_chunks, record["shifted_start"])
                foreign_audio = build_official_audio(other_mel_chunks, record["other_start"])
            else:
                pos_audio = build_local_audio(mel_chunks, record["start"], args.T)
                shifted_audio = build_local_audio(mel_chunks, record["shifted_start"], args.T)
                foreign_audio = build_local_audio(other_mel_chunks, record["other_start"], args.T)

            pos_score, shifted_score = score_teacher(spec, model, visual, pos_audio, shifted_audio, device)
            _, foreign_score = score_teacher(spec, model, visual, pos_audio, foreign_audio, device)

            shifted_margin = pos_score - shifted_score
            foreign_margin = pos_score - foreign_score
            shifted_margins.append(shifted_margin)
            foreign_margins.append(foreign_margin)
            shifted_hits += int(shifted_margin > 0)
            foreign_hits += int(foreign_margin > 0)
            per_sample.append({
                "sample_idx": record["sample_idx"],
                "speaker": record["speaker"],
                "other_speaker": record["other_speaker"],
                "pos_score": pos_score,
                "shifted_score": shifted_score,
                "foreign_score": foreign_score,
                "shifted_margin": shifted_margin,
                "foreign_margin": foreign_margin,
            })

        results["teachers"][spec["name"]] = {
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
    parser.add_argument("--speaker-list", default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--cache-size", type=int, default=16)
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
            f"shift_acc={metrics['shifted_pairwise_acc']:.3f}, "
            f"shift_margin={metrics['shifted_margin_mean']:.4f}, "
            f"foreign_acc={metrics['foreign_pairwise_acc']:.3f}, "
            f"foreign_margin={metrics['foreign_margin_mean']:.4f}, "
            f"acc_mean={metrics['pairwise_acc_mean']:.3f}, "
            f"margin_mean={metrics['margin_mean']:.4f}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
