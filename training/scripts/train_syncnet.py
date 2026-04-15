#!/usr/bin/env python3
"""
Stage 1: Train SyncNet — audio-visual synchronization discriminator.

Usage:
    python scripts/train_syncnet.py --config configs/default.yaml
"""

import argparse
from collections import deque
import importlib.util
import os
import random
import sys
import time
from contextlib import nullcontext
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
sys.path.insert(0, os.path.join(TRAINING_ROOT, "models"))
sys.path.insert(0, TRAINING_ROOT)
from syncnet import SyncNet
from syncnet_mirror import SyncNetMirror
from data import LipSyncDataset
from config_loader import load_config
from scripts.dataset_roots import get_dataset_roots


def update_ema(current, value, span):
    alpha = 2.0 / (span + 1.0)
    if current is None:
        return value
    return (alpha * value) + ((1.0 - alpha) * current)


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_remaining_eta(seconds_elapsed, units_done, units_total):
    if seconds_elapsed <= 0 or units_done <= 0:
        return None
    remaining_units = max(0, units_total - units_done)
    return (seconds_elapsed / units_done) * remaining_units


def compute_remaining_eta_from_recent(avg_unit_seconds, remaining_units):
    if avg_unit_seconds is None or avg_unit_seconds <= 0:
        return None
    return avg_unit_seconds * max(0, remaining_units)


def resolve_repo_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def build_syncnet_model(model_type, syncnet_T, device):
    if model_type == "mirror":
        return SyncNetMirror().to(device)
    return SyncNet(T=syncnet_T).to(device)


def syncnet_forward(model, model_type, visual, audio):
    if model_type == "mirror":
        audio_emb, face_emb = model(audio, visual)
        return audio_emb, face_emb
    visual_emb, audio_emb = model(visual, audio)
    return audio_emb, visual_emb


def syncnet_loss(model, model_type, audio_emb, visual_emb, labels):
    if model_type == "mirror":
        return SyncNetMirror.cosine_loss(audio_emb, visual_emb, labels)
    return SyncNet.cosine_loss(visual_emb, audio_emb, labels)


def syncnet_acc(cos_sim, model_type):
    threshold = 0.5 if model_type == "mirror" else 0.0
    return (cos_sim > threshold).float()


def build_syncnet_loader(dataset, cfg, batch_size, device, model_type, shuffle, is_eval=False):
    num_workers = 0 if is_eval else cfg["data"]["num_workers"]
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    if model_type != "mirror":
        loader_kwargs["pin_memory"] = (device == "cuda")
        if not is_eval:
            loader_kwargs["drop_last"] = True
        if device == "cuda":
            loader_kwargs["pin_memory_device"] = "cuda"
        if not is_eval and cfg["data"]["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = cfg["data"].get("persistent_workers", True)
            prefetch_factor = cfg["data"].get("prefetch_factor")
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_sync_alignment_kwargs(cfg):
    sync_cfg = cfg.get("data", {}).get("sync_alignment", {})
    return {
        "sync_alignment_enabled": sync_cfg.get("enabled", True),
        "sync_alignment_compute_if_missing": sync_cfg.get("compute_if_missing", True),
        "sync_alignment_guard_mel_ticks": sync_cfg.get("guard_mel_ticks", 10),
        "sync_alignment_search_mel_ticks": sync_cfg.get("search_mel_ticks", 10),
        "sync_alignment_samples": sync_cfg.get("samples", 0),
        "sync_alignment_sample_density_per_5s": sync_cfg.get("sample_density_per_5s", 10.0),
        "sync_alignment_seed": sync_cfg.get("seed", 20260403),
        "sync_alignment_min_start_gap_ratio": sync_cfg.get("min_start_gap_ratio", 0.0),
        "sync_alignment_start_gap_multiple": sync_cfg.get("start_gap_multiple", 0),
        "sync_alignment_device": sync_cfg.get("device", "auto"),
        "sync_alignment_batch_size": sync_cfg.get("batch_size", 640),
        "sync_alignment_outlier_trim_ratio": sync_cfg.get("outlier_trim_ratio", 0.2),
        "sync_alignment_min_consensus_ratio": sync_cfg.get("min_consensus_ratio"),
        "sync_alignment_max_shift_mad": sync_cfg.get("max_shift_mad"),
        "sync_alignment_syncnet_checkpoint": sync_cfg.get("syncnet_checkpoint"),
        "sync_alignment_write_manifest": sync_cfg.get("write_manifest", True),
    }


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


def load_official_syncnet_model(checkpoint_path, device):
    SyncNet = load_official_syncnet_class()
    model = SyncNet().to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        if "state_dict" in ck:
            state_dict = ck["state_dict"]
        elif "model" in ck:
            state_dict = ck["model"]
        else:
            state_dict = ck
    else:
        state_dict = ck
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_syncnet_init_weights(checkpoint_path, model, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        if "model" in ck:
            state_dict = ck["model"]
            kind = "syncnet_checkpoint"
        elif "state_dict" in ck:
            state_dict = ck["state_dict"]
            kind = "state_dict_checkpoint"
        else:
            state_dict = ck
            kind = "raw_state_dict"
    else:
        raise RuntimeError(f"Unsupported SyncNet init checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)
    return {
        "kind": kind,
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "global_step": ck.get("global_step") if isinstance(ck, dict) else None,
    }


def capture_rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_eval_rng(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def cuda_autocast(enabled):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def cuda_grad_scaler(enabled):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        return amp_mod.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def evaluate_syncnet_model(model, model_type, loader, device, max_batches=None, use_amp=False, seed=None):
    rng_state = capture_rng_state(device) if seed is not None else None
    if seed is not None:
        seed_eval_rng(seed, device)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    non_blocking = device == "cuda"

    with torch.inference_mode():
        for batch_idx, (visual, audio, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            visual = visual.to(device, non_blocking=non_blocking)
            audio = audio.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            amp_ctx = cuda_autocast(enabled=use_amp) if device == "cuda" else nullcontext()
            with amp_ctx:
                audio_emb, visual_emb = syncnet_forward(model, model_type, visual, audio)
                loss = syncnet_loss(model, model_type, audio_emb, visual_emb, labels)

            cos_sim = F.cosine_similarity(audio_emb, visual_emb)
            preds = syncnet_acc(cos_sim, model_type)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            batches += 1

    if was_training:
        model.train()
    if rng_state is not None:
        restore_rng_state(rng_state, device)

    if batches == 0:
        return None
    return {
        "loss": total_loss / batches,
        "acc": total_acc / batches,
        "batches": batches,
    }


def format_official_eval_comparison(metrics, baseline_metrics, label="official"):
    if baseline_metrics is None:
        return ""
    return (
        f" vs_{label} loss_delta={metrics['loss'] - baseline_metrics['loss']:+.4f} "
        f"acc_delta={metrics['acc'] - baseline_metrics['acc']:+.4f}"
    )


def build_pairwise_eval_items(dataset, min_frames):
    items = []
    for speaker_key in dataset.speakers:
        entry = dataset._entries[speaker_key]
        n_frames = dataset._entry_frame_count(entry)
        if n_frames <= (3 * dataset.syncnet_T):
            continue
        if n_frames < min_frames:
            continue
        items.append(
            {
                "name": entry["name"],
                "key": speaker_key,
                "n_frames": n_frames,
            }
        )
    return items


def load_pairwise_eval_item(dataset, cache, item):
    key = item["key"]
    value = cache.get(key)
    if value is not None:
        return value
    frames, mel_chunks, _ = dataset._load_speaker(key)
    n_frames = min(int(frames.shape[0]), len(mel_chunks))
    value = (frames, mel_chunks, n_frames)
    cache[key] = value
    return value


def resize_face(frame, size=96):
    import cv2

    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size))
    return frame.astype(np.float32) / 255.0


def build_pairwise_visual_window(frames, start, T, img_size=96):
    window = [resize_face(frames[start + t], img_size) for t in range(T)]
    lowers = [face[img_size // 2 :, :, :] for face in window]
    visual = np.concatenate([lower.transpose(2, 0, 1) for lower in lowers], axis=0)
    return visual.astype(np.float32, copy=False)


def build_pairwise_audio(mel_chunks, start, T, model_type):
    if model_type == "mirror":
        return mel_chunks[start][np.newaxis, np.newaxis]
    mel_concat = np.concatenate([mel_chunks[start + t] for t in range(T)], axis=1)
    return mel_concat[np.newaxis, np.newaxis]


def score_syncnet_pair(model, model_type, visual_np, pos_audio_np, neg_audio_np, device):
    visual = torch.from_numpy(visual_np).unsqueeze(0).to(device)
    pos_audio = torch.from_numpy(pos_audio_np).to(device)
    neg_audio = torch.from_numpy(neg_audio_np).to(device)

    with torch.inference_mode():
        pos_audio_emb, pos_visual_emb = syncnet_forward(model, model_type, visual, pos_audio)
        neg_audio_emb, neg_visual_emb = syncnet_forward(model, model_type, visual, neg_audio)
        pos_score = F.cosine_similarity(pos_audio_emb, pos_visual_emb).mean().item()
        neg_score = F.cosine_similarity(neg_audio_emb, neg_visual_emb).mean().item()
    return pos_score, neg_score


def evaluate_syncnet_pairwise(
    model,
    model_type,
    dataset,
    pairwise_items,
    device,
    samples,
    seed,
    img_size,
    T,
):
    if not pairwise_items or len(pairwise_items) < 2 or samples <= 0:
        return None

    rng_state = capture_rng_state(device)
    seed_eval_rng(seed, device)
    rng = random.Random(seed)

    was_training = model.training
    model.eval()

    cache = {}
    shifted_margins = []
    foreign_margins = []
    shifted_hits = 0
    foreign_hits = 0

    try:
        for _ in range(samples):
            item = rng.choice(pairwise_items)
            frames, mel_chunks, n_frames = load_pairwise_eval_item(dataset, cache, item)
            start = rng.randint(0, n_frames - T - 1)
            offset = rng.choice([-1, 1]) * rng.randint(5, max(5, n_frames // 2))
            shifted_start = (start + offset) % max(1, n_frames - T)

            other = rng.choice([x for x in pairwise_items if x["name"] != item["name"]])
            _, other_mel_chunks, other_n_frames = load_pairwise_eval_item(dataset, cache, other)
            other_start = rng.randint(0, other_n_frames - T - 1)

            visual = build_pairwise_visual_window(frames, start, T, img_size=img_size)
            pos_audio = build_pairwise_audio(mel_chunks, start, T, model_type)
            shifted_audio = build_pairwise_audio(mel_chunks, shifted_start, T, model_type)
            foreign_audio = build_pairwise_audio(other_mel_chunks, other_start, T, model_type)

            pos_score, shifted_score = score_syncnet_pair(model, model_type, visual, pos_audio, shifted_audio, device)
            _, foreign_score = score_syncnet_pair(model, model_type, visual, pos_audio, foreign_audio, device)

            shifted_margin = pos_score - shifted_score
            foreign_margin = pos_score - foreign_score
            shifted_margins.append(shifted_margin)
            foreign_margins.append(foreign_margin)
            shifted_hits += int(shifted_margin > 0.0)
            foreign_hits += int(foreign_margin > 0.0)
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_state, device)

    if not shifted_margins or not foreign_margins:
        return None
    return {
        "shifted_pairwise_acc": shifted_hits / max(1, samples),
        "shifted_margin_mean": float(np.mean(shifted_margins)),
        "foreign_pairwise_acc": foreign_hits / max(1, samples),
        "foreign_margin_mean": float(np.mean(foreign_margins)),
        "pairwise_acc_mean": float((shifted_hits + foreign_hits) / max(1, 2 * samples)),
        "margin_mean": float((np.mean(shifted_margins) + np.mean(foreign_margins)) / 2.0),
        "samples": samples,
    }


def format_pairwise_eval_comparison(metrics, baseline_metrics, label="official"):
    if baseline_metrics is None:
        return ""
    return (
        f" vs_{label} acc_mean_delta={metrics['pairwise_acc_mean'] - baseline_metrics['pairwise_acc_mean']:+.4f} "
        f"margin_delta={metrics['margin_mean'] - baseline_metrics['margin_mean']:+.4f}"
    )


def format_pairwise_eval_metrics(metrics):
    return (
        f"pairwise_acc={metrics['pairwise_acc_mean']:.3f} "
        f"margin={metrics['margin_mean']:.4f} "
        f"shift_acc={metrics['shifted_pairwise_acc']:.3f} "
        f"foreign_acc={metrics['foreign_pairwise_acc']:.3f}"
    )


def format_pairwise_eval_metrics_flexible(metrics):
    if (
        "shifted_pairwise_acc" in metrics
        and "foreign_pairwise_acc" in metrics
    ):
        return format_pairwise_eval_metrics(metrics)
    return format_pairwise_eval_metrics_compact(metrics)


def checkpoint_first_value(ck, keys):
    for key in keys:
        value = ck.get(key)
        if value is not None:
            return value
    return None


def checkpoint_official_eval_metrics(
    ck,
    *,
    loss_keys=("official_eval_loss", "val_loss"),
    acc_keys=("official_eval_acc", "val_acc"),
    batches_keys=("official_eval_batches",),
):
    loss = checkpoint_first_value(ck, loss_keys)
    if loss is None:
        return None
    metrics = {"loss": float(loss)}
    acc = checkpoint_first_value(ck, acc_keys)
    if acc is not None:
        metrics["acc"] = float(acc)
    batches = checkpoint_first_value(ck, batches_keys)
    if batches is not None:
        metrics["batches"] = int(batches)
    return metrics


def checkpoint_baseline_official_eval_metrics(ck):
    return checkpoint_official_eval_metrics(
        ck,
        loss_keys=("official_baseline_off_eval_loss", "official_val_loss"),
        acc_keys=("official_baseline_off_eval_acc", "official_val_acc"),
        batches_keys=("official_baseline_off_eval_batches",),
    )


def checkpoint_pairwise_eval_metrics(
    ck,
    *,
    acc_key,
    margin_key,
    shifted_key,
    foreign_key,
    samples_key=None,
):
    acc = ck.get(acc_key)
    margin = ck.get(margin_key)
    if acc is None or margin is None:
        return None
    metrics = {
        "pairwise_acc_mean": float(acc),
        "margin_mean": float(margin),
    }
    shifted = ck.get(shifted_key)
    foreign = ck.get(foreign_key)
    if shifted is not None:
        metrics["shifted_pairwise_acc"] = float(shifted)
    if foreign is not None:
        metrics["foreign_pairwise_acc"] = float(foreign)
    if samples_key is not None and ck.get(samples_key) is not None:
        metrics["samples"] = int(ck[samples_key])
    return metrics


def format_pairwise_eval_metrics_compact(metrics):
    return (
        f"pairwise_acc={metrics['pairwise_acc_mean']:.3f} "
        f"margin={metrics['margin_mean']:.4f}"
    )


def save_syncnet_checkpoint(
    path,
    *,
    epoch,
    global_step,
    batches_processed_in_epoch,
    checkpoint_kind,
    model,
    optimizer,
    cfg,
    model_type,
    speaker_list,
    val_speaker_list,
    train_metrics=None,
    official_eval_metrics=None,
    our_eval_metrics=None,
    best_off_eval_loss=None,
    best_off_eval_step=None,
    best_our_eval_acc=None,
    best_our_eval_margin=None,
    best_our_eval_step=None,
    official_baseline_off_eval_metrics=None,
    official_baseline_our_eval_metrics=None,
):
    torch.save(
        {
            "epoch": epoch,
            "global_epoch": epoch,
            "global_step": global_step,
            "batches_processed_in_epoch": batches_processed_in_epoch,
            "checkpoint_kind": checkpoint_kind,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": None if train_metrics is None else train_metrics.get("loss"),
            "acc": None if train_metrics is None else train_metrics.get("acc"),
            "val_loss": None if official_eval_metrics is None else official_eval_metrics.get("loss"),
            "val_acc": None if official_eval_metrics is None else official_eval_metrics.get("acc"),
            "official_eval_loss": None if official_eval_metrics is None else official_eval_metrics.get("loss"),
            "official_eval_acc": None if official_eval_metrics is None else official_eval_metrics.get("acc"),
            "official_eval_batches": None if official_eval_metrics is None else official_eval_metrics.get("batches"),
            "our_eval_pairwise_acc": None if our_eval_metrics is None else our_eval_metrics.get("pairwise_acc_mean"),
            "our_eval_margin_mean": None if our_eval_metrics is None else our_eval_metrics.get("margin_mean"),
            "our_eval_shifted_pairwise_acc": None if our_eval_metrics is None else our_eval_metrics.get("shifted_pairwise_acc"),
            "our_eval_foreign_pairwise_acc": None if our_eval_metrics is None else our_eval_metrics.get("foreign_pairwise_acc"),
            "our_eval_samples": None if our_eval_metrics is None else our_eval_metrics.get("samples"),
            "best_val_loss": best_off_eval_loss,
            "best_val_step": best_off_eval_step,
            "best_off_eval_loss": best_off_eval_loss,
            "best_off_eval_step": best_off_eval_step,
            "best_our_eval_acc": best_our_eval_acc,
            "best_our_eval_margin": best_our_eval_margin,
            "best_our_eval_step": best_our_eval_step,
            "official_val_loss": None if official_baseline_off_eval_metrics is None else official_baseline_off_eval_metrics.get("loss"),
            "official_val_acc": None if official_baseline_off_eval_metrics is None else official_baseline_off_eval_metrics.get("acc"),
            "official_baseline_off_eval_loss": None if official_baseline_off_eval_metrics is None else official_baseline_off_eval_metrics.get("loss"),
            "official_baseline_off_eval_acc": None if official_baseline_off_eval_metrics is None else official_baseline_off_eval_metrics.get("acc"),
            "official_baseline_off_eval_batches": None if official_baseline_off_eval_metrics is None else official_baseline_off_eval_metrics.get("batches"),
            "official_baseline_pairwise_acc": None if official_baseline_our_eval_metrics is None else official_baseline_our_eval_metrics.get("pairwise_acc_mean"),
            "official_baseline_margin_mean": None if official_baseline_our_eval_metrics is None else official_baseline_our_eval_metrics.get("margin_mean"),
            "official_baseline_shifted_pairwise_acc": None if official_baseline_our_eval_metrics is None else official_baseline_our_eval_metrics.get("shifted_pairwise_acc"),
            "official_baseline_foreign_pairwise_acc": None if official_baseline_our_eval_metrics is None else official_baseline_our_eval_metrics.get("foreign_pairwise_acc"),
            "official_baseline_pairwise_samples": None if official_baseline_our_eval_metrics is None else official_baseline_our_eval_metrics.get("samples"),
            "syncnet_kind": model_type,
            "config": cfg,
            "speaker_list": speaker_list,
            "val_speaker_list": val_speaker_list,
        },
        path,
    )


def load_syncnet_checkpoint_model(checkpoint_path, device, default_model_type="mirror", default_syncnet_T=5):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        if "model" in ck:
            state_dict = ck["model"]
        elif "state_dict" in ck:
            state_dict = ck["state_dict"]
        else:
            state_dict = ck
        model_type = ck.get("syncnet_kind", default_model_type)
        ck_cfg = ck.get("config") if isinstance(ck.get("config"), dict) else {}
        syncnet_T = int(ck_cfg.get("syncnet", {}).get("T", default_syncnet_T))
    else:
        state_dict = ck
        model_type = default_model_type
        syncnet_T = default_syncnet_T

    model = build_syncnet_model(model_type, syncnet_T, device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--init-model", default=None, help="Initialize model weights from checkpoint without resuming optimizer/state")
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated list of speaker dirs to include")
    parser.add_argument("--val-speaker-list", default=None, help="Optional newline-separated list of speaker dirs for eval")
    args = parser.parse_args()

    if args.resume and args.init_model:
        raise ValueError("--resume and --init-model are mutually exclusive")

    cfg = load_config(args.config)

    device = cfg["training"]["device"]
    model_type = cfg["syncnet"].get("model_type", "local")
    if model_type not in {"local", "mirror"}:
        raise ValueError(f"Unsupported syncnet.model_type={model_type!r}")
    output_dir = os.path.join(cfg["training"]["output_dir"], "syncnet")
    os.makedirs(output_dir, exist_ok=True)

    if device == "cuda":
        allow_tf32 = cfg["training"].get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cfg["training"].get("cudnn_benchmark", False)

    def log(msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    # Dataset
    roots = get_dataset_roots(cfg)
    speaker_allowlist = None
    if args.speaker_list:
        with open(args.speaker_list) as f:
            speaker_allowlist = [line.strip() for line in f if line.strip()]
        log(f"Loaded speaker snapshot: {len(speaker_allowlist)} entries from {args.speaker_list}")
    val_speaker_allowlist = None
    if args.val_speaker_list:
        with open(args.val_speaker_list) as f:
            val_speaker_allowlist = [line.strip() for line in f if line.strip()]
        log(f"Loaded val snapshot: {len(val_speaker_allowlist)} entries from {args.val_speaker_list}")
    dataset = LipSyncDataset(
        roots=roots,
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="syncnet",
        syncnet_style=model_type,
        cache_size=cfg["data"].get("cache_size", 8),
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        speaker_allowlist=speaker_allowlist,
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
        **build_sync_alignment_kwargs(cfg),
    )
    loader = build_syncnet_loader(
        dataset,
        cfg,
        cfg["syncnet"]["batch_size"],
        device,
        model_type,
        shuffle=True,
        is_eval=False,
    )
    val_loader = None
    if val_speaker_allowlist:
        val_dataset = LipSyncDataset(
            roots=roots,
            img_size=cfg["model"]["img_size"],
            mel_step_size=cfg["model"]["mel_steps"],
            fps=cfg["data"]["fps"],
            audio_cfg=cfg["audio"],
            syncnet_T=cfg["syncnet"]["T"],
            mode="syncnet",
            syncnet_style=model_type,
            cache_size=cfg["data"].get("cache_size", 8),
            skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
            speaker_allowlist=val_speaker_allowlist,
            lazy_cache_root=cfg["data"].get("lazy_cache_root"),
            ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
            materialize_timeout=cfg["data"].get("materialize_timeout", 600),
            materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
            **build_sync_alignment_kwargs(cfg),
        )
        val_loader = build_syncnet_loader(
            val_dataset,
            cfg,
            cfg["syncnet"]["batch_size"],
            device,
            model_type,
            shuffle=False,
            is_eval=True,
        )
        log(f"ValLoader: {len(val_loader)} batches, batch_size={cfg['syncnet']['batch_size']}")

    # Model
    model = build_syncnet_model(model_type, cfg["syncnet"]["T"], device)
    if args.init_model:
        init_info = load_syncnet_init_weights(args.init_model, model, device)
        log(
            "Initialized SyncNet weights from "
            f"{args.init_model} (kind={init_info['kind']}, "
            f"epoch={init_info['epoch']}, global_step={init_info['global_step']})"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["syncnet"]["lr"])
    use_amp = cfg["training"].get("mixed_precision", False) and device == "cuda"
    scaler = cuda_grad_scaler(use_amp) if device == "cuda" else None

    start_epoch = 0
    resume_batches_in_epoch = 0
    global_step = 0
    best_off_eval_loss = float("inf")
    best_off_eval_step = None
    best_our_eval_acc = float("-inf")
    best_our_eval_margin = float("-inf")
    best_our_eval_step = None
    official_baseline_off_eval = None
    official_baseline_our_eval = None
    current_best_baseline_off_eval = None
    current_best_baseline_our_eval = None
    resume_official_eval = None
    resume_our_eval = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        state_key = "model" if "model" in ck else "state_dict"
        model.load_state_dict(ck[state_key])
        optimizer_state = ck.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        checkpoint_kind = ck.get("checkpoint_kind", "epoch")
        if checkpoint_kind == "step":
            start_epoch = int(ck["epoch"])
            resume_batches_in_epoch = int(ck.get("batches_processed_in_epoch", 0))
            log(f"Resumed mid-epoch from epoch {start_epoch} after {resume_batches_in_epoch} batches")
        else:
            start_epoch = int(ck.get("epoch", ck.get("global_epoch", 0))) + 1
            log(f"Resumed from epoch {start_epoch}")
        global_step = int(ck.get("global_step", 0))
        if ck.get("best_off_eval_loss") is not None:
            best_off_eval_loss = float(ck["best_off_eval_loss"])
        elif ck.get("best_val_loss") is not None:
            best_off_eval_loss = float(ck["best_val_loss"])
        if ck.get("best_off_eval_step") is not None:
            best_off_eval_step = int(ck["best_off_eval_step"])
        elif ck.get("best_val_step") is not None:
            best_off_eval_step = int(ck["best_val_step"])
        if ck.get("best_our_eval_acc") is not None:
            best_our_eval_acc = float(ck["best_our_eval_acc"])
        if ck.get("best_our_eval_margin") is not None:
            best_our_eval_margin = float(ck["best_our_eval_margin"])
        if ck.get("best_our_eval_step") is not None:
            best_our_eval_step = int(ck["best_our_eval_step"])
        official_baseline_off_eval = checkpoint_baseline_official_eval_metrics(ck)
        official_baseline_our_eval = checkpoint_pairwise_eval_metrics(
            ck,
            acc_key="official_baseline_pairwise_acc",
            margin_key="official_baseline_margin_mean",
            shifted_key="official_baseline_shifted_pairwise_acc",
            foreign_key="official_baseline_foreign_pairwise_acc",
            samples_key="official_baseline_pairwise_samples",
        )
        resume_official_eval = checkpoint_official_eval_metrics(ck)
        resume_our_eval = checkpoint_pairwise_eval_metrics(
            ck,
            acc_key="our_eval_pairwise_acc",
            margin_key="our_eval_margin_mean",
            shifted_key="our_eval_shifted_pairwise_acc",
            foreign_key="our_eval_foreign_pairwise_acc",
            samples_key="our_eval_samples",
        )

    log(f"SyncNet dataset roots: {roots}")
    log(f"SyncNet training: {len(dataset)} samples, {len(loader)} batches/epoch")
    log(f"Device: {device}, T={cfg['syncnet']['T']}, AMP={use_amp}, model_type={model_type}")
    if device == "cuda":
        log(
            "CUDA backend: "
            f"tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
        )
    log(f"Grad clip: {cfg['training'].get('gradient_clip', 0.0)}")

    eval_interval_steps = int(cfg["syncnet"].get("eval_interval_steps", 0) or 0)
    eval_batches = int(cfg["syncnet"].get("eval_batches", 1400) or 0)
    eval_seed = int(cfg["syncnet"].get("eval_seed", 20260329))
    pairwise_eval_samples = int(cfg["syncnet"].get("pairwise_eval_samples", 200) or 0)
    pairwise_eval_seed = int(cfg["syncnet"].get("pairwise_eval_seed", eval_seed))
    latest_ckpt_path = os.path.join(output_dir, "syncnet_latest.pth")
    best_off_eval_ckpt_path = os.path.join(output_dir, "syncnet_best_off_eval.pth")
    best_our_eval_ckpt_path = os.path.join(output_dir, "syncnet_best_our_eval.pth")
    legacy_best_ckpt_path = os.path.join(output_dir, "syncnet_best.pth")
    pairwise_eval_items = build_pairwise_eval_items(val_dataset, min_frames=12) if val_loader is not None else None
    official_checkpoint = resolve_repo_path(cfg["syncnet"].get("official_checkpoint"))
    current_best_checkpoint = resolve_repo_path(cfg["syncnet"].get("current_best_checkpoint"))
    current_best_label = str(cfg["syncnet"].get("current_best_label", "current_best")).strip() or "current_best"
    if eval_interval_steps > 0:
        log(
            f"Eval cadence: every {eval_interval_steps} global steps, "
            f"eval_batches={eval_batches}, eval_seed={eval_seed}"
        )
        log(f"Latest checkpoint path: {latest_ckpt_path}")
        log(f"Best off-eval checkpoint path: {best_off_eval_ckpt_path}")
        log(f"Best our-eval checkpoint path: {best_our_eval_ckpt_path}")
        if pairwise_eval_samples > 0:
            log(
                f"Our eval cadence: pairwise samples={pairwise_eval_samples}, "
                f"seed={pairwise_eval_seed}"
            )
        if current_best_checkpoint:
            log(f"Current-best baseline checkpoint: {current_best_checkpoint}")
    if eval_interval_steps > 0 and val_loader is None:
        log("WARNING: eval is enabled but no --val-speaker-list was provided; periodic eval/latest/best will stay disabled.")
    baseline_metrics_updated = False
    if eval_interval_steps > 0 and model_type == "mirror" and val_loader is not None:
        reused_baseline_off_eval = official_baseline_off_eval is not None
        reused_baseline_our_eval = official_baseline_our_eval is not None
        need_official_model = (
            official_baseline_off_eval is None
            or (
                pairwise_eval_samples > 0
                and pairwise_eval_items
                and official_baseline_our_eval is None
            )
        )
        official_model = None
        if need_official_model:
            if official_checkpoint and os.path.exists(official_checkpoint):
                official_model = load_official_syncnet_model(official_checkpoint, device)
            else:
                log(f"WARNING: official SyncNet checkpoint not found for baseline eval: {official_checkpoint}")

        if official_baseline_off_eval is None and official_model is not None:
            official_baseline_off_eval = evaluate_syncnet_model(
                official_model,
                "mirror",
                val_loader,
                device,
                max_batches=eval_batches,
                use_amp=False,
                seed=eval_seed,
            )
            baseline_metrics_updated = baseline_metrics_updated or (official_baseline_off_eval is not None)

        if official_baseline_off_eval is not None:
            batches_part = (
                f" batches={official_baseline_off_eval['batches']}"
                if official_baseline_off_eval.get("batches") is not None else ""
            )
            source = "reused from checkpoint" if reused_baseline_off_eval else "baseline"
            log(
                f"Official eval {source}: loss={official_baseline_off_eval['loss']:.4f} "
                f"acc={official_baseline_off_eval['acc']:.3f}{batches_part}"
            )

        if (
            pairwise_eval_samples > 0
            and pairwise_eval_items
            and official_baseline_our_eval is None
            and official_model is not None
        ):
            official_baseline_our_eval = evaluate_syncnet_pairwise(
                official_model,
                "mirror",
                val_dataset,
                pairwise_eval_items,
                device,
                samples=pairwise_eval_samples,
                seed=pairwise_eval_seed,
                img_size=cfg["model"]["img_size"],
                T=cfg["syncnet"]["T"],
            )
            baseline_metrics_updated = baseline_metrics_updated or (official_baseline_our_eval is not None)

        if official_baseline_our_eval is not None:
            samples_part = (
                f" samples={official_baseline_our_eval['samples']}"
                if official_baseline_our_eval.get("samples") is not None else ""
            )
            source = "reused from checkpoint" if reused_baseline_our_eval else "baseline"
            log(
                "Official our-eval "
                f"{source}: {format_pairwise_eval_metrics_flexible(official_baseline_our_eval)}"
                f"{samples_part}"
            )

        if current_best_checkpoint:
            if os.path.exists(current_best_checkpoint):
                current_best_model, current_best_model_type = load_syncnet_checkpoint_model(
                    current_best_checkpoint,
                    device,
                    default_model_type="mirror",
                    default_syncnet_T=cfg["syncnet"]["T"],
                )
                current_best_baseline_off_eval = evaluate_syncnet_model(
                    current_best_model,
                    current_best_model_type,
                    val_loader,
                    device,
                    max_batches=eval_batches,
                    use_amp=False,
                    seed=eval_seed,
                )
                if pairwise_eval_samples > 0 and pairwise_eval_items:
                    current_best_baseline_our_eval = evaluate_syncnet_pairwise(
                        current_best_model,
                        current_best_model_type,
                        val_dataset,
                        pairwise_eval_items,
                        device,
                        samples=pairwise_eval_samples,
                        seed=pairwise_eval_seed,
                        img_size=cfg["model"]["img_size"],
                        T=cfg["syncnet"]["T"],
                    )
            else:
                log(f"WARNING: current-best baseline checkpoint not found for eval: {current_best_checkpoint}")

        if current_best_baseline_off_eval is not None:
            batches_part = (
                f" batches={current_best_baseline_off_eval['batches']}"
                if current_best_baseline_off_eval.get("batches") is not None else ""
            )
            log(
                f"{current_best_label} eval baseline: loss={current_best_baseline_off_eval['loss']:.4f} "
                f"acc={current_best_baseline_off_eval['acc']:.3f}{batches_part}"
            )
        if current_best_baseline_our_eval is not None:
            samples_part = (
                f" samples={current_best_baseline_our_eval['samples']}"
                if current_best_baseline_our_eval.get("samples") is not None else ""
            )
            log(
                f"{current_best_label} our-eval baseline: "
                f"{format_pairwise_eval_metrics_flexible(current_best_baseline_our_eval)}"
                f"{samples_part}"
            )

    if eval_interval_steps > 0 and val_loader is not None:
        reused_initial_off_eval = resume_official_eval is not None
        reused_initial_our_eval = resume_our_eval is not None
        initial_off_eval = resume_official_eval
        if initial_off_eval is None:
            initial_off_eval = evaluate_syncnet_model(
                model,
                model_type,
                val_loader,
                device,
                max_batches=eval_batches,
                use_amp=use_amp,
                seed=eval_seed,
            )
        initial_our_eval = resume_our_eval
        if initial_our_eval is None and pairwise_eval_samples > 0 and pairwise_eval_items:
            initial_our_eval = evaluate_syncnet_pairwise(
                model,
                model_type,
                val_dataset,
                pairwise_eval_items,
                device,
                samples=pairwise_eval_samples,
                seed=pairwise_eval_seed,
                img_size=cfg["model"]["img_size"],
                T=cfg["syncnet"]["T"],
            )
        if initial_off_eval is not None:
            initial_is_best_off = initial_off_eval["loss"] < best_off_eval_loss
            current_best_off_eval_loss = initial_off_eval["loss"] if initial_is_best_off else best_off_eval_loss
            current_best_off_eval_step = global_step if initial_is_best_off else best_off_eval_step
            is_best_our = False
            current_best_our_eval_acc = best_our_eval_acc
            current_best_our_eval_margin = best_our_eval_margin
            current_best_our_eval_step = best_our_eval_step
            if initial_our_eval is not None:
                is_best_our = (
                    initial_our_eval["pairwise_acc_mean"] > best_our_eval_acc
                    or (
                        initial_our_eval["pairwise_acc_mean"] == best_our_eval_acc
                        and initial_our_eval["margin_mean"] > best_our_eval_margin
                    )
                )
                if is_best_our:
                    current_best_our_eval_acc = initial_our_eval["pairwise_acc_mean"]
                    current_best_our_eval_margin = initial_our_eval["margin_mean"]
                    current_best_our_eval_step = global_step
            initial_eval_updated_checkpoint = (
                baseline_metrics_updated
                or not reused_initial_off_eval
                or (initial_our_eval is not None and not reused_initial_our_eval)
                or initial_is_best_off
                or is_best_our
            )
            off_eval_label = "Initial official-eval reused from checkpoint" if reused_initial_off_eval else "Initial official-eval"
            log(
                f"{off_eval_label}: loss={initial_off_eval['loss']:.4f} "
                f"acc={initial_off_eval['acc']:.3f}"
                f"{format_official_eval_comparison(initial_off_eval, official_baseline_off_eval, 'official')}"
                f"{format_official_eval_comparison(initial_off_eval, current_best_baseline_off_eval, current_best_label)}"
            )
            if initial_our_eval is not None:
                our_eval_label = "Initial our-eval reused from checkpoint" if reused_initial_our_eval else "Initial our-eval"
                log(
                    f"{our_eval_label}: "
                    f"{format_pairwise_eval_metrics_flexible(initial_our_eval)}"
                    f"{format_pairwise_eval_comparison(initial_our_eval, official_baseline_our_eval, 'official')}"
                    f"{format_pairwise_eval_comparison(initial_our_eval, current_best_baseline_our_eval, current_best_label)}"
                )
            if initial_eval_updated_checkpoint:
                save_syncnet_checkpoint(
                    latest_ckpt_path,
                    epoch=start_epoch,
                    global_step=global_step,
                    batches_processed_in_epoch=resume_batches_in_epoch,
                    checkpoint_kind="step",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    model_type=model_type,
                    speaker_list=args.speaker_list,
                    val_speaker_list=args.val_speaker_list,
                    official_eval_metrics=initial_off_eval,
                    our_eval_metrics=initial_our_eval,
                    best_off_eval_loss=current_best_off_eval_loss,
                    best_off_eval_step=current_best_off_eval_step,
                    best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                    best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                    best_our_eval_step=current_best_our_eval_step,
                    official_baseline_off_eval_metrics=official_baseline_off_eval,
                    official_baseline_our_eval_metrics=official_baseline_our_eval,
                )
                log(f"Saved latest {latest_ckpt_path}")
            if initial_is_best_off and initial_eval_updated_checkpoint:
                best_off_eval_loss = initial_off_eval["loss"]
                best_off_eval_step = global_step
                save_syncnet_checkpoint(
                    best_off_eval_ckpt_path,
                    epoch=start_epoch,
                    global_step=global_step,
                    batches_processed_in_epoch=resume_batches_in_epoch,
                    checkpoint_kind="step",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    model_type=model_type,
                    speaker_list=args.speaker_list,
                    val_speaker_list=args.val_speaker_list,
                    official_eval_metrics=initial_off_eval,
                    our_eval_metrics=initial_our_eval,
                    best_off_eval_loss=best_off_eval_loss,
                    best_off_eval_step=best_off_eval_step,
                    best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                    best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                    best_our_eval_step=current_best_our_eval_step,
                    official_baseline_off_eval_metrics=official_baseline_off_eval,
                    official_baseline_our_eval_metrics=official_baseline_our_eval,
                )
                save_syncnet_checkpoint(
                    legacy_best_ckpt_path,
                    epoch=start_epoch,
                    global_step=global_step,
                    batches_processed_in_epoch=resume_batches_in_epoch,
                    checkpoint_kind="step",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    model_type=model_type,
                    speaker_list=args.speaker_list,
                    val_speaker_list=args.val_speaker_list,
                    official_eval_metrics=initial_off_eval,
                    our_eval_metrics=initial_our_eval,
                    best_off_eval_loss=best_off_eval_loss,
                    best_off_eval_step=best_off_eval_step,
                    best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                    best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                    best_our_eval_step=current_best_our_eval_step,
                    official_baseline_off_eval_metrics=official_baseline_off_eval,
                    official_baseline_our_eval_metrics=official_baseline_our_eval,
                )
                log(f"Initial official-eval is current best: loss={best_off_eval_loss:.4f} -> {best_off_eval_ckpt_path}")
            if is_best_our and initial_eval_updated_checkpoint:
                best_our_eval_acc = initial_our_eval["pairwise_acc_mean"]
                best_our_eval_margin = initial_our_eval["margin_mean"]
                best_our_eval_step = global_step
                save_syncnet_checkpoint(
                    best_our_eval_ckpt_path,
                    epoch=start_epoch,
                    global_step=global_step,
                    batches_processed_in_epoch=resume_batches_in_epoch,
                    checkpoint_kind="step",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    model_type=model_type,
                    speaker_list=args.speaker_list,
                    val_speaker_list=args.val_speaker_list,
                    official_eval_metrics=initial_off_eval,
                    our_eval_metrics=initial_our_eval,
                    best_off_eval_loss=best_off_eval_loss if best_off_eval_loss < float("inf") else None,
                    best_off_eval_step=best_off_eval_step,
                    best_our_eval_acc=best_our_eval_acc,
                    best_our_eval_margin=best_our_eval_margin,
                    best_our_eval_step=best_our_eval_step,
                    official_baseline_off_eval_metrics=official_baseline_off_eval,
                    official_baseline_our_eval_metrics=official_baseline_our_eval,
                )
                log(
                    f"Initial our-eval is current best: "
                    f"pairwise_acc={best_our_eval_acc:.3f} margin={best_our_eval_margin:.4f} "
                    f"-> {best_our_eval_ckpt_path}"
                )

    epoch_total_batches = len(loader)
    total_batches_remaining = max(
        0,
        ((cfg["syncnet"]["epochs"] - start_epoch) * epoch_total_batches) - resume_batches_in_epoch,
    )
    training_t0 = time.time()
    for epoch in range(start_epoch, cfg["syncnet"]["epochs"]):
        model.train()
        total_loss = 0
        total_acc = 0
        train_batches_done = 0
        t0 = time.time()
        recent_batch_times = deque(maxlen=100)
        last_batch_end_t = t0
        loss_ema100 = None
        acc_ema100 = None
        epoch_resume_batches = resume_batches_in_epoch if epoch == start_epoch else 0
        epoch_batches_target = max(0, epoch_total_batches - epoch_resume_batches)
        if epoch_resume_batches > 0:
            log(
                f"Resuming epoch {epoch} at virtual batch {epoch_resume_batches}; "
                f"processing remaining {epoch_batches_target} fresh batches without replay"
            )
        if epoch_batches_target == 0:
            log(f"Epoch {epoch} already complete in checkpoint; rolling forward")
            resume_batches_in_epoch = 0
            continue

        for batch_offset, (visual, audio, labels) in enumerate(loader):
            if batch_offset >= epoch_batches_target:
                break
            virtual_batch_idx = epoch_resume_batches + batch_offset

            if model_type == "mirror" and visual.size(0) < 2:
                log(
                    f"  Skipping undersized train batch at E{epoch} "
                    f"[{virtual_batch_idx}/{len(loader)}]: batch_size={visual.size(0)}"
                )
                continue

            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = cuda_autocast(enabled=use_amp) if device == "cuda" else nullcontext()
            with amp_ctx:
                audio_emb, visual_emb = syncnet_forward(model, model_type, visual, audio)
                loss = syncnet_loss(model, model_type, audio_emb, visual_emb, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg["training"].get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["training"].get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()

            # Accuracy
            with torch.no_grad():
                cos_sim = torch.nn.functional.cosine_similarity(audio_emb, visual_emb)
                preds = syncnet_acc(cos_sim, model_type)
                acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            train_batches_done += 1
            loss_ema100 = update_ema(loss_ema100, loss.item(), 100)
            acc_ema100 = update_ema(acc_ema100, acc, 100)
            global_step += 1
            effective_epoch_batches = virtual_batch_idx + 1
            batch_end_t = time.time()
            recent_batch_times.append(batch_end_t - last_batch_end_t)
            last_batch_end_t = batch_end_t

            if virtual_batch_idx % 50 == 0:
                avg_batch_time = (
                    sum(recent_batch_times) / len(recent_batch_times)
                    if recent_batch_times else None
                )
                remaining_epoch_batches = max(0, epoch_total_batches - effective_epoch_batches)
                completed_total = (
                    ((epoch - start_epoch) * epoch_total_batches)
                    + (effective_epoch_batches - epoch_resume_batches)
                )
                remaining_total_batches = max(0, total_batches_remaining - completed_total)
                epoch_eta = compute_remaining_eta_from_recent(avg_batch_time, remaining_epoch_batches)
                full_eta = compute_remaining_eta_from_recent(avg_batch_time, remaining_total_batches)
                log(
                    f"  E{epoch} [{virtual_batch_idx}/{len(loader)}] "
                    f"loss={loss.item():.4f} acc={acc:.3f}"
                )
                log(
                    f"    ema100 loss={loss_ema100:.4f} acc={acc_ema100:.3f}"
                )
                log(
                    f"    eta epoch={format_eta(epoch_eta)} full={format_eta(full_eta)}"
                )

            if (
                val_loader is not None
                and eval_interval_steps > 0
                and global_step % eval_interval_steps == 0
            ):
                off_eval_metrics = evaluate_syncnet_model(
                    model,
                    model_type,
                    val_loader,
                    device,
                    max_batches=eval_batches,
                    use_amp=use_amp,
                    seed=eval_seed,
                )
                our_eval_metrics = None
                if pairwise_eval_samples > 0 and pairwise_eval_items:
                    our_eval_metrics = evaluate_syncnet_pairwise(
                        model,
                        model_type,
                        val_dataset,
                        pairwise_eval_items,
                        device,
                        samples=pairwise_eval_samples,
                        seed=pairwise_eval_seed,
                        img_size=cfg["model"]["img_size"],
                        T=cfg["syncnet"]["T"],
                    )
                if off_eval_metrics is not None:
                    log(
                        f"  Official eval step {global_step}: loss={off_eval_metrics['loss']:.4f} "
                        f"acc={off_eval_metrics['acc']:.3f}"
                        f"{format_official_eval_comparison(off_eval_metrics, official_baseline_off_eval, 'official')}"
                        f"{format_official_eval_comparison(off_eval_metrics, current_best_baseline_off_eval, current_best_label)}"
                    )
                    if our_eval_metrics is not None:
                        log(
                            "  Our eval step "
                            f"{global_step}: {format_pairwise_eval_metrics_flexible(our_eval_metrics)}"
                            f"{format_pairwise_eval_comparison(our_eval_metrics, official_baseline_our_eval, 'official')}"
                            f"{format_pairwise_eval_comparison(our_eval_metrics, current_best_baseline_our_eval, current_best_label)}"
                        )
                    is_best_off = off_eval_metrics["loss"] < best_off_eval_loss
                    current_best_off_eval_loss = off_eval_metrics["loss"] if is_best_off else best_off_eval_loss
                    current_best_off_eval_step = global_step if is_best_off else best_off_eval_step
                    is_best_our = False
                    current_best_our_eval_acc = best_our_eval_acc
                    current_best_our_eval_margin = best_our_eval_margin
                    current_best_our_eval_step = best_our_eval_step
                    if our_eval_metrics is not None:
                        is_best_our = (
                            our_eval_metrics["pairwise_acc_mean"] > best_our_eval_acc
                            or (
                                our_eval_metrics["pairwise_acc_mean"] == best_our_eval_acc
                                and our_eval_metrics["margin_mean"] > best_our_eval_margin
                            )
                        )
                        if is_best_our:
                            current_best_our_eval_acc = our_eval_metrics["pairwise_acc_mean"]
                            current_best_our_eval_margin = our_eval_metrics["margin_mean"]
                            current_best_our_eval_step = global_step
                    save_syncnet_checkpoint(
                        latest_ckpt_path,
                        epoch=epoch,
                        global_step=global_step,
                        batches_processed_in_epoch=effective_epoch_batches,
                        checkpoint_kind="step",
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        model_type=model_type,
                        speaker_list=args.speaker_list,
                        val_speaker_list=args.val_speaker_list,
                        train_metrics={"loss": loss.item(), "acc": acc},
                        official_eval_metrics=off_eval_metrics,
                        our_eval_metrics=our_eval_metrics,
                        best_off_eval_loss=current_best_off_eval_loss,
                        best_off_eval_step=current_best_off_eval_step,
                        best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                        best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                        best_our_eval_step=current_best_our_eval_step,
                        official_baseline_off_eval_metrics=official_baseline_off_eval,
                        official_baseline_our_eval_metrics=official_baseline_our_eval,
                    )
                    log(f"  Saved latest {latest_ckpt_path}")
                    if is_best_off:
                        best_off_eval_loss = off_eval_metrics["loss"]
                        best_off_eval_step = global_step
                        save_syncnet_checkpoint(
                            best_off_eval_ckpt_path,
                            epoch=epoch,
                            global_step=global_step,
                            batches_processed_in_epoch=effective_epoch_batches,
                            checkpoint_kind="step",
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            model_type=model_type,
                            speaker_list=args.speaker_list,
                            val_speaker_list=args.val_speaker_list,
                            train_metrics={"loss": loss.item(), "acc": acc},
                            official_eval_metrics=off_eval_metrics,
                            our_eval_metrics=our_eval_metrics,
                            best_off_eval_loss=best_off_eval_loss,
                            best_off_eval_step=best_off_eval_step,
                            best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                            best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                            best_our_eval_step=current_best_our_eval_step,
                            official_baseline_off_eval_metrics=official_baseline_off_eval,
                            official_baseline_our_eval_metrics=official_baseline_our_eval,
                        )
                        save_syncnet_checkpoint(
                            legacy_best_ckpt_path,
                            epoch=epoch,
                            global_step=global_step,
                            batches_processed_in_epoch=effective_epoch_batches,
                            checkpoint_kind="step",
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            model_type=model_type,
                            speaker_list=args.speaker_list,
                            val_speaker_list=args.val_speaker_list,
                            train_metrics={"loss": loss.item(), "acc": acc},
                            official_eval_metrics=off_eval_metrics,
                            our_eval_metrics=our_eval_metrics,
                            best_off_eval_loss=best_off_eval_loss,
                            best_off_eval_step=best_off_eval_step,
                            best_our_eval_acc=current_best_our_eval_acc if current_best_our_eval_acc > float("-inf") else None,
                            best_our_eval_margin=current_best_our_eval_margin if current_best_our_eval_margin > float("-inf") else None,
                            best_our_eval_step=current_best_our_eval_step,
                            official_baseline_off_eval_metrics=official_baseline_off_eval,
                            official_baseline_our_eval_metrics=official_baseline_our_eval,
                        )
                        log(
                            f"  New best off-eval: loss={best_off_eval_loss:.4f} "
                            f"at step {best_off_eval_step} -> {best_off_eval_ckpt_path}"
                        )
                    if is_best_our:
                        best_our_eval_acc = our_eval_metrics["pairwise_acc_mean"]
                        best_our_eval_margin = our_eval_metrics["margin_mean"]
                        best_our_eval_step = global_step
                        save_syncnet_checkpoint(
                            best_our_eval_ckpt_path,
                            epoch=epoch,
                            global_step=global_step,
                            batches_processed_in_epoch=effective_epoch_batches,
                            checkpoint_kind="step",
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            model_type=model_type,
                            speaker_list=args.speaker_list,
                            val_speaker_list=args.val_speaker_list,
                            train_metrics={"loss": loss.item(), "acc": acc},
                            official_eval_metrics=off_eval_metrics,
                            our_eval_metrics=our_eval_metrics,
                            best_off_eval_loss=best_off_eval_loss if best_off_eval_loss < float("inf") else None,
                            best_off_eval_step=best_off_eval_step,
                            best_our_eval_acc=best_our_eval_acc,
                            best_our_eval_margin=best_our_eval_margin,
                            best_our_eval_step=best_our_eval_step,
                            official_baseline_off_eval_metrics=official_baseline_off_eval,
                            official_baseline_our_eval_metrics=official_baseline_our_eval,
                        )
                        log(
                            f"  New best our-eval: pairwise_acc={best_our_eval_acc:.3f} "
                            f"margin={best_our_eval_margin:.4f} at step {best_our_eval_step} "
                            f"-> {best_our_eval_ckpt_path}"
                        )
                last_batch_end_t = time.time()

            if batch_offset + 1 >= epoch_batches_target:
                break

        epoch_batches_done = max(1, train_batches_done)
        avg_loss = total_loss / epoch_batches_done
        avg_acc = total_acc / epoch_batches_done
        elapsed = time.time() - t0
        avg_batch_time = (
            sum(recent_batch_times) / len(recent_batch_times)
            if recent_batch_times else None
        )
        completed_total = (
            ((epoch - start_epoch) * epoch_total_batches)
            + (epoch_total_batches - epoch_resume_batches)
        )
        remaining_total_batches = max(0, total_batches_remaining - completed_total)
        full_eta = compute_remaining_eta_from_recent(avg_batch_time, remaining_total_batches)
        next_epoch_eta = compute_remaining_eta_from_recent(avg_batch_time, epoch_total_batches)
        if next_epoch_eta is None:
            next_epoch_eta = elapsed

        log(
            f"Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.3f} "
            f"({elapsed:.0f}s)"
        )
        log(
            f"  Epoch {epoch} ema100: loss={loss_ema100:.4f} acc={acc_ema100:.3f}"
        )
        log(
            f"  ETA next_epoch={format_eta(next_epoch_eta)} full={format_eta(full_eta)}"
        )

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 or epoch == cfg["syncnet"]["epochs"] - 1:
            ck_path = os.path.join(output_dir, f"syncnet_epoch{epoch:03d}.pth")
            save_syncnet_checkpoint(
                ck_path,
                epoch=epoch,
                global_step=global_step,
                batches_processed_in_epoch=epoch_total_batches,
                checkpoint_kind="epoch",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                model_type=model_type,
                speaker_list=args.speaker_list,
                val_speaker_list=args.val_speaker_list,
                train_metrics={"loss": avg_loss, "acc": avg_acc},
                best_off_eval_loss=best_off_eval_loss if best_off_eval_loss < float("inf") else None,
                best_off_eval_step=best_off_eval_step,
                best_our_eval_acc=best_our_eval_acc if best_our_eval_acc > float("-inf") else None,
                best_our_eval_margin=best_our_eval_margin if best_our_eval_margin > float("-inf") else None,
                best_our_eval_step=best_our_eval_step,
                official_baseline_off_eval_metrics=official_baseline_off_eval,
                official_baseline_our_eval_metrics=official_baseline_our_eval,
            )
            log(f"  Saved {ck_path}")
            if val_loader is not None:
                save_syncnet_checkpoint(
                    latest_ckpt_path,
                    epoch=epoch,
                    global_step=global_step,
                    batches_processed_in_epoch=epoch_total_batches,
                    checkpoint_kind="epoch",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    model_type=model_type,
                    speaker_list=args.speaker_list,
                    val_speaker_list=args.val_speaker_list,
                    train_metrics={"loss": avg_loss, "acc": avg_acc},
                    best_off_eval_loss=best_off_eval_loss if best_off_eval_loss < float("inf") else None,
                    best_off_eval_step=best_off_eval_step,
                    best_our_eval_acc=best_our_eval_acc if best_our_eval_acc > float("-inf") else None,
                    best_our_eval_margin=best_our_eval_margin if best_our_eval_margin > float("-inf") else None,
                    best_our_eval_step=best_our_eval_step,
                    official_baseline_off_eval_metrics=official_baseline_off_eval,
                    official_baseline_our_eval_metrics=official_baseline_our_eval,
                )

    log("SyncNet training complete!")


if __name__ == "__main__":
    main()
