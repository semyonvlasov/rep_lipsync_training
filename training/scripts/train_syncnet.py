#!/usr/bin/env python3
"""
Stage 1: Train SyncNet — audio-visual synchronization discriminator.

Usage:
    python scripts/train_syncnet.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TRAINING_ROOT, "models"))
sys.path.insert(0, TRAINING_ROOT)
from syncnet import SyncNet
from data import LipSyncDataset
from scripts.dataset_roots import get_dataset_roots


def update_ema(current, value, span):
    alpha = 2.0 / (span + 1.0)
    if current is None:
        return value
    return (alpha * value) + ((1.0 - alpha) * current)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated list of speaker dirs to include")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    output_dir = os.path.join(cfg["training"]["output_dir"], "syncnet")
    os.makedirs(output_dir, exist_ok=True)

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
    dataset = LipSyncDataset(
        roots=roots,
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="syncnet",
        cache_size=cfg["data"].get("cache_size", 8),
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        speaker_allowlist=speaker_allowlist,
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
        min_samples_per_speaker=cfg["data"].get("min_samples_per_speaker", 100),
    )
    loader_kwargs = {
        "batch_size": cfg["syncnet"]["batch_size"],
        "shuffle": True,
        "num_workers": cfg["data"]["num_workers"],
        "pin_memory": (device == "cuda"),
        "drop_last": True,
    }
    if device == "cuda":
        loader_kwargs["pin_memory_device"] = "cuda"
    if cfg["data"]["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = cfg["data"].get("persistent_workers", True)
        prefetch_factor = cfg["data"].get("prefetch_factor")
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    # Model
    model = SyncNet(
        T=cfg["syncnet"]["T"],
        audio_temporal_kernels=cfg["model"].get("audio_temporal_kernels"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["syncnet"]["lr"])
    use_amp = cfg["training"].get("mixed_precision", False) and device == "cuda"
    scaler = GradScaler(enabled=use_amp) if device == "cuda" else None

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        log(f"Resumed from epoch {start_epoch}")

    log(f"SyncNet dataset roots: {roots}")
    log(f"SyncNet training: {len(dataset)} samples, {len(loader)} batches/epoch")
    log(f"Device: {device}, T={cfg['syncnet']['T']}, AMP={use_amp}")

    for epoch in range(start_epoch, cfg["syncnet"]["epochs"]):
        model.train()
        total_loss = 0
        total_acc = 0
        t0 = time.time()
        loss_ema50 = None
        loss_ema100 = None
        acc_ema50 = None
        acc_ema100 = None

        for batch_idx, (visual, audio, labels) in enumerate(loader):
            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = autocast(enabled=use_amp) if device == "cuda" else nullcontext()
            with amp_ctx:
                v_emb, a_emb = model(visual, audio)
                loss = SyncNet.cosine_loss(v_emb, a_emb, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                optimizer.step()

            # Accuracy
            with torch.no_grad():
                cos_sim = torch.nn.functional.cosine_similarity(v_emb, a_emb)
                preds = (cos_sim > 0).float()
                acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            loss_ema50 = update_ema(loss_ema50, loss.item(), 50)
            loss_ema100 = update_ema(loss_ema100, loss.item(), 100)
            acc_ema50 = update_ema(acc_ema50, acc, 50)
            acc_ema100 = update_ema(acc_ema100, acc, 100)

            if batch_idx % 50 == 0:
                log(
                    f"  E{epoch} [{batch_idx}/{len(loader)}] "
                    f"loss={loss.item():.4f} acc={acc:.3f} "
                    f"ema50_loss={loss_ema50:.4f} ema50_acc={acc_ema50:.3f} "
                    f"ema100_loss={loss_ema100:.4f} ema100_acc={acc_ema100:.3f}"
                )

        avg_loss = total_loss / max(1, len(loader))
        avg_acc = total_acc / max(1, len(loader))
        elapsed = time.time() - t0

        log(
            f"Epoch {epoch}: loss={avg_loss:.4f} acc={avg_acc:.3f} "
            f"ema50_loss={loss_ema50:.4f} ema50_acc={acc_ema50:.3f} "
            f"ema100_loss={loss_ema100:.4f} ema100_acc={acc_ema100:.3f} "
            f"({elapsed:.0f}s)"
        )

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 or epoch == cfg["syncnet"]["epochs"] - 1:
            ck_path = os.path.join(output_dir, f"syncnet_epoch{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "acc": avg_acc,
                "config": cfg,
                "speaker_list": args.speaker_list,
            }, ck_path)
            log(f"  Saved {ck_path}")

    log("SyncNet training complete!")


if __name__ == "__main__":
    main()
