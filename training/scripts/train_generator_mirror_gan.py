#!/usr/bin/env python3
"""
Official-style Wav2Lip HQ GAN generator trainer on the merged lazy dataset.

This mirrors the upstream `models/wav2lip/hq_wav2lip_train.py` training loop,
checkpoint cadence, and loss composition as closely as possible while using:
  - merged HDTF/TalkVid dataset roots from repo configs
  - our lazy/materialized dataset backend
  - speaker snapshot lists instead of LRS2 filelists

Deliberate compatibility shims:
  - cosine->BCE uses a probability clamp to stay valid on modern PyTorch
  - the official discriminator perceptual path is patched to allocate targets on
    the current device instead of hard-coding `.cuda()`
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
sys.path.insert(0, TRAINING_ROOT)

from data import LipSyncDataset
from scripts.dataset_roots import get_dataset_roots


global_step = 0
global_epoch = 0

syncnet_T = 5
syncnet_mel_step_size = 16

def log(msg: str) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_official_wav2lip_classes():
    candidate_roots = [
        Path(REPO_ROOT) / "models" / "official_syncnet",
        Path(REPO_ROOT) / "models" / "wav2lip",
        Path(REPO_ROOT).parent / "rep_lipsync_training" / "models" / "official_syncnet",
        Path(REPO_ROOT).parent / "models" / "wav2lip",
    ]
    for root in candidate_roots:
        if not (root / "models").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        if "models" in sys.modules:
            del sys.modules["models"]
        from models import SyncNet_color, Wav2Lip, Wav2Lip_disc_qual

        def _patched_perceptual_forward(self, false_face_sequences):
            false_face_sequences = self.to_2d(false_face_sequences)
            false_face_sequences = self.get_lower_half(false_face_sequences)

            false_feats = false_face_sequences
            for block in self.face_encoder_blocks:
                false_feats = block(false_feats)

            pred = self.binary_pred(false_feats).view(len(false_feats), -1)
            targets = torch.ones((len(false_feats), 1), device=false_feats.device, dtype=pred.dtype)
            return F.binary_cross_entropy(pred, targets)

        Wav2Lip_disc_qual.perceptual_forward = _patched_perceptual_forward
        return SyncNet_color, Wav2Lip, Wav2Lip_disc_qual
    raise RuntimeError("Could not locate local official Wav2Lip package root")


OfficialSyncNet, OfficialWav2Lip, OfficialWav2LipDiscQual = load_official_wav2lip_classes()


def load_allowlist(path: str | None):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_sync_alignment_kwargs(cfg: dict) -> dict:
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
        "sync_alignment_syncnet_checkpoint": sync_cfg.get("syncnet_checkpoint"),
        "sync_alignment_write_manifest": sync_cfg.get("write_manifest", True),
    }


class GeneratorMirrorGanDataset(LipSyncDataset):
    """Match the upstream HQ trainer sampling semantics."""

    def __len__(self):
        return max(self._total_frames, len(self.speakers) * 100)

    def __getitem__(self, idx):
        del idx
        if not self.speakers:
            raise RuntimeError("GeneratorMirrorGanDataset has no speakers")

        while True:
            speaker_idx = random.choices(
                range(len(self.speakers)),
                weights=self._frame_counts,
                k=1,
            )[0]
            speaker_key = self.speakers[speaker_idx]
            try:
                frames, mel_chunks, _ = self._load_speaker(speaker_key)
            except Exception:
                continue

            n_frames = min(len(frames), len(mel_chunks))
            if n_frames <= 3 * self.syncnet_T:
                continue

            start = random.randint(0, n_frames - self.syncnet_T)
            wrong_start = random.randint(0, n_frames - self.syncnet_T)
            while wrong_start == start:
                wrong_start = random.randint(0, n_frames - self.syncnet_T)

            return self._build_generator_sample(frames, mel_chunks, start, wrong_start)


def build_dataset(cfg: dict, roots: list[str], speaker_allowlist):
    return GeneratorMirrorGanDataset(
        roots=roots,
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="generator",
        cache_size=cfg["data"].get("cache_size", 8),
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        speaker_allowlist=speaker_allowlist,
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
        **build_sync_alignment_kwargs(cfg),
    )


def build_loader(dataset, batch_size: int, num_workers: int, device: str, shuffle: bool):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": True,
        "pin_memory": device == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def _normalize_official_state_dict(state_dict: dict) -> OrderedDict:
    new_state = OrderedDict()
    for key, value in state_dict.items():
        new_state[key.replace("module.", "")] = value
    return new_state


def load_syncnet_teacher(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OfficialSyncNet().to(device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        cfg = checkpoint.get("config") or {}
        sync_cfg = cfg.get("syncnet") or {}
        kind = checkpoint.get("syncnet_kind") or sync_cfg.get("model_type") or "local"
        if kind != "mirror":
            raise RuntimeError(
                f"generator_mirror_gan expects an official/mirror SyncNet checkpoint, got kind={kind!r}"
            )
        model.load_state_dict(_normalize_official_state_dict(checkpoint["model"]))
        epoch = checkpoint.get("epoch")
        step = checkpoint.get("global_step")
        source = "mirror"
    else:
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(_normalize_official_state_dict(state_dict))
        epoch = checkpoint.get("global_epoch") if isinstance(checkpoint, dict) else None
        step = checkpoint.get("global_step") if isinstance(checkpoint, dict) else None
        source = "official"

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, source, epoch, step


def official_sync_loss_from_cosine(cos_sim: torch.Tensor) -> torch.Tensor:
    targets = torch.ones((cos_sim.size(0), 1), device=cos_sim.device, dtype=torch.float32)
    probs = torch.nan_to_num(
        cos_sim.float(),
        nan=0.5,
        posinf=1.0 - 1.0e-6,
        neginf=1.0e-6,
    ).clamp_(1.0e-6, 1.0 - 1.0e-6).unsqueeze(1)
    return F.binary_cross_entropy(probs, targets)


def cosine_loss(a: torch.Tensor, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    del y
    d = F.cosine_similarity(a, v)
    return official_sync_loss_from_cosine(d)


def get_sync_loss(syncnet: nn.Module, mel: torch.Tensor, g: torch.Tensor, device: torch.device) -> torch.Tensor:
    g = g[:, :, :, g.size(3) // 2 :]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1, device=device)
    return cosine_loss(a, v, y)


def save_sample_images(x, g, gt, step: int, checkpoint_dir: str) -> None:
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = os.path.join(checkpoint_dir, f"samples_step{step:09d}")
    os.makedirs(folder, exist_ok=True)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, sample in enumerate(collage):
        for t in range(len(sample)):
            cv2.imwrite(os.path.join(folder, f"{batch_idx}_{t}.jpg"), sample[t])


def better_official_eval(
    candidate_sync,
    candidate_l1,
    candidate_step,
    best_sync,
    best_l1,
    best_step,
    eps=1.0e-8,
):
    if candidate_sync < best_sync - eps:
        return True
    if abs(candidate_sync - best_sync) <= eps:
        if candidate_l1 < best_l1 - eps:
            return True
        if abs(candidate_l1 - best_l1) <= eps and (best_step is None or candidate_step > best_step):
            return True
    return False


def better_scalar_metric(
    candidate_value,
    candidate_step,
    best_value,
    best_step,
    candidate_secondary=None,
    best_secondary=None,
    eps=1.0e-8,
):
    if candidate_value < best_value - eps:
        return True
    if abs(candidate_value - best_value) <= eps:
        if candidate_secondary is not None:
            ref_secondary = float("inf") if best_secondary is None else best_secondary
            if candidate_secondary < ref_secondary - eps:
                return True
            if abs(candidate_secondary - ref_secondary) <= eps and (best_step is None or candidate_step > best_step):
                return True
        elif best_step is None or candidate_step > best_step:
            return True
    return False


def build_best_evals_payload(
    *,
    off_sync=None,
    off_l1=None,
    off_perceptual=None,
    off_step=None,
    l1_value=None,
    l1_sync=None,
    l1_perceptual=None,
    l1_step=None,
    perceptual_value=None,
    perceptual_sync=None,
    perceptual_l1=None,
    perceptual_step=None,
):
    payload = {}
    if off_step is not None:
        payload["off"] = {
            "sync": float(off_sync),
            "l1": float(off_l1),
            "perceptual": None if off_perceptual is None else float(off_perceptual),
            "step": int(off_step),
        }
    if l1_step is not None:
        payload["l1"] = {
            "l1": float(l1_value),
            "sync": None if l1_sync is None else float(l1_sync),
            "perceptual": None if l1_perceptual is None else float(l1_perceptual),
            "step": int(l1_step),
        }
    if perceptual_step is not None:
        payload["perceptual"] = {
            "perceptual": float(perceptual_value),
            "sync": None if perceptual_sync is None else float(perceptual_sync),
            "l1": None if perceptual_l1 is None else float(perceptual_l1),
            "step": int(perceptual_step),
        }
    return payload


def extract_best_evals_from_checkpoint(ck):
    payload = ck.get("best_evals")
    if isinstance(payload, dict):
        return payload

    legacy = {}
    if ck.get("best_off_eval_step") is not None:
        legacy["off"] = {
            "sync": float(ck["best_off_eval_sync"]),
            "l1": float(ck["best_off_eval_l1"]),
            "perceptual": None if ck.get("best_off_eval_perceptual") is None else float(ck["best_off_eval_perceptual"]),
            "step": int(ck["best_off_eval_step"]),
        }
    if ck.get("best_l1_eval_step") is not None:
        legacy["l1"] = {
            "l1": float(ck["best_l1_eval_l1"]),
            "sync": None if ck.get("best_l1_eval_sync") is None else float(ck["best_l1_eval_sync"]),
            "perceptual": None if ck.get("best_l1_eval_perceptual") is None else float(ck["best_l1_eval_perceptual"]),
            "step": int(ck["best_l1_eval_step"]),
        }
    if ck.get("best_perceptual_eval_step") is not None:
        legacy["perceptual"] = {
            "perceptual": float(ck["best_perceptual_eval_perceptual"]),
            "sync": None if ck.get("best_perceptual_eval_sync") is None else float(ck["best_perceptual_eval_sync"]),
            "l1": None if ck.get("best_perceptual_eval_l1") is None else float(ck["best_perceptual_eval_l1"]),
            "step": int(ck["best_perceptual_eval_step"]),
        }
    return legacy


def build_checkpoint_payload(model, optimizer, step: int, epoch: int, save_optimizer_state: bool, best_evals=None):
    optimizer_state = optimizer.state_dict() if (optimizer is not None and save_optimizer_state) else None
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": int(step),
        "global_epoch": int(epoch),
    }
    if best_evals:
        checkpoint["best_evals"] = best_evals
        off_eval = best_evals.get("off")
        if off_eval is not None:
            checkpoint["best_off_eval_sync"] = float(off_eval["sync"])
            checkpoint["best_off_eval_l1"] = float(off_eval["l1"])
            checkpoint["best_off_eval_perceptual"] = (
                None if off_eval.get("perceptual") is None else float(off_eval["perceptual"])
            )
            checkpoint["best_off_eval_step"] = int(off_eval["step"])
        l1_eval = best_evals.get("l1")
        if l1_eval is not None:
            checkpoint["best_l1_eval_l1"] = float(l1_eval["l1"])
            checkpoint["best_l1_eval_sync"] = (
                None if l1_eval.get("sync") is None else float(l1_eval["sync"])
            )
            checkpoint["best_l1_eval_perceptual"] = (
                None if l1_eval.get("perceptual") is None else float(l1_eval["perceptual"])
            )
            checkpoint["best_l1_eval_step"] = int(l1_eval["step"])
        perceptual_eval = best_evals.get("perceptual")
        if perceptual_eval is not None:
            checkpoint["best_perceptual_eval_perceptual"] = float(perceptual_eval["perceptual"])
            checkpoint["best_perceptual_eval_sync"] = (
                None if perceptual_eval.get("sync") is None else float(perceptual_eval["sync"])
            )
            checkpoint["best_perceptual_eval_l1"] = (
                None if perceptual_eval.get("l1") is None else float(perceptual_eval["l1"])
            )
            checkpoint["best_perceptual_eval_step"] = int(perceptual_eval["step"])
    return checkpoint


def save_checkpoint(
    model,
    optimizer,
    step: int,
    checkpoint_dir: str,
    epoch: int,
    save_optimizer_state: bool,
    prefix: str = "",
    filename: str | None = None,
    best_evals=None,
):
    checkpoint_path = os.path.join(
        checkpoint_dir,
        filename if filename is not None else f"{prefix}checkpoint_step{step:09d}.pth",
    )
    torch.save(
        build_checkpoint_payload(
            model,
            optimizer,
            step,
            epoch,
            save_optimizer_state,
            best_evals=best_evals,
        ),
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)


def _load_checkpoint(path: str, use_cuda: bool):
    if use_cuda:
        return torch.load(path, weights_only=False)
    return torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)


def load_checkpoint(path: str, model, optimizer, reset_optimizer: bool = False, overwrite_global_states: bool = True):
    global global_step
    global global_epoch

    print(f"Load checkpoint from: {path}", flush=True)
    checkpoint = _load_checkpoint(path, torch.cuda.is_available())
    state_dict = _normalize_official_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    if not reset_optimizer and optimizer is not None:
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            print(f"Load optimizer state from {path}", flush=True)
            optimizer.load_state_dict(optimizer_state)
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
    return model


def eval_model(test_data_loader, device, model, disc, syncnet, cfg_mirror: dict) -> dict[str, float]:
    eval_steps = int(cfg_mirror["eval_steps"])
    print(f"Evaluating for {eval_steps} steps", flush=True)
    running_sync_loss = []
    running_l1_loss = []
    running_disc_real_loss = []
    running_disc_fake_loss = []
    running_perceptual_loss = []
    recon_loss = nn.L1Loss()

    while True:
        for step, (x, indiv_mels, mel, gt) in enumerate(test_data_loader):
            model.eval()
            disc.eval()

            x = x.to(device, non_blocking=device.type == "cuda")
            mel = mel.to(device, non_blocking=device.type == "cuda")
            indiv_mels = indiv_mels.to(device, non_blocking=device.type == "cuda")
            gt = gt.to(device, non_blocking=device.type == "cuda")

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1), device=device))

            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1), device=device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(syncnet, mel, g, device)
            perceptual_loss = disc.perceptual_forward(g) if cfg_mirror["disc_wt"] > 0 else 0.0
            l1loss = recon_loss(g, gt)

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            running_perceptual_loss.append(
                perceptual_loss.item() if hasattr(perceptual_loss, "item") else float(perceptual_loss)
            )

            if step > eval_steps:
                break

        metrics = {
            "l1": sum(running_l1_loss) / len(running_l1_loss),
            "sync": sum(running_sync_loss) / len(running_sync_loss),
            "perceptual": sum(running_perceptual_loss) / len(running_perceptual_loss),
            "disc_fake": sum(running_disc_fake_loss) / len(running_disc_fake_loss),
            "disc_real": sum(running_disc_real_loss) / len(running_disc_real_loss),
        }
        print(
            "L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}".format(
                metrics["l1"],
                metrics["sync"],
                metrics["perceptual"],
                metrics["disc_fake"],
                metrics["disc_real"],
            ),
            flush=True,
        )
        return metrics


def resolve_mirror_cfg(cfg: dict) -> dict:
    section = dict(cfg.get("generator_mirror_gan") or {})
    generator_cfg = cfg.get("generator") or {}

    resolved = {
        "epochs": int(section.get("epochs", generator_cfg.get("epochs", 200))),
        "batch_size": int(section.get("batch_size", generator_cfg.get("batch_size", 16))),
        "lr": float(section.get("lr", generator_cfg.get("lr", 1.0e-4))),
        "disc_lr": float(section.get("disc_lr", section.get("lr", generator_cfg.get("lr", 1.0e-4)))),
        "betas": tuple(section.get("betas", generator_cfg.get("gan_betas", (0.5, 0.999)))),
        "checkpoint_interval_steps": int(section.get("checkpoint_interval_steps", 3000)),
        "sample_interval_steps": int(section.get("sample_interval_steps", 0)),
        "save_sample_images": bool(section.get("save_sample_images", False)),
        "eval_interval_steps": int(section.get("eval_interval_steps", 3000)),
        "eval_steps": int(section.get("eval_steps", 300)),
        "save_optimizer_state": bool(section.get("save_optimizer_state", True)),
        "syncnet_wt_initial": float(section.get("syncnet_wt_initial", 0.0)),
        "syncnet_wt_target": float(section.get("syncnet_wt_target", 0.03)),
        "syncnet_enable_threshold": float(section.get("syncnet_enable_threshold", 0.75)),
        "disc_wt": float(section.get("disc_wt", 0.07)),
        "val_num_workers": int(section.get("val_num_workers", 4)),
    }
    if resolved["syncnet_wt_initial"] + resolved["disc_wt"] >= 1.0:
        raise ValueError("generator_mirror_gan requires syncnet_wt_initial + disc_wt < 1.0")
    if resolved["syncnet_wt_target"] + resolved["disc_wt"] >= 1.0:
        raise ValueError("generator_mirror_gan requires syncnet_wt_target + disc_wt < 1.0")
    return resolved


def train(
    device,
    model,
    disc,
    syncnet,
    train_data_loader,
    test_data_loader,
    optimizer,
    disc_optimizer,
    checkpoint_dir: str,
    cfg_mirror: dict,
    initial_best_evals=None,
):
    global global_step
    global global_epoch

    syncnet_wt = float(cfg_mirror["syncnet_wt_initial"])
    resumed_step = global_step
    recon_loss = nn.L1Loss()
    restored_best_evals = initial_best_evals or {}
    off_eval = restored_best_evals.get("off")
    l1_eval = restored_best_evals.get("l1")
    perceptual_eval = restored_best_evals.get("perceptual")

    best_off_eval_sync = float(off_eval["sync"]) if off_eval is not None else float("inf")
    best_off_eval_l1 = float(off_eval["l1"]) if off_eval is not None else float("inf")
    best_off_eval_perceptual = (
        float(off_eval["perceptual"]) if (off_eval is not None and off_eval.get("perceptual") is not None) else float("inf")
    )
    best_off_eval_step = int(off_eval["step"]) if off_eval is not None else None

    best_l1_eval_l1 = float(l1_eval["l1"]) if l1_eval is not None else float("inf")
    best_l1_eval_sync = (
        float(l1_eval["sync"]) if (l1_eval is not None and l1_eval.get("sync") is not None) else float("inf")
    )
    best_l1_eval_perceptual = (
        float(l1_eval["perceptual"]) if (l1_eval is not None and l1_eval.get("perceptual") is not None) else float("inf")
    )
    best_l1_eval_step = int(l1_eval["step"]) if l1_eval is not None else None

    best_perceptual_eval_perceptual = (
        float(perceptual_eval["perceptual"]) if perceptual_eval is not None else float("inf")
    )
    best_perceptual_eval_sync = (
        float(perceptual_eval["sync"])
        if (perceptual_eval is not None and perceptual_eval.get("sync") is not None)
        else float("inf")
    )
    best_perceptual_eval_l1 = (
        float(perceptual_eval["l1"])
        if (perceptual_eval is not None and perceptual_eval.get("l1") is not None)
        else float("inf")
    )
    best_perceptual_eval_step = int(perceptual_eval["step"]) if perceptual_eval is not None else None

    latest_ckpt_path = os.path.join(checkpoint_dir, "generator_latest.pth")
    disc_latest_ckpt_path = os.path.join(checkpoint_dir, "disc_latest.pth")
    best_off_eval_ckpt_path = os.path.join(checkpoint_dir, "generator_best_off_eval.pth")
    disc_best_off_eval_ckpt_path = os.path.join(checkpoint_dir, "disc_best_off_eval.pth")
    best_l1_eval_ckpt_path = os.path.join(checkpoint_dir, "generator_best_l1_eval.pth")
    disc_best_l1_eval_ckpt_path = os.path.join(checkpoint_dir, "disc_best_l1_eval.pth")
    best_perceptual_eval_ckpt_path = os.path.join(checkpoint_dir, "generator_best_perceptual_eval.pth")
    disc_best_perceptual_eval_ckpt_path = os.path.join(checkpoint_dir, "disc_best_perceptual_eval.pth")

    def current_best_evals_payload():
        return build_best_evals_payload(
            off_sync=best_off_eval_sync if best_off_eval_step is not None else None,
            off_l1=best_off_eval_l1 if best_off_eval_step is not None else None,
            off_perceptual=best_off_eval_perceptual if best_off_eval_step is not None else None,
            off_step=best_off_eval_step,
            l1_value=best_l1_eval_l1 if best_l1_eval_step is not None else None,
            l1_sync=best_l1_eval_sync if best_l1_eval_step is not None else None,
            l1_perceptual=best_l1_eval_perceptual if best_l1_eval_step is not None else None,
            l1_step=best_l1_eval_step,
            perceptual_value=best_perceptual_eval_perceptual if best_perceptual_eval_step is not None else None,
            perceptual_sync=best_perceptual_eval_sync if best_perceptual_eval_step is not None else None,
            perceptual_l1=best_perceptual_eval_l1 if best_perceptual_eval_step is not None else None,
            perceptual_step=best_perceptual_eval_step,
        )

    def save_named_checkpoint_pair(generator_filename: str, disc_filename: str):
        best_evals = current_best_evals_payload()
        save_checkpoint(
            model,
            optimizer,
            global_step,
            checkpoint_dir,
            global_epoch,
            cfg_mirror["save_optimizer_state"],
            filename=generator_filename,
            best_evals=best_evals,
        )
        save_checkpoint(
            disc,
            disc_optimizer,
            global_step,
            checkpoint_dir,
            global_epoch,
            cfg_mirror["save_optimizer_state"],
            filename=disc_filename,
            best_evals=best_evals,
        )

    log(f"Named latest checkpoint path: {latest_ckpt_path}")
    log(f"Named latest discriminator path: {disc_latest_ckpt_path}")
    log(f"Named best sync checkpoint path: {best_off_eval_ckpt_path}")
    log(f"Named best sync discriminator path: {disc_best_off_eval_ckpt_path}")
    log(f"Named best L1 checkpoint path: {best_l1_eval_ckpt_path}")
    log(f"Named best L1 discriminator path: {disc_best_l1_eval_ckpt_path}")
    log(f"Named best perceptual checkpoint path: {best_perceptual_eval_ckpt_path}")
    log(f"Named best perceptual discriminator path: {disc_best_perceptual_eval_ckpt_path}")

    while global_epoch < cfg_mirror["epochs"]:
        print(f"Starting Epoch: {global_epoch}", flush=True)
        running_sync_loss, running_l1_loss, running_perceptual_loss = 0.0, 0.0, 0.0
        running_disc_real_loss, running_disc_fake_loss = 0.0, 0.0

        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            disc.train()
            model.train()

            x = x.to(device, non_blocking=device.type == "cuda")
            mel = mel.to(device, non_blocking=device.type == "cuda")
            indiv_mels = indiv_mels.to(device, non_blocking=device.type == "cuda")
            gt = gt.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)

            if syncnet_wt > 0.0:
                sync_loss = get_sync_loss(syncnet, mel, g, device)
            else:
                sync_loss = torch.tensor(0.0, device=device)

            if cfg_mirror["disc_wt"] > 0.0:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = torch.tensor(0.0, device=device)

            l1loss = recon_loss(g, gt)
            loss = (
                syncnet_wt * sync_loss
                + cfg_mirror["disc_wt"] * perceptual_loss
                + (1.0 - syncnet_wt - cfg_mirror["disc_wt"]) * l1loss
            )

            loss.backward()
            optimizer.step()

            disc_optimizer.zero_grad()

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1), device=device))
            disc_real_loss.backward()

            pred = disc(g.detach())
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1), device=device))
            disc_fake_loss.backward()

            disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            if (
                cfg_mirror["save_sample_images"]
                and cfg_mirror["sample_interval_steps"] > 0
                and global_step % cfg_mirror["sample_interval_steps"] == 0
            ):
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1
            _cur_session_steps = global_step - resumed_step
            del _cur_session_steps

            running_l1_loss += l1loss.item()
            running_sync_loss += sync_loss.item() if syncnet_wt > 0.0 else 0.0
            running_perceptual_loss += (
                perceptual_loss.item() if cfg_mirror["disc_wt"] > 0.0 else 0.0
            )

            if global_step == 1 or global_step % cfg_mirror["checkpoint_interval_steps"] == 0:
                save_named_checkpoint_pair("generator_latest.pth", "disc_latest.pth")

            if (
                test_data_loader is not None
                and cfg_mirror["eval_interval_steps"] > 0
                and global_step % cfg_mirror["eval_interval_steps"] == 0
            ):
                with torch.no_grad():
                    eval_metrics = eval_model(test_data_loader, device, model, disc, syncnet, cfg_mirror)
                    average_sync_loss = float(eval_metrics["sync"])
                    average_l1_loss = float(eval_metrics["l1"])
                    average_perceptual_loss = float(eval_metrics["perceptual"])
                    if average_sync_loss < cfg_mirror["syncnet_enable_threshold"]:
                        syncnet_wt = cfg_mirror["syncnet_wt_target"]
                    if better_official_eval(
                        candidate_sync=average_sync_loss,
                        candidate_l1=average_l1_loss,
                        candidate_step=global_step,
                        best_sync=best_off_eval_sync,
                        best_l1=best_off_eval_l1,
                        best_step=best_off_eval_step,
                    ):
                        best_off_eval_sync = average_sync_loss
                        best_off_eval_l1 = average_l1_loss
                        best_off_eval_perceptual = average_perceptual_loss
                        best_off_eval_step = int(global_step)
                        save_named_checkpoint_pair("generator_best_off_eval.pth", "disc_best_off_eval.pth")
                        log(
                            "New best sync eval: "
                            f"sync={best_off_eval_sync:.4f} l1={best_off_eval_l1:.4f} "
                            f"perceptual={best_off_eval_perceptual:.4f} step={best_off_eval_step}"
                        )
                    if better_scalar_metric(
                        candidate_value=average_l1_loss,
                        candidate_step=global_step,
                        best_value=best_l1_eval_l1,
                        best_step=best_l1_eval_step,
                        candidate_secondary=average_perceptual_loss,
                        best_secondary=best_l1_eval_perceptual,
                    ):
                        best_l1_eval_l1 = average_l1_loss
                        best_l1_eval_sync = average_sync_loss
                        best_l1_eval_perceptual = average_perceptual_loss
                        best_l1_eval_step = int(global_step)
                        save_named_checkpoint_pair("generator_best_l1_eval.pth", "disc_best_l1_eval.pth")
                        log(
                            "New best L1 eval: "
                            f"l1={best_l1_eval_l1:.4f} sync={best_l1_eval_sync:.4f} "
                            f"perceptual={best_l1_eval_perceptual:.4f} step={best_l1_eval_step}"
                        )
                    if better_scalar_metric(
                        candidate_value=average_perceptual_loss,
                        candidate_step=global_step,
                        best_value=best_perceptual_eval_perceptual,
                        best_step=best_perceptual_eval_step,
                        candidate_secondary=average_l1_loss,
                        best_secondary=best_perceptual_eval_l1,
                    ):
                        best_perceptual_eval_perceptual = average_perceptual_loss
                        best_perceptual_eval_sync = average_sync_loss
                        best_perceptual_eval_l1 = average_l1_loss
                        best_perceptual_eval_step = int(global_step)
                        save_named_checkpoint_pair(
                            "generator_best_perceptual_eval.pth",
                            "disc_best_perceptual_eval.pth",
                        )
                        log(
                            "New best perceptual eval: "
                            f"perceptual={best_perceptual_eval_perceptual:.4f} sync={best_perceptual_eval_sync:.4f} "
                            f"l1={best_perceptual_eval_l1:.4f} step={best_perceptual_eval_step}"
                        )
                    save_named_checkpoint_pair("generator_latest.pth", "disc_latest.pth")

            prog_bar.set_description(
                "L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}".format(
                    running_l1_loss / (step + 1),
                    running_sync_loss / (step + 1),
                    running_perceptual_loss / (step + 1),
                    running_disc_fake_loss / (step + 1),
                    running_disc_real_loss / (step + 1),
                )
            )

        global_epoch += 1


def main():
    parser = argparse.ArgumentParser(description="Official-style Wav2Lip HQ GAN mirror trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--syncnet", required=True, help="Path to official/mirror SyncNet checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="Resume generator from official-style checkpoint")
    parser.add_argument("--disc-checkpoint-path", default=None, help="Resume quality discriminator from official-style checkpoint")
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated speaker snapshot")
    parser.add_argument("--val-speaker-list", default=None, help="Optional newline-separated validation snapshot")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if int(cfg["model"]["img_size"]) != 96:
        raise SystemExit("generator_mirror_gan only supports model.img_size=96")
    if bool(cfg["model"].get("predict_alpha", False)):
        raise SystemExit("generator_mirror_gan requires model.predict_alpha=false")
    if int(cfg["syncnet"]["T"]) != syncnet_T:
        raise SystemExit(f"generator_mirror_gan requires syncnet.T={syncnet_T}")
    if int(cfg["model"]["mel_steps"]) != syncnet_mel_step_size:
        raise SystemExit(f"generator_mirror_gan requires model.mel_steps={syncnet_mel_step_size}")

    cfg_mirror = resolve_mirror_cfg(cfg)
    device_name = cfg["training"]["device"]
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("generator_mirror_gan requested cuda but CUDA is unavailable")

    roots = get_dataset_roots(cfg)
    log(f"Dataset roots: {roots}")
    speaker_allowlist = load_allowlist(args.speaker_list)
    val_allowlist = load_allowlist(args.val_speaker_list)
    if speaker_allowlist is not None:
        log(f"Loaded speaker snapshot: {len(speaker_allowlist)} entries from {args.speaker_list}")
    if val_allowlist is not None:
        log(f"Loaded val snapshot: {len(val_allowlist)} entries from {args.val_speaker_list}")

    train_dataset = build_dataset(cfg, roots, speaker_allowlist)
    val_dataset = build_dataset(cfg, roots, val_allowlist) if val_allowlist is not None else None

    train_data_loader = build_loader(
        train_dataset,
        cfg_mirror["batch_size"],
        int(cfg["data"]["num_workers"]),
        device.type,
        shuffle=True,
    )
    test_data_loader = (
        build_loader(
            val_dataset,
            cfg_mirror["batch_size"],
            cfg_mirror["val_num_workers"],
            device.type,
            shuffle=False,
        )
        if val_dataset is not None
        else None
    )

    model = OfficialWav2Lip().to(device)
    disc = OfficialWav2LipDiscQual().to(device)
    syncnet, syncnet_kind, sync_epoch, sync_step = load_syncnet_teacher(args.syncnet, device)

    log(
        f"Models: generator={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M, "
        f"disc={sum(p.numel() for p in disc.parameters() if p.requires_grad)/1e6:.1f}M"
    )
    log(f"SyncNet teacher loaded: kind={syncnet_kind}, epoch={sync_epoch}, step={sync_step}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg_mirror["lr"],
        betas=cfg_mirror["betas"],
    )
    disc_optimizer = torch.optim.Adam(
        [p for p in disc.parameters() if p.requires_grad],
        lr=cfg_mirror["disc_lr"],
        betas=cfg_mirror["betas"],
    )

    initial_best_evals = {}
    if args.checkpoint_path is not None:
        initial_best_evals = extract_best_evals_from_checkpoint(
            _load_checkpoint(args.checkpoint_path, torch.cuda.is_available())
        )
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
    if args.disc_checkpoint_path is not None:
        load_checkpoint(
            args.disc_checkpoint_path,
            disc,
            disc_optimizer,
            reset_optimizer=False,
            overwrite_global_states=False,
        )

    checkpoint_dir = os.path.join(cfg["training"]["output_dir"], "generator_mirror_gan")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log(
        "Mirror-GAN config: "
        f"epochs={cfg_mirror['epochs']} batch_size={cfg_mirror['batch_size']} "
        f"lr={cfg_mirror['lr']:.2e} disc_lr={cfg_mirror['disc_lr']:.2e} "
        f"sync_wt_initial={cfg_mirror['syncnet_wt_initial']:.4f} "
        f"sync_wt_target={cfg_mirror['syncnet_wt_target']:.4f} "
        f"disc_wt={cfg_mirror['disc_wt']:.4f} "
        f"save_sample_images={cfg_mirror['save_sample_images']} "
        f"sample_interval_steps={cfg_mirror['sample_interval_steps']}"
    )
    log(
        f"Checkpoint dir: {checkpoint_dir} | train_batches={len(train_data_loader)}"
        + (f" | val_batches={len(test_data_loader)}" if test_data_loader is not None else "")
    )

    train(
        device,
        model,
        disc,
        syncnet,
        train_data_loader,
        test_data_loader,
        optimizer,
        disc_optimizer,
        checkpoint_dir=checkpoint_dir,
        cfg_mirror=cfg_mirror,
        initial_best_evals=initial_best_evals,
    )
    print("Generator mirror GAN training complete!", flush=True)


if __name__ == "__main__":
    main()
