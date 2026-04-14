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
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import OrderedDict, deque
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
from config_loader import load_config
from scripts.dataset_roots import get_dataset_roots


global_step = 0
global_epoch = 0

syncnet_T = 5
syncnet_mel_step_size = 16

def log(msg: str) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt5(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.5f}"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


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
        "sync_alignment_min_consensus_ratio": sync_cfg.get("min_consensus_ratio"),
        "sync_alignment_max_shift_mad": sync_cfg.get("max_shift_mad"),
        "sync_alignment_syncnet_checkpoint": sync_cfg.get("syncnet_checkpoint"),
        "sync_alignment_write_manifest": sync_cfg.get("write_manifest", True),
    }


class GeneratorMirrorGanDataset(LipSyncDataset):
    """Match the upstream HQ trainer sampling semantics.

    The official HQ trainer samples videos uniformly and treats one epoch as one
    pass over the video list. Keep frame-weighted sampling only as an explicit
    ablation mode.
    """

    def __init__(
        self,
        *args,
        sampling_mode: str = "uniform",
        deterministic_eval: bool = False,
        deterministic_eval_seed: int = 20260408,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sampling_mode = str(sampling_mode).strip().lower()
        if sampling_mode not in {"frame_weighted", "uniform"}:
            raise ValueError(
                f"Unsupported generator_mirror_gan sampling_mode={sampling_mode!r}; "
                "expected 'frame_weighted' or 'uniform'"
            )
        self.sampling_mode = sampling_mode
        self.deterministic_eval = bool(deterministic_eval)
        self.deterministic_eval_seed = int(deterministic_eval_seed)

    def _stable_u64(self, *parts) -> int:
        digest = hashlib.blake2b(digest_size=8)
        for part in parts:
            digest.update(str(part).encode("utf-8"))
            digest.update(b"\0")
        return int.from_bytes(digest.digest(), byteorder="big", signed=False)

    def _get_deterministic_generator_sample(self, idx):
        if not self.speakers:
            raise RuntimeError("GeneratorMirrorGanDataset has no speakers")

        total_speakers = len(self.speakers)
        for offset in range(total_speakers):
            speaker_idx = (int(idx) + offset) % total_speakers
            speaker_key = self.speakers[speaker_idx]
            try:
                frames, mel_chunks, _ = self._load_speaker(speaker_key)
            except Exception:
                continue

            n_frames = min(len(frames), len(mel_chunks))
            if n_frames <= 3 * self.syncnet_T:
                continue

            max_start = n_frames - self.syncnet_T
            if max_start < 0:
                continue

            start = self._stable_u64(
                self.deterministic_eval_seed,
                "start",
                idx,
                speaker_key,
            ) % (max_start + 1)
            wrong_start = self._stable_u64(
                self.deterministic_eval_seed,
                "wrong",
                idx,
                speaker_key,
            ) % (max_start + 1)
            if wrong_start == start:
                wrong_start = (wrong_start + 1) % (max_start + 1)

            return self._build_generator_sample(frames, mel_chunks, start, wrong_start)

        raise RuntimeError("GeneratorMirrorGanDataset deterministic eval could not produce a valid sample")

    def __len__(self):
        if self.sampling_mode == "uniform":
            return len(self.speakers)
        return max(self._total_frames, len(self.speakers))

    def __getitem__(self, idx):
        if self.deterministic_eval:
            return self._get_deterministic_generator_sample(idx)

        del idx
        if not self.speakers:
            raise RuntimeError("GeneratorMirrorGanDataset has no speakers")

        while True:
            if self.sampling_mode == "uniform":
                speaker_idx = random.randrange(len(self.speakers))
            else:
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


def build_dataset(
    cfg: dict,
    roots: list[str],
    speaker_allowlist,
    *,
    deterministic_eval: bool = False,
):
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
        sampling_mode=cfg.get("generator_mirror_gan", {}).get("sampling_mode", "uniform"),
        deterministic_eval=deterministic_eval,
        deterministic_eval_seed=cfg.get("generator_mirror_gan", {}).get("eval_seed", 20260408),
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


def weighted_eval_score(sync_wt: float, disc_wt: float, l1: float, sync: float, perceptual: float) -> float:
    return (
        sync_wt * float(sync)
        + disc_wt * float(perceptual)
        + (1.0 - sync_wt - disc_wt) * float(l1)
    )


def maybe_float(value):
    if value is None:
        return None
    return float(value)


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
    overall_score=None,
    overall_sync=None,
    overall_l1=None,
    overall_perceptual=None,
    overall_sync_wt=None,
    overall_disc_wt=None,
    overall_step=None,
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
    if overall_step is not None:
        payload["overall"] = {
            "score": float(overall_score),
            "sync": float(overall_sync),
            "l1": float(overall_l1),
            "perceptual": float(overall_perceptual),
            "sync_wt": float(overall_sync_wt),
            "disc_wt": float(overall_disc_wt),
            "step": int(overall_step),
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
    if ck.get("best_overall_eval_step") is not None:
        legacy["overall"] = {
            "score": float(ck["best_overall_eval_score"]),
            "sync": float(ck["best_overall_eval_sync"]),
            "l1": float(ck["best_overall_eval_l1"]),
            "perceptual": float(ck["best_overall_eval_perceptual"]),
            "sync_wt": float(ck["best_overall_eval_sync_wt"]),
            "disc_wt": float(ck["best_overall_eval_disc_wt"]),
            "step": int(ck["best_overall_eval_step"]),
        }
    return legacy


def extract_scheduler_state_from_checkpoint(ck):
    payload = ck.get("mirror_scheduler_state")
    if isinstance(payload, dict):
        extracted = {}
        if payload.get("sync_wt") is not None:
            extracted["sync_wt"] = float(payload["sync_wt"])
        if payload.get("disc_wt") is not None:
            extracted["disc_wt"] = float(payload["disc_wt"])
        if payload.get("guard_best_l1") is not None:
            extracted["guard_best_l1"] = float(payload["guard_best_l1"])
        if payload.get("sync_gate_ever_open") is not None:
            extracted["sync_gate_ever_open"] = bool(payload["sync_gate_ever_open"])
        if payload.get("freeze_disc_updates_until_clean_streak") is not None:
            extracted["freeze_disc_updates_until_clean_streak"] = bool(
                payload["freeze_disc_updates_until_clean_streak"]
            )
        elif payload.get("freeze_disc_updates_until_regular_eval") is not None:
            extracted["freeze_disc_updates_until_clean_streak"] = bool(
                payload["freeze_disc_updates_until_regular_eval"]
            )
        if payload.get("freeze_disc_clean_streak") is not None:
            extracted["freeze_disc_clean_streak"] = int(payload["freeze_disc_clean_streak"])
        return extracted

    legacy = {}
    if ck.get("current_sync_wt") is not None:
        legacy["sync_wt"] = float(ck["current_sync_wt"])
    if ck.get("current_disc_wt") is not None:
        legacy["disc_wt"] = float(ck["current_disc_wt"])
    return legacy


def build_checkpoint_payload(
    model,
    optimizer,
    step: int,
    epoch: int,
    save_optimizer_state: bool,
    best_evals=None,
    scheduler_state=None,
):
    optimizer_state = optimizer.state_dict() if (optimizer is not None and save_optimizer_state) else None
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": int(step),
        "global_epoch": int(epoch),
    }
    if scheduler_state:
        checkpoint["mirror_scheduler_state"] = dict(scheduler_state)
        if scheduler_state.get("sync_wt") is not None:
            checkpoint["current_sync_wt"] = float(scheduler_state["sync_wt"])
        if scheduler_state.get("disc_wt") is not None:
            checkpoint["current_disc_wt"] = float(scheduler_state["disc_wt"])
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
        overall_eval = best_evals.get("overall")
        if overall_eval is not None:
            checkpoint["best_overall_eval_score"] = float(overall_eval["score"])
            checkpoint["best_overall_eval_sync"] = float(overall_eval["sync"])
            checkpoint["best_overall_eval_l1"] = float(overall_eval["l1"])
            checkpoint["best_overall_eval_perceptual"] = float(overall_eval["perceptual"])
            checkpoint["best_overall_eval_sync_wt"] = float(overall_eval["sync_wt"])
            checkpoint["best_overall_eval_disc_wt"] = float(overall_eval["disc_wt"])
            checkpoint["best_overall_eval_step"] = int(overall_eval["step"])
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
    scheduler_state=None,
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
            scheduler_state=scheduler_state,
        ),
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)


def slim_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    slim = OrderedDict()
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.half()
        slim[key] = tensor
    return slim


def build_recovery_checkpoint_payload(
    model,
    step: int,
    epoch: int,
    scheduler_state=None,
    eval_metrics=None,
    eval_weights=None,
):
    payload = {
        "state_dict": slim_state_dict(model.state_dict()),
        "global_step": int(step),
        "global_epoch": int(epoch),
        "checkpoint_kind": "mirror_gan_recovery_fp16",
    }
    if scheduler_state:
        payload["mirror_scheduler_state"] = dict(scheduler_state)
    if eval_metrics is not None:
        payload["eval_metrics"] = dict(eval_metrics)
    if eval_weights is not None:
        payload["eval_weights"] = dict(eval_weights)
    return payload


def save_recovery_checkpoint(
    model,
    step: int,
    checkpoint_path: str,
    epoch: int,
    scheduler_state=None,
    eval_metrics=None,
    eval_weights=None,
):
    torch.save(
        build_recovery_checkpoint_payload(
            model,
            step=step,
            epoch=epoch,
            scheduler_state=scheduler_state,
            eval_metrics=eval_metrics,
            eval_weights=eval_weights,
        ),
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)


def resolve_training_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((Path(TRAINING_ROOT) / path).resolve())


def resolve_benchmark_cfg(cfg: dict) -> dict:
    section = dict(cfg.get("benchmark") or {})
    faces = [resolve_training_path(item) for item in section.get("faces", [])]
    script = str(section.get("script", "run_official_wav2lip_benchmark.py")).strip()
    if not script:
        script = "run_official_wav2lip_benchmark.py"
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = Path(TRAINING_ROOT) / "scripts" / script_path.name
    return {
        "enabled": bool(section.get("enabled", False)),
        "script": str(script_path),
        "audio": resolve_training_path(section.get("audio")),
        "faces": [item for item in faces if item],
        "device": str(section.get("device", "cuda")),
        "detector_device": str(section.get("detector_device", "cuda")),
        "landmarker_device": str(section.get("landmarker_device", "cpu")),
        "batch_size": int(section.get("batch_size", 16)),
        "face_det_batch_size": int(section.get("face_det_batch_size", 4)),
        "s3fd_path": resolve_training_path(section.get("s3fd_path")),
        "cache_root": resolve_training_path(section.get("cache_root")),
        "no_cache": bool(section.get("no_cache", False)),
    }


def resolve_checkpoint_publish_cfg(cfg: dict) -> dict:
    section = dict(cfg.get("checkpoint_publish") or {})
    output_dir = Path(cfg["training"]["output_dir"]).name
    return {
        "enabled": bool(section.get("enabled", False)),
        "remote": str(section.get("remote", "gdrive:")),
        "drive_root_folder_id": str(section.get("drive_root_folder_id", "1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv")),
        "remote_dir_prefix": str(section.get("remote_dir_prefix", "checkpoints")).strip("/"),
        "experiment_name": str(section.get("experiment_name", output_dir)),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_output_name(face_path: str, audio_path: str, checkpoint_path: str) -> str:
    face_stem = Path(face_path).stem
    audio_stem = Path(audio_path).stem
    ckpt_stem = Path(checkpoint_path).stem
    return f"{face_stem}_{audio_stem}_{ckpt_stem}.mp4"


def run_logged(cmd: list[str], prefix: str) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"{prefix} {line}")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_checkpoint_benchmark(
    checkpoint_path: str,
    artifact_dir: Path,
    benchmark_cfg: dict,
) -> list[dict]:
    if not benchmark_cfg["enabled"]:
        return []
    if not benchmark_cfg["audio"] or not benchmark_cfg["faces"]:
        log("Checkpoint benchmark disabled for this step: benchmark audio/faces not configured")
        return []

    benchmark_dir = artifact_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_script = Path(benchmark_cfg["script"])
    benchmark_is_tilt_aware = benchmark_script.name == "run_tilt_aware_x96_benchmark.py"
    outputs = []
    for face_path in benchmark_cfg["faces"]:
        out_path = benchmark_dir / benchmark_output_name(face_path, benchmark_cfg["audio"], checkpoint_path)
        cmd = [
            sys.executable,
            str(benchmark_script),
            "--face",
            str(face_path),
            "--audio",
            str(benchmark_cfg["audio"]),
            "--checkpoint",
            str(checkpoint_path),
            "--outfile",
            str(out_path),
            "--device",
            benchmark_cfg["device"],
            "--batch_size",
            str(benchmark_cfg["batch_size"]),
        ]
        if benchmark_is_tilt_aware:
            cmd.extend(["--landmarker_device", benchmark_cfg["landmarker_device"]])
        else:
            cmd.extend(
                [
                    "--detector_device",
                    benchmark_cfg["detector_device"],
                    "--face_det_batch_size",
                    str(benchmark_cfg["face_det_batch_size"]),
                ]
            )
        if benchmark_cfg["s3fd_path"] and not benchmark_is_tilt_aware:
            cmd.extend(["--s3fd_path", str(benchmark_cfg["s3fd_path"])])
        if benchmark_cfg["cache_root"]:
            cmd.extend(["--cache_root", str(benchmark_cfg["cache_root"])])
        if benchmark_cfg["no_cache"]:
            cmd.append("--no_cache")
        run_logged(cmd, prefix="[CheckpointBench]")
        outputs.append(
            {
                "face": str(face_path),
                "audio": str(benchmark_cfg["audio"]),
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
            }
        )
    return outputs


def create_artifact_tar(artifact_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(artifact_dir, arcname=artifact_dir.name)


def spawn_background_uploader(
    *,
    tar_path: Path,
    artifact_dir: Path,
    upload_cfg: dict,
    upload_log_path: Path,
) -> None:
    upload_log_path.parent.mkdir(parents=True, exist_ok=True)
    uploader_script = Path(TRAINING_ROOT) / "scripts" / "upload_checkpoint_bundle.py"
    cmd = [
        sys.executable,
        str(uploader_script),
        "--tar-path",
        str(tar_path),
        "--artifact-dir",
        str(artifact_dir),
        "--remote",
        upload_cfg["remote"],
        "--drive-root-folder-id",
        upload_cfg["drive_root_folder_id"],
        "--remote-path",
        f"{upload_cfg['remote_dir_prefix']}/{upload_cfg['experiment_name']}/{tar_path.name}",
    ]
    with open(upload_log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"{timestamp()} [UploadSpawn] cmd={json.dumps(cmd)}\n")
        log_handle.flush()
        subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


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


def eval_model(
    test_data_loader,
    device,
    model,
    disc,
    syncnet,
    cfg_mirror: dict,
    eval_step: int,
    sample_checkpoint_dir: str | None = None,
) -> dict[str, float]:
    eval_steps = int(cfg_mirror["eval_steps"])
    print(f"Evaluating for {eval_steps} steps at global_step={eval_step}", flush=True)
    running_sync_loss = []
    running_l1_loss = []
    running_disc_real_loss = []
    running_disc_fake_loss = []
    running_perceptual_loss = []
    recon_loss = nn.L1Loss()
    saved_samples = False

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
            if sample_checkpoint_dir is not None and not saved_samples:
                save_sample_images(x, g, gt, eval_step, sample_checkpoint_dir)
                saved_samples = True

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
            f"Eval@{eval_step} | "
            f"L1: {fmt5(metrics['l1'])}, "
            f"Sync: {fmt5(metrics['sync'])}, "
            f"Percep: {fmt5(metrics['perceptual'])} | "
            f"Fake: {fmt5(metrics['disc_fake'])}, "
            f"Real: {fmt5(metrics['disc_real'])}",
            flush=True,
        )
        return metrics


def eval_is_healthy(metrics: dict[str, float], cfg_mirror: dict) -> bool:
    status = disc_band_status(
        fake_value=float(metrics["disc_fake"]),
        real_value=float(metrics["disc_real"]),
        fake_min=cfg_mirror["healthy_fake_min"],
        fake_max=cfg_mirror["healthy_fake_max"],
        real_min=cfg_mirror["healthy_real_min"],
        real_max=cfg_mirror["healthy_real_max"],
        max_gap=cfg_mirror["healthy_max_fake_real_gap"],
    )
    return bool(status["ok"])


def disc_band_status(
    *,
    fake_value: float,
    real_value: float,
    fake_min: float | None,
    fake_max: float | None,
    real_min: float | None,
    real_max: float | None,
    max_gap: float | None,
) -> dict[str, object]:
    reasons: list[str] = []
    if fake_min is not None and fake_value < fake_min:
        reasons.append(f"fake<{fmt5(fake_min)}")
    if fake_max is not None and fake_value > fake_max:
        reasons.append(f"fake>{fmt5(fake_max)}")
    if real_min is not None and real_value < real_min:
        reasons.append(f"real<{fmt5(real_min)}")
    if real_max is not None and real_value > real_max:
        reasons.append(f"real>{fmt5(real_max)}")
    gap = abs(fake_value - real_value)
    if max_gap is not None and gap > max_gap:
        reasons.append(f"gap>{fmt5(max_gap)}")
    return {
        "ok": not reasons,
        "gap": float(gap),
        "reasons": reasons,
    }


def emergency_disc_status(
    *,
    fake_value: float,
    real_value: float,
    fake_update_off_max: float,
    fake_step_down_min: float,
    fake_minus_real_step_down_min: float,
) -> dict[str, object]:
    fake_minus_real = fake_value - real_value
    freeze_disc_updates = fake_value < fake_update_off_max
    emergency_disc_down = fake_value > fake_step_down_min and fake_minus_real > fake_minus_real_step_down_min

    reasons: list[str] = []
    if freeze_disc_updates:
        reasons.append(f"fake<{fmt5(fake_update_off_max)}")
    if emergency_disc_down:
        reasons.append(f"fake>{fmt5(fake_step_down_min)}")
        reasons.append(f"fake-real>{fmt5(fake_minus_real_step_down_min)}")

    return {
        "ok": not reasons,
        "fake_minus_real": float(fake_minus_real),
        "freeze_disc_updates": freeze_disc_updates,
        "disc_down": emergency_disc_down,
        "reasons": reasons,
    }


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
        "eval_seed": int(section.get("eval_seed", 20260408)),
        "train_log_interval_steps": int(section.get("train_log_interval_steps", 50)),
        "save_optimizer_state": bool(section.get("save_optimizer_state", True)),
        "adaptive_loss_schedule": bool(section.get("adaptive_loss_schedule", False)),
        "l1_only_warmup_steps": int(section.get("l1_only_warmup_steps", 0)),
        "adaptive_weight_freeze_step": int(section.get("adaptive_weight_freeze_step", -1)),
        "adaptive_weight_freeze_sync_wt": maybe_float(section.get("adaptive_weight_freeze_sync_wt")),
        "adaptive_weight_freeze_disc_wt": maybe_float(section.get("adaptive_weight_freeze_disc_wt")),
        "syncnet_wt_initial": float(section.get("syncnet_wt_initial", 0.0)),
        "syncnet_wt_target": float(section.get("syncnet_wt_target", 0.03)),
        "syncnet_enable_threshold": float(section.get("syncnet_enable_threshold", 0.90)),
        "sync_wt_min": float(section.get("sync_wt_min", 0.0005)),
        "sync_wt_step_up": float(section.get("sync_wt_step_up", 0.005)),
        "sync_wt_step_down": float(section.get("sync_wt_step_down", 0.01)),
        "disc_wt": float(section.get("disc_wt", 0.07)),
        "disc_wt_initial": float(section.get("disc_wt_initial", section.get("disc_wt", 0.07))),
        "min_disc_wt": float(section.get("min_disc_wt", 0.0)),
        "disc_wt_step_up": float(section.get("disc_wt_step_up", 0.005)),
        "disc_wt_step_down": float(section.get("disc_wt_step_down", 0.01)),
        "disc_up_fake_max": float(section.get("disc_up_fake_max", 0.8)),
        "disc_hold_abs_gap": float(
            section.get("disc_hold_abs_gap", section.get("disc_hold_fake_minus_real_gap", 0.5))
        ),
        "max_sync_wt": float(section.get("max_sync_wt", section.get("syncnet_wt_target", 0.03))),
        "max_disc_wt": float(section.get("max_disc_wt", section.get("disc_wt", 0.07))),
        "max_total_wt": float(section.get("max_total_wt", 0.35)),
        "sync_l1_guard_max_ratio": float(section.get("sync_l1_guard_max_ratio", section.get("l1_guard_max_ratio", 1.15))),
        "disc_l1_guard_max_ratio": float(section.get("disc_l1_guard_max_ratio", section.get("l1_guard_max_ratio", 1.15))),
        "disc_guard_max_gap": float(section.get("disc_guard_max_gap", section.get("healthy_max_fake_real_gap", 0.15))),
        "emergency_eval_enabled": bool(section.get("emergency_eval_enabled", True)),
        "emergency_eval_step_turn_on": int(section.get("emergency_eval_step_turn_on", 5000)),
        "emergency_window_batches": int(section.get("emergency_window_batches", 40)),
        "emergency_disc_update_off_fake_max": float(section.get("emergency_disc_update_off_fake_max", 0.3)),
        "emergency_disc_unfreeze_clean_streak": int(section.get("emergency_disc_unfreeze_clean_streak", 10)),
        "emergency_disc_down_fake_min": float(section.get("emergency_disc_down_fake_min", 0.9)),
        "emergency_disc_down_fake_minus_real_gap": float(
            section.get("emergency_disc_down_fake_minus_real_gap", 0.5)
        ),
        "emergency_disc_step_down": float(section.get("emergency_disc_step_down", 0.002)),
        "val_num_workers": int(section.get("val_num_workers", 4)),
        "save_last_healthy": bool(section.get("save_last_healthy", False)),
        "healthy_fake_min": maybe_float(section.get("healthy_fake_min")),
        "healthy_fake_max": maybe_float(section.get("healthy_fake_max")),
        "healthy_real_min": maybe_float(section.get("healthy_real_min")),
        "healthy_real_max": maybe_float(section.get("healthy_real_max")),
        "healthy_max_fake_real_gap": maybe_float(section.get("healthy_max_fake_real_gap", 0.15)),
    }
    if resolved["adaptive_loss_schedule"]:
        if resolved["syncnet_wt_initial"] + resolved["disc_wt_initial"] >= 1.0:
            raise ValueError("adaptive generator_mirror_gan requires syncnet_wt_initial + disc_wt_initial < 1.0")
        if resolved["max_total_wt"] >= 1.0:
            raise ValueError("adaptive generator_mirror_gan requires max_total_wt < 1.0")
        if resolved["sync_l1_guard_max_ratio"] < 1.0:
            raise ValueError("adaptive generator_mirror_gan requires sync_l1_guard_max_ratio >= 1.0")
        if resolved["disc_l1_guard_max_ratio"] < 1.0:
            raise ValueError("adaptive generator_mirror_gan requires disc_l1_guard_max_ratio >= 1.0")
        if resolved["sync_wt_min"] < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires sync_wt_min >= 0.0")
        if resolved["sync_wt_min"] > resolved["max_sync_wt"]:
            raise ValueError("adaptive generator_mirror_gan requires sync_wt_min <= max_sync_wt")
        if resolved["min_disc_wt"] < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires min_disc_wt >= 0.0")
        if resolved["min_disc_wt"] > resolved["max_disc_wt"]:
            raise ValueError("adaptive generator_mirror_gan requires min_disc_wt <= max_disc_wt")
        if resolved["emergency_eval_step_turn_on"] < 0:
            raise ValueError("adaptive generator_mirror_gan requires emergency_eval_step_turn_on >= 0")
        if resolved["l1_only_warmup_steps"] < 0:
            raise ValueError("adaptive generator_mirror_gan requires l1_only_warmup_steps >= 0")
        if resolved["adaptive_weight_freeze_step"] < -1:
            raise ValueError("adaptive generator_mirror_gan requires adaptive_weight_freeze_step >= -1")
        if resolved["emergency_window_batches"] <= 0:
            raise ValueError("adaptive generator_mirror_gan requires emergency_window_batches > 0")
        if resolved["emergency_disc_unfreeze_clean_streak"] <= 0:
            raise ValueError("adaptive generator_mirror_gan requires emergency_disc_unfreeze_clean_streak > 0")
        if resolved["emergency_disc_step_down"] < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires emergency_disc_step_down >= 0.0")
        if resolved["disc_hold_abs_gap"] < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires disc_hold_abs_gap >= 0.0")
        if resolved["emergency_disc_down_fake_minus_real_gap"] < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires emergency_disc_down_fake_minus_real_gap >= 0.0")
        freeze_sync_wt = resolved["adaptive_weight_freeze_sync_wt"]
        freeze_disc_wt = resolved["adaptive_weight_freeze_disc_wt"]
        if freeze_sync_wt is not None and freeze_sync_wt < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires adaptive_weight_freeze_sync_wt >= 0.0")
        if freeze_disc_wt is not None and freeze_disc_wt < 0.0:
            raise ValueError("adaptive generator_mirror_gan requires adaptive_weight_freeze_disc_wt >= 0.0")
        if freeze_sync_wt is not None and freeze_sync_wt > resolved["max_sync_wt"]:
            raise ValueError("adaptive generator_mirror_gan requires adaptive_weight_freeze_sync_wt <= max_sync_wt")
        if freeze_disc_wt is not None and freeze_disc_wt > resolved["max_disc_wt"]:
            raise ValueError("adaptive generator_mirror_gan requires adaptive_weight_freeze_disc_wt <= max_disc_wt")
        if freeze_sync_wt is not None and freeze_disc_wt is not None:
            if freeze_sync_wt + freeze_disc_wt > resolved["max_total_wt"]:
                raise ValueError(
                    "adaptive generator_mirror_gan requires adaptive_weight_freeze_sync_wt + "
                    "adaptive_weight_freeze_disc_wt <= max_total_wt"
                )
    else:
        if resolved["syncnet_wt_initial"] + resolved["disc_wt"] >= 1.0:
            raise ValueError("generator_mirror_gan requires syncnet_wt_initial + disc_wt < 1.0")
        if resolved["syncnet_wt_target"] + resolved["disc_wt"] >= 1.0:
            raise ValueError("generator_mirror_gan requires syncnet_wt_target + disc_wt < 1.0")
    if resolved["train_log_interval_steps"] < 0:
        raise ValueError("generator_mirror_gan requires train_log_interval_steps >= 0")
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
    initial_scheduler_state=None,
):
    global global_step
    global global_epoch
    restored_scheduler_state = initial_scheduler_state or {}

    def adaptive_weight_freeze_active(step: int) -> bool:
        freeze_step = int(cfg_mirror["adaptive_weight_freeze_step"])
        return freeze_step >= 0 and step >= freeze_step

    def resolve_frozen_weights(current_sync_wt: float, current_disc_wt: float) -> tuple[float, float]:
        frozen_sync_wt = cfg_mirror["adaptive_weight_freeze_sync_wt"]
        frozen_disc_wt = cfg_mirror["adaptive_weight_freeze_disc_wt"]
        if frozen_sync_wt is None:
            frozen_sync_wt = current_sync_wt
        if frozen_disc_wt is None:
            frozen_disc_wt = current_disc_wt
        return float(frozen_sync_wt), float(frozen_disc_wt)

    if cfg_mirror["adaptive_loss_schedule"]:
        syncnet_wt = float(restored_scheduler_state.get("sync_wt", cfg_mirror["syncnet_wt_initial"]))
        disc_wt = float(restored_scheduler_state.get("disc_wt", cfg_mirror["disc_wt_initial"]))
        if disc_wt > 0.0 and disc_wt < cfg_mirror["min_disc_wt"]:
            disc_wt = float(cfg_mirror["min_disc_wt"])
        sync_gate_ever_open = bool(restored_scheduler_state.get("sync_gate_ever_open", syncnet_wt > 0.0))
        freeze_disc_updates_until_clean_streak = bool(
            restored_scheduler_state.get("freeze_disc_updates_until_clean_streak", False)
        )
        freeze_disc_clean_streak = int(restored_scheduler_state.get("freeze_disc_clean_streak", 0))
        if sync_gate_ever_open and syncnet_wt < cfg_mirror["sync_wt_min"]:
            syncnet_wt = float(cfg_mirror["sync_wt_min"])
        if global_step < cfg_mirror["l1_only_warmup_steps"]:
            syncnet_wt = 0.0
            disc_wt = 0.0
            sync_gate_ever_open = False
            freeze_disc_updates_until_clean_streak = False
            freeze_disc_clean_streak = 0
        elif adaptive_weight_freeze_active(global_step):
            syncnet_wt, disc_wt = resolve_frozen_weights(syncnet_wt, disc_wt)
            if syncnet_wt > 0.0:
                sync_gate_ever_open = True
    else:
        syncnet_wt = float(restored_scheduler_state.get("sync_wt", cfg_mirror["syncnet_wt_initial"]))
        disc_wt = float(cfg_mirror["disc_wt"])
        sync_gate_ever_open = False
        freeze_disc_updates_until_clean_streak = False
        freeze_disc_clean_streak = 0
    guard_best_l1 = maybe_float(restored_scheduler_state.get("guard_best_l1"))
    resumed_step = global_step
    recon_loss = nn.L1Loss()

    latest_ckpt_path = os.path.join(checkpoint_dir, "generator_latest.pth")
    disc_latest_ckpt_path = os.path.join(checkpoint_dir, "disc_latest.pth")
    last_healthy_ckpt_path = os.path.join(checkpoint_dir, "generator_last_healthy.pth")
    disc_last_healthy_ckpt_path = os.path.join(checkpoint_dir, "disc_last_healthy.pth")
    artifact_root = Path(checkpoint_dir) / "checkpoint_artifacts"
    upload_logs_dir = Path(checkpoint_dir) / "upload_logs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    upload_logs_dir.mkdir(parents=True, exist_ok=True)

    def current_scheduler_state() -> dict:
        payload = {
            "sync_wt": float(syncnet_wt),
            "disc_wt": float(disc_wt),
            "sync_gate_ever_open": bool(sync_gate_ever_open),
            "freeze_disc_updates_until_clean_streak": bool(freeze_disc_updates_until_clean_streak),
            "freeze_disc_clean_streak": int(freeze_disc_clean_streak),
        }
        if guard_best_l1 is not None:
            payload["guard_best_l1"] = float(guard_best_l1)
        return payload

    def save_recovery_pair(
        generator_path: str,
        discriminator_path: str,
        eval_metrics: dict | None,
        eval_weights: dict | None,
    ) -> None:
        scheduler_state = current_scheduler_state()
        save_recovery_checkpoint(
            model,
            step=global_step,
            checkpoint_path=generator_path,
            epoch=global_epoch,
            scheduler_state=scheduler_state,
            eval_metrics=eval_metrics,
            eval_weights=eval_weights,
        )
        save_recovery_checkpoint(
            disc,
            step=global_step,
            checkpoint_path=discriminator_path,
            epoch=global_epoch,
            scheduler_state=scheduler_state,
            eval_metrics=eval_metrics,
            eval_weights=eval_weights,
        )

    def save_checkpoint_artifacts(
        *,
        step: int,
        eval_metrics: dict | None,
        measured_weights: dict | None,
        benchmark_cfg: dict,
        upload_cfg: dict,
    ) -> None:
        artifact_dir = artifact_root / f"step{step:09d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        generator_path = artifact_dir / f"generator_step{step:09d}.pth"
        disc_path = artifact_dir / f"disc_step{step:09d}.pth"
        eval_metrics_payload = None if eval_metrics is None else dict(eval_metrics)
        eval_weights_payload = None if measured_weights is None else dict(measured_weights)

        try:
            save_recovery_checkpoint(
                model,
                step=step,
                checkpoint_path=str(generator_path),
                epoch=global_epoch,
                scheduler_state=current_scheduler_state(),
                eval_metrics=eval_metrics_payload,
                eval_weights=eval_weights_payload,
            )
            save_recovery_checkpoint(
                disc,
                step=step,
                checkpoint_path=str(disc_path),
                epoch=global_epoch,
                scheduler_state=current_scheduler_state(),
                eval_metrics=eval_metrics_payload,
                eval_weights=eval_weights_payload,
            )
            shutil.copy2(generator_path, latest_ckpt_path)
            shutil.copy2(disc_path, disc_latest_ckpt_path)

            benchmark_outputs = run_checkpoint_benchmark(str(generator_path), artifact_dir, benchmark_cfg)
            manifest = {
                "created_at": timestamp(),
                "step": int(step),
                "epoch": int(global_epoch),
                "generator_checkpoint": str(generator_path),
                "disc_checkpoint": str(disc_path),
                "eval_metrics": eval_metrics_payload,
                "eval_weights": eval_weights_payload,
                "scheduler_state": current_scheduler_state(),
                "benchmark_outputs": benchmark_outputs,
            }
            write_json(artifact_dir / "checkpoint_manifest.json", manifest)

            tar_path = artifact_root / f"step{step:09d}.tar"
            create_artifact_tar(artifact_dir, tar_path)
            log(f"Packaged checkpoint artifact: {tar_path}")

            if not upload_cfg["enabled"]:
                log(f"Checkpoint upload disabled; kept local artifact {tar_path}")
                return

            upload_log_path = upload_logs_dir / f"step{step:09d}.log"
            spawn_background_uploader(
                tar_path=tar_path,
                artifact_dir=artifact_dir,
                upload_cfg=upload_cfg,
                upload_log_path=upload_log_path,
            )
            log(f"Spawned background uploader for step {step}: log={upload_log_path}")
        except Exception as exc:  # noqa: BLE001
            log(
                f"Checkpoint artifact pipeline failed at step {step}: {exc}. "
                f"Kept local artifacts in {artifact_dir}"
            )

    benchmark_cfg = cfg_mirror["benchmark_cfg"]
    upload_cfg = cfg_mirror["checkpoint_publish_cfg"]
    emergency_fake_window = deque(maxlen=cfg_mirror["emergency_window_batches"])
    emergency_real_window = deque(maxlen=cfg_mirror["emergency_window_batches"])
    regular_eval_interval = int(cfg_mirror["eval_interval_steps"])
    last_regular_eval_step = (
        (global_step // regular_eval_interval) * regular_eval_interval
        if regular_eval_interval > 0
        else global_step
    )
    emergency_eval_used_since_regular = False

    log(f"Named latest checkpoint path: {latest_ckpt_path}")
    log(f"Named latest discriminator path: {disc_latest_ckpt_path}")
    log(f"Named last healthy checkpoint path: {last_healthy_ckpt_path}")
    log(f"Named last healthy discriminator path: {disc_last_healthy_ckpt_path}")
    log(f"Artifact root: {artifact_root}")

    def format_train_status_line(epoch_step: int) -> str:
        denom = epoch_step + 1
        return (
            f"GStep: {global_step} | "
            f"SWt: {fmt5(syncnet_wt)}, "
            f"DWt: {fmt5(disc_wt)} | "
            f"L1: {fmt5(running_l1_loss / denom)}, "
            f"Sync: {fmt5(running_sync_loss / denom)}, "
            f"Percep: {fmt5(running_perceptual_loss / denom)} | "
            f"Fake: {fmt5(running_disc_fake_loss / denom)}, "
            f"Real: {fmt5(running_disc_real_loss / denom)}"
        )

    while global_epoch < cfg_mirror["epochs"]:
        print(f"Starting Epoch: {global_epoch}", flush=True)
        running_sync_loss, running_l1_loss, running_perceptual_loss = 0.0, 0.0, 0.0
        running_disc_real_loss, running_disc_fake_loss = 0.0, 0.0

        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            disc.train()
            model.train()
            l1_only_warmup_active = (
                cfg_mirror["adaptive_loss_schedule"]
                and global_step < cfg_mirror["l1_only_warmup_steps"]
            )
            effective_sync_wt = 0.0 if l1_only_warmup_active else syncnet_wt
            effective_disc_wt = 0.0 if l1_only_warmup_active else disc_wt

            x = x.to(device, non_blocking=device.type == "cuda")
            mel = mel.to(device, non_blocking=device.type == "cuda")
            indiv_mels = indiv_mels.to(device, non_blocking=device.type == "cuda")
            gt = gt.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)

            if effective_sync_wt > 0.0:
                sync_loss = get_sync_loss(syncnet, mel, g, device)
            else:
                sync_loss = torch.tensor(0.0, device=device)

            if effective_disc_wt > 0.0:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = torch.tensor(0.0, device=device)

            l1loss = recon_loss(g, gt)
            loss = (
                effective_sync_wt * sync_loss
                + effective_disc_wt * perceptual_loss
                + (1.0 - effective_sync_wt - effective_disc_wt) * l1loss
            )

            loss.backward()
            optimizer.step()

            disc_optimizer.zero_grad()

            if l1_only_warmup_active:
                disc_real_loss = torch.tensor(0.0, device=device)
                disc_fake_loss = torch.tensor(0.0, device=device)
            else:
                pred = disc(gt)
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1), device=device))

                pred = disc(g.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1), device=device))
                if not freeze_disc_updates_until_clean_streak:
                    disc_real_loss.backward()
                    disc_fake_loss.backward()
                    disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()
            if not l1_only_warmup_active:
                emergency_fake_window.append(disc_fake_loss.item())
                emergency_real_window.append(disc_real_loss.item())
            if (
                cfg_mirror["adaptive_loss_schedule"]
                and cfg_mirror["emergency_eval_enabled"]
                and freeze_disc_updates_until_clean_streak
                and not l1_only_warmup_active
                and len(emergency_fake_window) == cfg_mirror["emergency_window_batches"]
            ):
                freeze_window_fake_mean = sum(emergency_fake_window) / len(emergency_fake_window)
                freeze_window_real_mean = sum(emergency_real_window) / len(emergency_real_window)
                freeze_window_status = emergency_disc_status(
                    fake_value=freeze_window_fake_mean,
                    real_value=freeze_window_real_mean,
                    fake_update_off_max=cfg_mirror["emergency_disc_update_off_fake_max"],
                    fake_step_down_min=cfg_mirror["emergency_disc_down_fake_min"],
                    fake_minus_real_step_down_min=cfg_mirror["emergency_disc_down_fake_minus_real_gap"],
                )
                if freeze_window_status["ok"]:
                    freeze_disc_clean_streak += 1
                    if freeze_disc_clean_streak >= cfg_mirror["emergency_disc_unfreeze_clean_streak"]:
                        freeze_disc_updates_until_clean_streak = False
                        freeze_disc_clean_streak = 0
                        log(
                            "Emergency recovery: "
                            f"step={global_step + 1} mean_fake={fmt5(freeze_window_fake_mean)} "
                            f"mean_real={fmt5(freeze_window_real_mean)} "
                            f"clean_streak={cfg_mirror['emergency_disc_unfreeze_clean_streak']}/"
                            f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} "
                            "-> disc_updates=on"
                        )
                else:
                    freeze_disc_clean_streak = 0
            else:
                freeze_disc_clean_streak = 0

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
            running_sync_loss += sync_loss.item() if effective_sync_wt > 0.0 else 0.0
            running_perceptual_loss += perceptual_loss.item() if effective_disc_wt > 0.0 else 0.0

            checkpoint_due = (
                cfg_mirror["checkpoint_interval_steps"] > 0
                and global_step % cfg_mirror["checkpoint_interval_steps"] == 0
            )
            regular_eval_due = (
                test_data_loader is not None
                and cfg_mirror["eval_interval_steps"] > 0
                and global_step % cfg_mirror["eval_interval_steps"] == 0
            )
            emergency_eval_due = False
            emergency_tripwire_status = None
            if (
                test_data_loader is not None
                and cfg_mirror["adaptive_loss_schedule"]
                and cfg_mirror["emergency_eval_enabled"]
                and regular_eval_interval > 0
                and not regular_eval_due
                and global_step >= cfg_mirror["emergency_eval_step_turn_on"]
                and not emergency_eval_used_since_regular
                and len(emergency_fake_window) == cfg_mirror["emergency_window_batches"]
            ):
                steps_since_regular_eval = global_step - last_regular_eval_step
                if (
                    steps_since_regular_eval >= cfg_mirror["emergency_window_batches"]
                    and steps_since_regular_eval < regular_eval_interval
                ):
                    emergency_fake_mean = sum(emergency_fake_window) / len(emergency_fake_window)
                    emergency_real_mean = sum(emergency_real_window) / len(emergency_real_window)
                    emergency_tripwire_status = emergency_disc_status(
                        fake_value=emergency_fake_mean,
                        real_value=emergency_real_mean,
                        fake_update_off_max=cfg_mirror["emergency_disc_update_off_fake_max"],
                        fake_step_down_min=cfg_mirror["emergency_disc_down_fake_min"],
                        fake_minus_real_step_down_min=cfg_mirror["emergency_disc_down_fake_minus_real_gap"],
                    )
                    if not emergency_tripwire_status["ok"]:
                        emergency_eval_due = True
                        emergency_eval_used_since_regular = True
                        log(
                            "Emergency tripwire: "
                            f"step={global_step} mean_fake={fmt5(emergency_fake_mean)} "
                            f"mean_real={fmt5(emergency_real_mean)} "
                            f"fake-real={fmt5(emergency_tripwire_status['fake_minus_real'])} "
                            f"reasons={','.join(emergency_tripwire_status['reasons'])} "
                            "-> scheduling emergency eval"
                        )
            eval_due = regular_eval_due or emergency_eval_due
            need_eval = eval_due or (checkpoint_due and test_data_loader is not None)

            if need_eval:
                with torch.no_grad():
                    eval_kind = "regular"
                    if emergency_eval_due:
                        eval_kind = "emergency"
                    elif checkpoint_due and not regular_eval_due:
                        eval_kind = "checkpoint"
                    eval_cfg = dict(cfg_mirror)
                    eval_cfg["disc_wt"] = float(disc_wt)
                    sample_checkpoint_dir = (
                        str(artifact_root / f"step{global_step:09d}")
                        if checkpoint_due and eval_kind != "emergency"
                        else None
                    )
                    eval_metrics = eval_model(
                        test_data_loader,
                        device,
                        model,
                        disc,
                        syncnet,
                        eval_cfg,
                        global_step,
                        sample_checkpoint_dir=sample_checkpoint_dir,
                    )
                    average_sync_loss = float(eval_metrics["sync"])
                    average_l1_loss = float(eval_metrics["l1"])
                    average_perceptual_loss = float(eval_metrics["perceptual"])
                    average_disc_fake = float(eval_metrics["disc_fake"])
                    average_disc_real = float(eval_metrics["disc_real"])
                    measured_sync_wt = float(syncnet_wt)
                    measured_disc_wt = float(disc_wt)
                    current_overall_score = weighted_eval_score(
                        sync_wt=measured_sync_wt,
                        disc_wt=measured_disc_wt,
                        l1=average_l1_loss,
                        sync=average_sync_loss,
                        perceptual=average_perceptual_loss,
                    )
                    log(
                        f"Eval summary step={global_step} kind={eval_kind} overall={fmt5(current_overall_score)} "
                        f"(sync_wt={fmt5(measured_sync_wt)}, disc_wt={fmt5(measured_disc_wt)}) | "
                        f"l1={fmt5(average_l1_loss)} sync={fmt5(average_sync_loss)} "
                        f"perceptual={fmt5(average_perceptual_loss)} "
                        f"fake={fmt5(average_disc_fake)} real={fmt5(average_disc_real)}"
                    )

                    if guard_best_l1 is None or average_l1_loss < guard_best_l1:
                        guard_best_l1 = average_l1_loss

                    scheduler_log = None
                    next_sync_wt = syncnet_wt
                    next_disc_wt = disc_wt
                    if cfg_mirror["adaptive_loss_schedule"]:
                        l1_only_warmup_active = global_step < cfg_mirror["l1_only_warmup_steps"]
                        adaptive_weights_frozen = adaptive_weight_freeze_active(global_step)
                        sync_l1_limit = float(guard_best_l1) * cfg_mirror["sync_l1_guard_max_ratio"]
                        disc_l1_limit = float(guard_best_l1) * cfg_mirror["disc_l1_guard_max_ratio"]
                        sync_l1_guard_breached = average_l1_loss > sync_l1_limit
                        disc_l1_guard_breached = average_l1_loss > disc_l1_limit
                        fake_minus_real = average_disc_fake - average_disc_real
                        if average_sync_loss < cfg_mirror["syncnet_enable_threshold"]:
                            sync_gate_ever_open = True
                        sync_gate_open = bool(sync_gate_ever_open)
                        old_sync_wt = next_sync_wt
                        old_disc_wt = next_disc_wt
                        actions = []

                        if l1_only_warmup_active:
                            next_sync_wt = 0.0
                            next_disc_wt = 0.0
                            sync_gate_ever_open = False
                            freeze_disc_updates_until_clean_streak = False
                            freeze_disc_clean_streak = 0
                            scheduler_log = (
                                "Adaptive scheduler: "
                                f"guard_best_l1={fmt5(guard_best_l1)} "
                                f"sync_l1_limit={fmt5(sync_l1_limit)} disc_l1_limit={fmt5(disc_l1_limit)} "
                                f"fake-real={fmt5(fake_minus_real)} "
                                f"warmup_l1_only={global_step}/{cfg_mirror['l1_only_warmup_steps']} | "
                                f"sync_wt {fmt5(old_sync_wt)}->{fmt5(next_sync_wt)} "
                                f"disc_wt {fmt5(old_disc_wt)}->{fmt5(next_disc_wt)} "
                                " | disc_updates=off clean_streak=0/"
                                f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} actions=warmup_hold"
                            )
                        elif adaptive_weights_frozen:
                            next_sync_wt, next_disc_wt = resolve_frozen_weights(next_sync_wt, next_disc_wt)
                            scheduler_log = (
                                "Adaptive scheduler: "
                                f"guard_best_l1={fmt5(guard_best_l1)} "
                                f"sync_l1_limit={fmt5(sync_l1_limit)} disc_l1_limit={fmt5(disc_l1_limit)} "
                                f"fake-real={fmt5(fake_minus_real)} "
                                f"weights_frozen_after={cfg_mirror['adaptive_weight_freeze_step']} | "
                                f"sync_wt {fmt5(old_sync_wt)}->{fmt5(next_sync_wt)} "
                                f"disc_wt {fmt5(old_disc_wt)}->{fmt5(next_disc_wt)}"
                                + (
                                    f" | disc_updates={'off' if freeze_disc_updates_until_clean_streak else 'on'} "
                                    f"clean_streak={freeze_disc_clean_streak}/"
                                    f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} actions=weights_frozen"
                                )
                            )
                        elif eval_kind == "emergency":
                            emergency_eval_status = emergency_disc_status(
                                fake_value=average_disc_fake,
                                real_value=average_disc_real,
                                fake_update_off_max=cfg_mirror["emergency_disc_update_off_fake_max"],
                                fake_step_down_min=cfg_mirror["emergency_disc_down_fake_min"],
                                fake_minus_real_step_down_min=cfg_mirror["emergency_disc_down_fake_minus_real_gap"],
                            )
                            emergency_confirmed = not emergency_eval_status["ok"]
                            if emergency_confirmed:
                                if emergency_eval_status["freeze_disc_updates"]:
                                    freeze_disc_updates_until_clean_streak = True
                                    freeze_disc_clean_streak = 0
                                    actions.append("emergency_disc_update_off")
                                elif emergency_eval_status["disc_down"]:
                                    next_disc_wt = max(
                                        cfg_mirror["min_disc_wt"],
                                        next_disc_wt - cfg_mirror["emergency_disc_step_down"],
                                    )
                                    actions.append(f"emergency_disc_down->{fmt5(next_disc_wt)}")
                            scheduler_log = (
                                "Emergency scheduler: "
                                f"guard_best_l1={fmt5(guard_best_l1)} "
                                f"sync_l1_limit={fmt5(sync_l1_limit)} disc_l1_limit={fmt5(disc_l1_limit)} "
                                f"fake-real={fmt5(fake_minus_real)} "
                                f"tripwire_fake-real="
                                f"{fmt5(emergency_tripwire_status['fake_minus_real']) if emergency_tripwire_status is not None else 'None'} "
                                f"tripwire_reasons={','.join(emergency_tripwire_status['reasons']) if emergency_tripwire_status is not None else 'none'} "
                                f"eval_reasons={','.join(emergency_eval_status['reasons']) if emergency_eval_status['reasons'] else 'none'} | "
                                f"sync_wt {fmt5(old_sync_wt)}->{fmt5(next_sync_wt)} "
                                f"disc_wt {fmt5(old_disc_wt)}->{fmt5(next_disc_wt)}"
                                + (
                                    f" | disc_updates={'off' if freeze_disc_updates_until_clean_streak else 'on'} "
                                    f"clean_streak={freeze_disc_clean_streak}/"
                                    f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} "
                                    f"actions={','.join(actions)}"
                                    if actions
                                    else (
                                        f" | disc_updates={'off' if freeze_disc_updates_until_clean_streak else 'on'} "
                                        f"clean_streak={freeze_disc_clean_streak}/"
                                        f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} actions=hold"
                                    )
                                )
                            )
                        else:
                            if sync_l1_guard_breached and next_sync_wt > 0.0:
                                next_sync_wt = max(0.0, next_sync_wt - cfg_mirror["sync_wt_step_down"])
                                actions.append(f"sync_down->{fmt5(next_sync_wt)}")
                            elif sync_gate_open:
                                proposed_sync_wt = next_sync_wt + cfg_mirror["sync_wt_step_up"]
                                if (
                                    proposed_sync_wt <= cfg_mirror["max_sync_wt"]
                                    and (proposed_sync_wt + next_disc_wt) <= cfg_mirror["max_total_wt"]
                                ):
                                    next_sync_wt = proposed_sync_wt
                                    actions.append(f"sync_up->{fmt5(next_sync_wt)}")

                            if sync_gate_open and next_sync_wt < cfg_mirror["sync_wt_min"]:
                                next_sync_wt = float(cfg_mirror["sync_wt_min"])
                                actions.append(f"sync_floor->{fmt5(next_sync_wt)}")

                            disc_hold_due_to_gap = abs(fake_minus_real) > cfg_mirror["disc_hold_abs_gap"]
                            disc_can_up = (
                                not disc_l1_guard_breached
                                and average_disc_fake < cfg_mirror["disc_up_fake_max"]
                                and not disc_hold_due_to_gap
                            )
                            if disc_can_up:
                                proposed_disc_wt = max(
                                    cfg_mirror["min_disc_wt"],
                                    next_disc_wt + cfg_mirror["disc_wt_step_up"],
                                )
                                if (
                                    proposed_disc_wt <= cfg_mirror["max_disc_wt"]
                                    and (next_sync_wt + proposed_disc_wt) <= cfg_mirror["max_total_wt"]
                                ):
                                    next_disc_wt = proposed_disc_wt
                                    actions.append(f"disc_up->{fmt5(next_disc_wt)}")
                            elif disc_l1_guard_breached or average_disc_fake >= cfg_mirror["disc_up_fake_max"]:
                                next_disc_wt = max(
                                    cfg_mirror["min_disc_wt"],
                                    next_disc_wt - cfg_mirror["disc_wt_step_down"],
                                )
                                actions.append(f"disc_down->{fmt5(next_disc_wt)}")
                            elif disc_hold_due_to_gap:
                                actions.append("disc_hold_abs_gap")

                            scheduler_log = (
                                "Adaptive scheduler: "
                                f"guard_best_l1={fmt5(guard_best_l1)} "
                                f"sync_l1_limit={fmt5(sync_l1_limit)} disc_l1_limit={fmt5(disc_l1_limit)} "
                                f"fake-real={fmt5(fake_minus_real)} sync_gate={'open' if sync_gate_open else 'closed'} | "
                                f"sync_wt {fmt5(old_sync_wt)}->{fmt5(next_sync_wt)} "
                                f"disc_wt {fmt5(old_disc_wt)}->{fmt5(next_disc_wt)}"
                                + (
                                    f" | disc_updates={'off' if freeze_disc_updates_until_clean_streak else 'on'} "
                                    f"clean_streak={freeze_disc_clean_streak}/"
                                    f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} "
                                    f"actions={','.join(actions)}"
                                    if actions
                                    else (
                                        f" | disc_updates={'off' if freeze_disc_updates_until_clean_streak else 'on'} "
                                        f"clean_streak={freeze_disc_clean_streak}/"
                                        f"{cfg_mirror['emergency_disc_unfreeze_clean_streak']} actions=hold"
                                    )
                                )
                            )
                    elif average_sync_loss < cfg_mirror["syncnet_enable_threshold"]:
                        next_sync_wt = cfg_mirror["syncnet_wt_target"]

                    syncnet_wt = next_sync_wt
                    disc_wt = next_disc_wt
                    if scheduler_log is not None:
                        log(scheduler_log)
                    if regular_eval_due:
                        last_regular_eval_step = global_step
                        emergency_eval_used_since_regular = False

                    measured_weights = {
                        "sync_wt": measured_sync_wt,
                        "disc_wt": measured_disc_wt,
                        "overall": current_overall_score,
                    }

                    if cfg_mirror["save_last_healthy"] and eval_is_healthy(eval_metrics, cfg_mirror):
                        save_recovery_pair(
                            last_healthy_ckpt_path,
                            disc_last_healthy_ckpt_path,
                            eval_metrics=eval_metrics,
                            eval_weights=measured_weights,
                        )
                        log(
                            "Updated last healthy checkpoint: "
                            f"step={global_step} fake={fmt5(average_disc_fake)} real={fmt5(average_disc_real)} "
                            f"sync={fmt5(average_sync_loss)} l1={fmt5(average_l1_loss)} "
                            f"perceptual={fmt5(average_perceptual_loss)}"
                        )

                    if checkpoint_due:
                        save_checkpoint_artifacts(
                            step=global_step,
                            eval_metrics=eval_metrics,
                            measured_weights=measured_weights,
                            benchmark_cfg=benchmark_cfg,
                            upload_cfg=upload_cfg,
                        )
            elif checkpoint_due:
                save_checkpoint_artifacts(
                    step=global_step,
                    eval_metrics=None,
                    measured_weights=None,
                    benchmark_cfg=benchmark_cfg,
                    upload_cfg=upload_cfg,
                )

            train_status_line = format_train_status_line(step)
            prog_bar.set_description(train_status_line)
            if cfg_mirror["train_log_interval_steps"] > 0 and (
                global_step == 1 or global_step % cfg_mirror["train_log_interval_steps"] == 0
            ):
                log(f"Train status step={global_step} | {train_status_line}")

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

    cfg = load_config(args.config)

    if int(cfg["model"]["img_size"]) != 96:
        raise SystemExit("generator_mirror_gan only supports model.img_size=96")
    if bool(cfg["model"].get("predict_alpha", False)):
        raise SystemExit("generator_mirror_gan requires model.predict_alpha=false")
    if int(cfg["syncnet"]["T"]) != syncnet_T:
        raise SystemExit(f"generator_mirror_gan requires syncnet.T={syncnet_T}")
    if int(cfg["model"]["mel_steps"]) != syncnet_mel_step_size:
        raise SystemExit(f"generator_mirror_gan requires model.mel_steps={syncnet_mel_step_size}")

    cfg_mirror = resolve_mirror_cfg(cfg)
    cfg_mirror["benchmark_cfg"] = resolve_benchmark_cfg(cfg)
    cfg_mirror["checkpoint_publish_cfg"] = resolve_checkpoint_publish_cfg(cfg)
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
    val_dataset = (
        build_dataset(cfg, roots, val_allowlist, deterministic_eval=True)
        if val_allowlist is not None
        else None
    )

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

    initial_scheduler_state = {}
    if args.checkpoint_path is not None:
        generator_checkpoint = _load_checkpoint(args.checkpoint_path, torch.cuda.is_available())
        initial_scheduler_state = extract_scheduler_state_from_checkpoint(generator_checkpoint)
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

    base_log = (
        "Mirror-GAN config: "
        f"epochs={cfg_mirror['epochs']} batch_size={cfg_mirror['batch_size']} "
        f"lr={cfg_mirror['lr']:.2e} disc_lr={cfg_mirror['disc_lr']:.2e} "
        f"save_sample_images={cfg_mirror['save_sample_images']} "
        f"sample_interval_steps={cfg_mirror['sample_interval_steps']}"
    )
    if cfg_mirror["adaptive_loss_schedule"]:
        log(
            base_log
            + " "
            + f"sync_wt_initial={cfg_mirror['syncnet_wt_initial']:.4f} "
            + f"disc_wt_initial={cfg_mirror['disc_wt_initial']:.4f}"
        )
        log(
            "Adaptive weights: "
            f"warmup_l1_only_steps={cfg_mirror['l1_only_warmup_steps']} "
            f"freeze_after_step={cfg_mirror['adaptive_weight_freeze_step']} "
            f"freeze_sync_wt={fmt5(cfg_mirror['adaptive_weight_freeze_sync_wt']) if cfg_mirror['adaptive_weight_freeze_sync_wt'] is not None else 'auto'} "
            f"freeze_disc_wt={fmt5(cfg_mirror['adaptive_weight_freeze_disc_wt']) if cfg_mirror['adaptive_weight_freeze_disc_wt'] is not None else 'auto'} "
            f"sync_min={cfg_mirror['sync_wt_min']:.4f} "
            f"sync_up={cfg_mirror['sync_wt_step_up']:.4f} sync_down={cfg_mirror['sync_wt_step_down']:.4f} "
            f"disc_up={cfg_mirror['disc_wt_step_up']:.4f} disc_down={cfg_mirror['disc_wt_step_down']:.4f} "
            f"min_disc={cfg_mirror['min_disc_wt']:.4f} "
            f"max_sync={cfg_mirror['max_sync_wt']:.4f} max_disc={cfg_mirror['max_disc_wt']:.4f} "
            f"max_total={cfg_mirror['max_total_wt']:.4f} "
            f"sync_l1_guard={cfg_mirror['sync_l1_guard_max_ratio']:.4f} "
            f"disc_l1_guard={cfg_mirror['disc_l1_guard_max_ratio']:.4f} "
            f"disc_up_fake<{cfg_mirror['disc_up_fake_max']:.4f} "
            f"disc_hold_if_abs(fake-real)>{cfg_mirror['disc_hold_abs_gap']:.4f}"
        )
        log(
            "Emergency eval: "
            f"enabled={cfg_mirror['emergency_eval_enabled']} "
            f"step_turn_on={cfg_mirror['emergency_eval_step_turn_on']} "
            f"window={cfg_mirror['emergency_window_batches']} "
            f"disc_update_off_if_fake<{fmt5(cfg_mirror['emergency_disc_update_off_fake_max'])} "
            f"unfreeze_clean_streak={cfg_mirror['emergency_disc_unfreeze_clean_streak']} "
            f"disc_down_if_fake>{fmt5(cfg_mirror['emergency_disc_down_fake_min'])} "
            f"and_fake-real>{fmt5(cfg_mirror['emergency_disc_down_fake_minus_real_gap'])} "
            f"disc_down={fmt5(cfg_mirror['emergency_disc_step_down'])}"
        )
    else:
        log(
            base_log
            + " "
            + f"sync_wt_initial={cfg_mirror['syncnet_wt_initial']:.4f} "
            + f"sync_wt_target={cfg_mirror['syncnet_wt_target']:.4f} "
            + f"disc_wt={cfg_mirror['disc_wt']:.4f}"
        )
    log(
        "Checkpoint publish: "
        f"enabled={cfg_mirror['checkpoint_publish_cfg']['enabled']} "
        f"remote={cfg_mirror['checkpoint_publish_cfg']['remote']} "
        f"experiment={cfg_mirror['checkpoint_publish_cfg']['experiment_name']}"
    )
    log(
        "Benchmark publish: "
        f"enabled={cfg_mirror['benchmark_cfg']['enabled']} "
        f"faces={len(cfg_mirror['benchmark_cfg']['faces'])} "
        f"audio={cfg_mirror['benchmark_cfg']['audio']}"
    )
    log(
        "Eval sampling: "
        f"deterministic={'yes' if val_dataset is not None else 'n/a'} "
        f"eval_seed={cfg_mirror['eval_seed']}"
    )
    log(
        f"Sampling mode: {train_dataset.sampling_mode} | train_samples={len(train_dataset.speakers)} "
        f"| epoch_items={len(train_dataset)}"
    )
    if train_dataset.sampling_mode != "uniform":
        log(
            "WARNING: frame_weighted sampling diverges from official HQ Wav2Lip "
            "and makes epoch boundaries non-comparable to upstream"
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
        initial_scheduler_state=initial_scheduler_state,
    )
    print("Generator mirror GAN training complete!", flush=True)


if __name__ == "__main__":
    main()
