#!/usr/bin/env python3
"""
Train LipSync Generator with multi-loss supervision.

Losses:
  - L1 reconstruction (pixel-level)
  - VGG perceptual (feature-level)
  - SyncNet sync (lip-audio alignment) — uses pretrained official SyncNet_color
  - GAN adversarial (realism)
  - Alpha regularization (smooth blending mask)

Usage:
    python scripts/train_generator.py --config configs/default.yaml \
        --syncnet ../models/official_syncnet/checkpoints/lipsync_expert.pth
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tv_models

# Support both cuda and mps
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)

# Import our training models first (before the official reference code pollutes 'models')
sys.path.insert(0, TRAINING_ROOT)
from models import (
    LipSyncGenerator,
    Discriminator,
    OfficialQualityDiscriminator,
    SyncNet as LocalSyncNet,
)
from data import LipSyncDataset
from scripts.dataset_roots import get_dataset_roots

# Import official SyncNet — avoid module name conflicts with our training/models
import importlib.util
_official_models_dir = os.path.join(REPO_ROOT, "models", "official_syncnet", "models")

def _load_official_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load conv first (syncnet depends on it via `from .conv import Conv2d`)
_conv_mod = _load_official_module("official_syncnet_models.conv",
    os.path.join(_official_models_dir, "conv.py"))
# Patch the official syncnet source to use absolute import
with open(os.path.join(_official_models_dir, "syncnet.py")) as f:
    _syncnet_src = f.read()
_syncnet_src = _syncnet_src.replace("from .conv import Conv2d", "")
_syncnet_ns = {"__builtins__": __builtins__}
exec("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n", _syncnet_ns)
_syncnet_ns["Conv2d"] = _conv_mod.Conv2d
exec(_syncnet_src, _syncnet_ns)
LipSyncSyncNet = _syncnet_ns["SyncNet_color"]


class VGGPerceptualLoss(nn.Module):
    """VGG19 feature matching loss."""
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT).features[:16]
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize from [0,1] to ImageNet range
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def log(msg):
    """Timestamped logging."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def flatten_temporal(x):
    """(B, C, T, H, W) -> (B*T, C, H, W); passthrough for 4D tensors."""
    if x.dim() == 5:
        return torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
    return x


def lower_half(x):
    """Crop the lower facial region for both 4D and 5D tensors."""
    if x.dim() == 5:
        return x[:, :, :, x.size(3) // 2:, :]
    return x[:, :, x.size(2) // 2:, :]


def prepare_syncnet_visual(face_sequences):
    """
    Convert generator output to a stacked lower-face representation:
      (B, 3, T, H, W) -> (B, 15, 48, 96)
    This shape works for both the official official SyncNet and the local SyncNet.
    """
    lower = lower_half(face_sequences)
    lower = flatten_temporal(lower)
    lower = F.interpolate(lower, size=(48, 96), mode="bilinear", align_corners=False)

    batch_size = face_sequences.size(0)
    time_steps = face_sequences.size(2)
    lower = torch.stack(torch.split(lower, batch_size, dim=0), dim=2)
    return torch.cat([lower[:, :, i] for i in range(time_steps)], dim=1)


def prepare_local_syncnet_audio(indiv_mels):
    """
    Convert per-frame mel chunks:
      (B, T, 1, 80, 16) -> (B, 1, 80, T*16)
    """
    return torch.cat([indiv_mels[:, t] for t in range(indiv_mels.size(1))], dim=-1)


def load_syncnet_teacher(checkpoint_path, device, syncnet_T):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        model = LocalSyncNet(T=syncnet_T).to(device)
        model.load_state_dict(ck["model"])
        kind = "local"
        epoch = ck.get("epoch", "?")
    else:
        model = LipSyncSyncNet().to(device)
        state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state_dict)
        kind = "official"
        epoch = ck.get("global_epoch", "?") if isinstance(ck, dict) else "?"

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, kind, epoch


def sync_cosine_score(syncnet, syncnet_kind, mel, indiv_mels, face_sequences):
    sync_face = prepare_syncnet_visual(face_sequences)
    if syncnet_kind == "official":
        audio_emb, video_emb = syncnet(mel, sync_face)
    else:
        sync_audio = prepare_local_syncnet_audio(indiv_mels)
        video_emb, audio_emb = syncnet(sync_face, sync_audio)
    return F.cosine_similarity(audio_emb, video_emb)


def official_sync_loss_from_cosine(cos_sim):
    # The reference code applies BCE directly to cosine similarity. Modern
    # PyTorch rejects that path under autocast and expects probability inputs
    # in (0, 1), so we mirror the intent with an autocast-safe compatibility
    # shim rather than silently switching to a different objective.
    targets = torch.ones((cos_sim.size(0), 1), device=cos_sim.device, dtype=torch.float32)
    probs = cos_sim.float().clamp_(1.0e-6, 1.0 - 1.0e-6).unsqueeze(1)
    if cos_sim.device.type == "cuda":
        with torch.amp.autocast("cuda", enabled=False):
            return F.binary_cross_entropy(probs, targets)
    return F.binary_cross_entropy(probs, targets)


def compute_adaptive_sync_weight(avg_sync, start_sync, full_sync, target_sync_wt):
    if target_sync_wt <= 0:
        return 0.0
    if start_sync <= full_sync:
        return target_sync_wt if avg_sync <= full_sync else 0.0
    progress = (start_sync - avg_sync) / (start_sync - full_sync)
    progress = max(0.0, min(1.0, progress))
    return target_sync_wt * progress


def load_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_generator_dataset(cfg, img_size, roots, speaker_allowlist):
    return LipSyncDataset(
        roots=roots,
        img_size=img_size,
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
    )


def build_loader(dataset, cfg, batch_size, device, shuffle):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
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
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def eval_syncnet_alignment(generator, syncnet, syncnet_kind, loader, device, max_batches, use_amp):
    generator.eval()
    sync_losses = []
    recon_losses = []

    for batch_idx, (face_input, indiv_mels, mel, gt) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        non_blocking = device == "cuda"
        face_input = face_input.to(device, non_blocking=non_blocking)
        indiv_mels = indiv_mels.to(device, non_blocking=non_blocking)
        mel = mel.to(device, non_blocking=non_blocking)
        gt = gt.to(device, non_blocking=non_blocking)

        amp_ctx = autocast(enabled=use_amp) if device == "cuda" else nullcontext()
        with amp_ctx:
            pred = generator(indiv_mels, face_input)
            if isinstance(pred, tuple):
                pred_face = pred[0]
            else:
                pred_face = pred

            cos_sim = sync_cosine_score(syncnet, syncnet_kind, mel, indiv_mels, pred_face)
            sync_loss = official_sync_loss_from_cosine(cos_sim)
            recon_loss = F.l1_loss(pred_face, gt)

        sync_losses.append(sync_loss.item())
        recon_losses.append(recon_loss.item())

    generator.train()
    if not sync_losses:
        return None, None
    return sum(sync_losses) / len(sync_losses), sum(recon_losses) / len(recon_losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--syncnet", required=True, help="Path to trained SyncNet checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated list of speaker dirs to include")
    parser.add_argument("--val-speaker-list", default=None, help="Optional newline-separated list of held-out dirs for official-style sync evaluation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"]["device"]
    output_dir = os.path.join(cfg["training"]["output_dir"], "generator")
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    if device == "cuda":
        allow_tf32 = cfg["training"].get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cfg["training"].get("cudnn_benchmark", True)
        matmul_precision = cfg["training"].get("float32_matmul_precision")
        if matmul_precision:
            torch.set_float32_matmul_precision(matmul_precision)

    img_size = cfg["model"]["img_size"]
    loss_cfg = cfg["generator"]["loss"]
    gan_mode = loss_cfg.get("gan_mode", "official_hq")

    log(f"Config: img_size={img_size}, base_ch={cfg['model']['base_channels']}, device={device}")

    # Dataset — supports both precomputed processed roots and lazy canonical
    # faceclip roots (mp4 + json -> cached frames.npy / mel_*.npy on demand).
    roots = get_dataset_roots(cfg)
    log(f"Dataset roots: {roots}")
    speaker_allowlist = load_allowlist(args.speaker_list)
    if speaker_allowlist is not None:
        log(f"Loaded speaker snapshot: {len(speaker_allowlist)} entries from {args.speaker_list}")

    dataset = build_generator_dataset(cfg, img_size, roots, speaker_allowlist)
    loader = build_loader(dataset, cfg, cfg["generator"]["batch_size"], device, shuffle=True)
    log(f"DataLoader: {len(loader)} batches, batch_size={cfg['generator']['batch_size']}")

    val_allowlist = load_allowlist(args.val_speaker_list)
    val_loader = None
    if val_allowlist is not None:
        log(f"Loaded val snapshot: {len(val_allowlist)} entries from {args.val_speaker_list}")
        val_dataset = build_generator_dataset(cfg, img_size, roots, val_allowlist)
        val_loader = build_loader(val_dataset, cfg, cfg["generator"]["batch_size"], device, shuffle=False)
        log(f"ValLoader: {len(val_loader)} batches, batch_size={cfg['generator']['batch_size']}")

    # Models
    log("Creating generator...")
    generator = LipSyncGenerator(
        img_size=img_size,
        base_channels=cfg["model"]["base_channels"],
        predict_alpha=cfg["model"]["predict_alpha"],
    ).to(device)
    param_count = sum(p.numel() for p in generator.parameters()) / 1e6
    log(f"Generator: {param_count:.1f}M params")

    discriminator = None
    if loss_cfg.get("gan", 0) > 0:
        if gan_mode == "official_hq":
            discriminator = OfficialQualityDiscriminator().to(device)
        elif gan_mode == "patch":
            discriminator = Discriminator().to(device)
        else:
            raise ValueError(f"Unsupported gan_mode={gan_mode!r}; expected 'official_hq' or 'patch'")
    if discriminator:
        d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
        log(f"Discriminator: {d_params:.1f}M params (mode={gan_mode})")

    # Load frozen teacher SyncNet: either official official SyncNet or our local SyncNet
    log(f"Loading SyncNet from {args.syncnet}...")
    syncnet, syncnet_kind, sync_epoch = load_syncnet_teacher(
        args.syncnet, device, cfg["syncnet"]["T"]
    )
    log(f"SyncNet loaded (kind={syncnet_kind}, epoch={sync_epoch})")

    # Losses
    vgg_loss = None
    if loss_cfg.get("perceptual", 0) > 0:
        log("Loading VGG19 for perceptual loss...")
        vgg_loss = VGGPerceptualLoss(device).to(device)
        log("VGG19 loaded")
    else:
        log("Perceptual loss disabled")

    # Optimizers
    base_betas = tuple(cfg["generator"].get("betas", (0.9, 0.999)))
    gan_betas = tuple(cfg["generator"].get("gan_betas", (0.5, 0.999)))
    opt_betas = gan_betas if discriminator and gan_mode == "official_hq" else base_betas
    g_opt = torch.optim.Adam(generator.parameters(), lr=cfg["generator"]["lr"], betas=opt_betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg["generator"]["lr"],
                              betas=opt_betas) if discriminator else None

    use_amp = cfg["training"]["mixed_precision"] and device == "cuda"
    if device == "cuda":
        scaler = GradScaler(enabled=use_amp)
    else:
        scaler = None

    log(f"Losses: L1={loss_cfg['l1']}, perc={loss_cfg['perceptual']}, "
        f"sync={loss_cfg['sync']} (warmup={loss_cfg.get('sync_warmup_epochs', 10)}), "
        f"gan={loss_cfg.get('gan', 0)}, alpha_reg={loss_cfg.get('alpha_reg', 0)}")
    log(f"Optimizer: Adam betas={opt_betas}")
    if device == "cuda":
        log(
            "CUDA opts: "
            f"tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
            f"persistent_workers={cfg['data'].get('persistent_workers', False)}, "
            f"prefetch_factor={cfg['data'].get('prefetch_factor', '-')}"
        )
    log(f"AMP: {use_amp}, Grad clip: {cfg['training']['gradient_clip']}")
    max_batches_per_epoch = cfg["training"].get("max_batches_per_epoch", 0)
    epoch_total_batches = min(len(loader), max_batches_per_epoch) if max_batches_per_epoch else len(loader)
    if max_batches_per_epoch:
        log(f"Max batches per epoch: {max_batches_per_epoch}")
    log("=" * 60)

    # LR scheduler
    if cfg["generator"].get("lr_scheduler") == "cosine":
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_opt, T_max=cfg["generator"]["epochs"]
        )
    else:
        g_scheduler = None

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ck["generator"])
        g_opt.load_state_dict(ck["g_optimizer"])
        if discriminator and "discriminator" in ck:
            discriminator.load_state_dict(ck["discriminator"])
            d_opt.load_state_dict(ck["d_optimizer"])
        start_epoch = ck["epoch"] + 1
        log(f"Resumed from epoch {start_epoch}")

    total_batches_remaining = max(0, cfg["generator"]["epochs"] - start_epoch) * epoch_total_batches
    global_step = start_epoch * len(loader)
    effective_sync_wt = loss_cfg.get("sync_initial", loss_cfg["sync"])
    official_sync_schedule = bool(loss_cfg.get("sync_official_schedule", False))
    adaptive_sync_schedule = bool(loss_cfg.get("sync_adaptive_schedule", False))
    sync_eval_interval = int(cfg["generator"].get("eval_interval_steps", 0))
    sync_eval_batches = int(cfg["generator"].get("eval_batches", 700))
    sync_eval_threshold = float(loss_cfg.get("sync_official_threshold", 0.75))
    sync_adaptive_start = float(loss_cfg.get("sync_adaptive_start", 4.5))
    sync_adaptive_full = float(loss_cfg.get("sync_adaptive_full", 2.6))
    if official_sync_schedule:
        log(
            "Official-style sync schedule: "
            f"initial={effective_sync_wt:.4f}, target={loss_cfg['sync']:.4f}, "
            f"threshold={sync_eval_threshold:.3f}, interval={sync_eval_interval}, eval_batches={sync_eval_batches}"
        )
    elif adaptive_sync_schedule:
        log(
            "Adaptive sync schedule: "
            f"initial={effective_sync_wt:.4f}, target={loss_cfg['sync']:.4f}, "
            f"start={sync_adaptive_start:.3f}, full={sync_adaptive_full:.3f}, "
            f"interval={sync_eval_interval}, eval_batches={sync_eval_batches}"
        )
    if (official_sync_schedule or adaptive_sync_schedule) and val_loader is None:
        log(
            "WARNING: sync schedule is enabled but no --val-speaker-list was provided; "
            "eval checkpoints will not run and effective_sync_wt will stay at its initial value."
        )

    sync_warmup_epochs = loss_cfg.get("sync_warmup_epochs", 10)
    training_t0 = time.time()
    for epoch in range(start_epoch, cfg["generator"]["epochs"]):
        generator.train()
        if discriminator:
            discriminator.train()

        totals = {"l1": 0, "perc": 0, "sync": 0, "sync_reward": 0, "gan_g": 0, "gan_d": 0,
                  "temporal": 0, "alpha": 0, "total": 0}
        t0 = time.time()
        batches_processed = 0
        ema100 = {key: None for key in totals}

        for batch_idx, (face_input, indiv_mels, mel, gt) in enumerate(loader):
            non_blocking = device == "cuda"
            face_input = face_input.to(device, non_blocking=non_blocking)
            indiv_mels = indiv_mels.to(device, non_blocking=non_blocking)
            mel = mel.to(device, non_blocking=non_blocking)
            gt = gt.to(device, non_blocking=non_blocking)

            # ---- Generator forward ----
            g_opt.zero_grad(set_to_none=True)

            amp_ctx = autocast(enabled=use_amp) if device == "cuda" else nullcontext()

            with amp_ctx:
                if cfg["model"]["predict_alpha"]:
                    pred_face, pred_alpha = generator(indiv_mels, face_input)
                else:
                    pred_face = generator(indiv_mels, face_input)
                    pred_alpha = None

                pred_lower = lower_half(pred_face)
                gt_lower = lower_half(gt)

                # Match the reference Wav2Lip objective: reconstruct the full
                # face patch, not only the lower half.
                l1 = F.l1_loss(pred_face, gt) * loss_cfg["l1"]

                # Perceptual loss
                perc = torch.tensor(0.0, device=device)
                if vgg_loss is not None and loss_cfg.get("perceptual", 0) > 0:
                    perc = vgg_loss(flatten_temporal(pred_lower), flatten_temporal(gt_lower))
                    perc = perc * loss_cfg["perceptual"]

                # Sync loss (if warmed up)
                # official SyncNet expects:
                #   face: (B, 15, 48, 96) — 5 frames × 3ch, lower half at 96px wide
                #   audio: (B, 1, 80, 16) — mel chunk
                sync_loss = torch.tensor(0.0, device=device)
                sync_reward = torch.tensor(0.0, device=device)
                batch_use_sync = (effective_sync_wt > 0) and epoch >= sync_warmup_epochs
                if batch_use_sync:
                    cos_sim = sync_cosine_score(syncnet, syncnet_kind, mel, indiv_mels, pred_face)
                    sync_reward = cos_sim.mean()
                    sync_loss = official_sync_loss_from_cosine(cos_sim)

                # GAN loss
                gan_g_loss = torch.tensor(0.0, device=device)
                if discriminator and loss_cfg.get("gan", 0) > 0:
                    if gan_mode == "official_hq":
                        gan_g_loss = discriminator.perceptual_forward(pred_face) * loss_cfg["gan"]
                    else:
                        lower_pred_d = flatten_temporal(pred_lower)
                        d_pred = discriminator(lower_pred_d)
                        gan_g_loss = F.binary_cross_entropy_with_logits(
                            d_pred, torch.ones_like(d_pred)
                        ) * loss_cfg["gan"]

                # Alpha regularization
                alpha_loss = torch.tensor(0.0, device=device)
                if pred_alpha is not None and loss_cfg.get("alpha_reg", 0) > 0:
                    tv_h = (pred_alpha[..., 1:, :] - pred_alpha[..., :-1, :]).abs().mean()
                    tv_w = (pred_alpha[..., :, 1:] - pred_alpha[..., :, :-1]).abs().mean()
                    alpha_loss = (tv_h + tv_w) * loss_cfg["alpha_reg"]

                sync_wt = effective_sync_wt if batch_use_sync else 0.0
                recon_wt = max(0.0, 1.0 - sync_wt)
                g_loss = (recon_wt * l1) + (sync_wt * sync_loss) + perc + gan_g_loss + alpha_loss

            if scaler:
                scaler.scale(g_loss).backward()
                scaler.unscale_(g_opt)
                if cfg["training"].get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), cfg["training"]["gradient_clip"])
                scaler.step(g_opt)
                scaler.update()
            else:
                g_loss.backward()
                if cfg["training"].get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), cfg["training"]["gradient_clip"])
                g_opt.step()

            # ---- Discriminator ----
            d_loss_val = 0
            if discriminator and loss_cfg.get("gan", 0) > 0:
                d_opt.zero_grad(set_to_none=True)
                if gan_mode == "official_hq":
                    d_real = discriminator(gt.detach())
                    d_fake = discriminator(pred_face.detach())
                    d_loss = (
                        F.binary_cross_entropy(d_real, torch.ones_like(d_real)) +
                        F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
                    ) * 0.5
                else:
                    lower_gt = flatten_temporal(gt_lower).detach()
                    lower_pred_d = flatten_temporal(pred_lower).detach()

                    d_real = discriminator(lower_gt)
                    d_fake = discriminator(lower_pred_d)

                    d_loss = (
                        F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
                        F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                    ) * 0.5

                if scaler:
                    scaler.scale(d_loss).backward()
                    scaler.step(d_opt)
                    scaler.update()
                else:
                    d_loss.backward()
                    d_opt.step()
                d_loss_val = d_loss.item()

            # Accumulate
            totals["l1"] += l1.item()
            totals["perc"] += perc.item()
            totals["sync"] += sync_loss.item()
            totals["sync_reward"] += sync_reward.item()
            totals["gan_g"] += gan_g_loss.item()
            totals["gan_d"] += d_loss_val
            totals["alpha"] += alpha_loss.item()
            totals["total"] += g_loss.item()
            batches_processed += 1
            global_step += 1

            current_metrics = {
                "l1": l1.item(),
                "perc": perc.item(),
                "sync": sync_loss.item(),
                "sync_reward": sync_reward.item(),
                "gan_g": gan_g_loss.item(),
                "gan_d": d_loss_val,
                "alpha": alpha_loss.item(),
                "total": g_loss.item(),
            }
            for key, value in current_metrics.items():
                ema100[key] = update_ema(ema100[key], value, 100)

            if batch_idx % 60 == 0:
                elapsed_epoch = time.time() - t0
                epoch_eta = compute_remaining_eta(elapsed_epoch, batches_processed, epoch_total_batches)
                elapsed_total = time.time() - training_t0
                completed_total = ((epoch - start_epoch) * epoch_total_batches) + batches_processed
                full_eta = compute_remaining_eta(elapsed_total, completed_total, total_batches_remaining)
                log(f"  E{epoch} [{batch_idx}/{len(loader)}] "
                    f"l1={l1.item():.4f} perc={perc.item():.4f} "
                    f"sync={sync_loss.item():.4f} reward={sync_reward.item():.4f} "
                    f"gan_g={gan_g_loss.item():.4f} "
                    f"total={g_loss.item():.4f}")
                log(
                    "    ema100 "
                    f"l1={ema100['l1']:.4f} perc={ema100['perc']:.4f} "
                    f"sync={ema100['sync']:.4f} reward={ema100['sync_reward']:.4f} "
                    f"gan_g={ema100['gan_g']:.4f} gan_d={ema100['gan_d']:.4f} "
                    f"alpha={ema100['alpha']:.4f} total={ema100['total']:.4f}"
                )
                log(
                    f"    eta epoch={format_eta(epoch_eta)} full={format_eta(full_eta)}"
                )

            if (
                (official_sync_schedule or adaptive_sync_schedule)
                and val_loader is not None
                and sync_eval_interval > 0
                and global_step % sync_eval_interval == 0
            ):
                avg_sync, avg_recon = eval_syncnet_alignment(
                    generator,
                    syncnet,
                    syncnet_kind,
                    val_loader,
                    device,
                    sync_eval_batches,
                    use_amp,
                )
                if avg_sync is not None:
                    log(
                        f"  Eval step {global_step}: val_sync={avg_sync:.4f} "
                        f"val_l1={avg_recon:.4f} active_sync_wt={effective_sync_wt:.4f}"
                    )
                    if official_sync_schedule:
                        if effective_sync_wt < loss_cfg["sync"] and avg_sync < sync_eval_threshold:
                            effective_sync_wt = loss_cfg["sync"]
                            log(
                                f"  Sync weight enabled: effective_sync_wt={effective_sync_wt:.4f} "
                                f"(val_sync={avg_sync:.4f} < {sync_eval_threshold:.4f})"
                            )
                    elif adaptive_sync_schedule:
                        next_sync_wt = compute_adaptive_sync_weight(
                            avg_sync=avg_sync,
                            start_sync=sync_adaptive_start,
                            full_sync=sync_adaptive_full,
                            target_sync_wt=loss_cfg["sync"],
                        )
                        if abs(next_sync_wt - effective_sync_wt) > 1.0e-8:
                            effective_sync_wt = next_sync_wt
                            log(
                                "  Adaptive sync weight update: "
                                f"effective_sync_wt={effective_sync_wt:.4f} "
                                f"(val_sync={avg_sync:.4f}, start={sync_adaptive_start:.4f}, "
                                f"full={sync_adaptive_full:.4f})"
                            )

            if max_batches_per_epoch and batches_processed >= max_batches_per_epoch:
                log(f"  Reached max_batches_per_epoch={max_batches_per_epoch}, ending epoch early")
                break

        # Epoch stats
        n = max(1, batches_processed)
        elapsed = time.time() - t0
        elapsed_total = time.time() - training_t0
        completed_total = ((epoch - start_epoch + 1) * epoch_total_batches)
        full_eta = compute_remaining_eta(elapsed_total, completed_total, total_batches_remaining)
        next_epoch_eta = (elapsed / n) * epoch_total_batches
        epoch_sync_active = (effective_sync_wt > 0) and epoch >= sync_warmup_epochs
        log(f"Epoch {epoch}/{cfg['generator']['epochs']}: "
            f"L1={totals['l1']/n:.4f} perc={totals['perc']/n:.4f} "
            f"sync={totals['sync']/n:.4f} reward={totals['sync_reward']/n:.4f} "
            f"gan_g={totals['gan_g']/n:.4f} "
            f"gan_d={totals['gan_d']/n:.4f} total={totals['total']/n:.4f} "
            f"({elapsed:.0f}s, {n*cfg['generator']['batch_size']/elapsed:.1f} samples/s) "
            f"lr={g_opt.param_groups[0]['lr']:.6f} sync_wt={effective_sync_wt:.4f}"
            f"{' [sync ON]' if epoch_sync_active else ' [sync OFF]'}")
        log(f"  ETA next_epoch={format_eta(next_epoch_eta)} full={format_eta(full_eta)}")

        if g_scheduler:
            g_scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 or epoch == cfg["generator"]["epochs"] - 1:
            ck_path = os.path.join(output_dir, f"generator_epoch{epoch:03d}.pth")
            ck = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "g_optimizer": g_opt.state_dict(),
                "config": cfg,
            }
            if discriminator:
                ck["discriminator"] = discriminator.state_dict()
                ck["d_optimizer"] = d_opt.state_dict()
            torch.save(ck, ck_path)
            print(f"  Saved {ck_path}")

        # Save sample images
        if (epoch + 1) % cfg["training"]["sample_every"] == 0:
            save_samples(generator, loader, device, cfg, sample_dir, epoch)

    print("Generator training complete!")


def save_samples(generator, loader, device, cfg, sample_dir, epoch):
    """Generate and save sample predictions."""
    import cv2
    generator.eval()
    batch = next(iter(loader))
    face_input, indiv_mels, _, gt = [b.to(device) for b in batch]

    with torch.no_grad():
        if cfg["model"]["predict_alpha"]:
            pred, alpha = generator(indiv_mels[:4], face_input[:4])
        else:
            pred = generator(indiv_mels[:4], face_input[:4])
            alpha = None

    for i in range(min(4, pred.shape[0])):
        tiles = []
        for t in range(pred.shape[2]):
            img = (pred[i, :, t].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            gt_img = (gt[i, :, t].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            tiles.append(cv2.hconcat([gt_img, img]))
        combined = cv2.hconcat(tiles)
        cv2.imwrite(os.path.join(sample_dir, f"epoch{epoch:03d}_sample{i}.png"), combined)

        if alpha is not None:
            alpha_tiles = [
                (alpha[i, 0, t].cpu().numpy() * 255).clip(0, 255).astype("uint8")
                for t in range(alpha.shape[2])
            ]
            a_img = cv2.hconcat(alpha_tiles)
            cv2.imwrite(os.path.join(sample_dir, f"epoch{epoch:03d}_alpha{i}.png"), a_img)

    generator.train()


if __name__ == "__main__":
    main()
