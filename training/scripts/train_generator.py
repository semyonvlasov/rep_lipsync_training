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
from models import LipSyncGenerator, Discriminator, SyncNet as LocalSyncNet
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
    targets = torch.ones((cos_sim.size(0), 1), device=cos_sim.device)
    return F.binary_cross_entropy(cos_sim.unsqueeze(1), targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--syncnet", required=True, help="Path to trained SyncNet checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated list of speaker dirs to include")
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

    log(f"Config: img_size={img_size}, base_ch={cfg['model']['base_channels']}, device={device}")

    # Dataset — supports both precomputed processed roots and lazy canonical
    # faceclip roots (mp4 + json -> cached frames.npy / mel_*.npy on demand).
    roots = get_dataset_roots(cfg)
    log(f"Dataset roots: {roots}")
    speaker_allowlist = None
    if args.speaker_list:
        with open(args.speaker_list) as f:
            speaker_allowlist = [line.strip() for line in f if line.strip()]
        log(f"Loaded speaker snapshot: {len(speaker_allowlist)} entries from {args.speaker_list}")

    dataset = LipSyncDataset(
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
    loader_kwargs = {
        "batch_size": cfg["generator"]["batch_size"],
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
    log(f"DataLoader: {len(loader)} batches, batch_size={cfg['generator']['batch_size']}")

    # Models
    log("Creating generator...")
    generator = LipSyncGenerator(
        img_size=img_size,
        base_channels=cfg["model"]["base_channels"],
        predict_alpha=cfg["model"]["predict_alpha"],
    ).to(device)
    param_count = sum(p.numel() for p in generator.parameters()) / 1e6
    log(f"Generator: {param_count:.1f}M params")

    discriminator = Discriminator().to(device) if loss_cfg.get("gan", 0) > 0 else None
    if discriminator:
        d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
        log(f"Discriminator: {d_params:.1f}M params")

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
    g_opt = torch.optim.Adam(generator.parameters(), lr=cfg["generator"]["lr"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg["generator"]["lr"],
                              betas=(0.5, 0.999)) if discriminator else None

    use_amp = cfg["training"]["mixed_precision"] and device == "cuda"
    if device == "cuda":
        scaler = GradScaler(enabled=use_amp)
    else:
        scaler = None

    log(f"Losses: L1={loss_cfg['l1']}, perc={loss_cfg['perceptual']}, "
        f"sync={loss_cfg['sync']} (warmup={loss_cfg.get('sync_warmup_epochs', 10)}), "
        f"gan={loss_cfg.get('gan', 0)}, alpha_reg={loss_cfg.get('alpha_reg', 0)}")
    if device == "cuda":
        log(
            "CUDA opts: "
            f"tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
            f"persistent_workers={loader_kwargs.get('persistent_workers', False)}, "
            f"prefetch_factor={loader_kwargs.get('prefetch_factor', '-')}"
        )
    log(f"AMP: {use_amp}, Grad clip: {cfg['training']['gradient_clip']}")
    max_batches_per_epoch = cfg["training"].get("max_batches_per_epoch", 0)
    if max_batches_per_epoch:
        log(f"Max batches per epoch: {max_batches_per_epoch}")
    log("=" * 60)

    # LR scheduler
    if cfg["generator"]["lr_scheduler"] == "cosine":
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

    for epoch in range(start_epoch, cfg["generator"]["epochs"]):
        generator.train()
        if discriminator:
            discriminator.train()

        totals = {"l1": 0, "perc": 0, "sync": 0, "sync_reward": 0, "gan_g": 0, "gan_d": 0,
                  "temporal": 0, "alpha": 0, "total": 0}
        t0 = time.time()
        batches_processed = 0
        ema100 = {key: None for key in totals}

        use_sync = epoch >= loss_cfg.get("sync_warmup_epochs", 10)

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
                if use_sync and loss_cfg["sync"] > 0:
                    cos_sim = sync_cosine_score(syncnet, syncnet_kind, mel, indiv_mels, pred_face)
                    sync_reward = cos_sim.mean()
                    sync_loss = official_sync_loss_from_cosine(cos_sim)

                # GAN loss
                gan_g_loss = torch.tensor(0.0, device=device)
                if discriminator and loss_cfg.get("gan", 0) > 0:
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

                sync_wt = loss_cfg["sync"] if use_sync else 0.0
                recon_wt = max(0.0, 1.0 - sync_wt)
                g_loss = (recon_wt * l1) + (sync_wt * sync_loss) + perc + gan_g_loss + alpha_loss

            if scaler:
                scaler.scale(g_loss).backward()
                scaler.unscale_(g_opt)
                nn.utils.clip_grad_norm_(generator.parameters(), cfg["training"]["gradient_clip"])
                scaler.step(g_opt)
                scaler.update()
            else:
                g_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), cfg["training"]["gradient_clip"])
                g_opt.step()

            # ---- Discriminator ----
            d_loss_val = 0
            if discriminator and loss_cfg.get("gan", 0) > 0:
                d_opt.zero_grad(set_to_none=True)
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

            if max_batches_per_epoch and batches_processed >= max_batches_per_epoch:
                log(f"  Reached max_batches_per_epoch={max_batches_per_epoch}, ending epoch early")
                break

        # Epoch stats
        n = max(1, batches_processed)
        elapsed = time.time() - t0
        log(f"Epoch {epoch}/{cfg['generator']['epochs']}: "
            f"L1={totals['l1']/n:.4f} perc={totals['perc']/n:.4f} "
            f"sync={totals['sync']/n:.4f} reward={totals['sync_reward']/n:.4f} "
            f"gan_g={totals['gan_g']/n:.4f} "
            f"gan_d={totals['gan_d']/n:.4f} total={totals['total']/n:.4f} "
            f"({elapsed:.0f}s, {n*cfg['generator']['batch_size']/elapsed:.1f} samples/s) "
            f"lr={g_opt.param_groups[0]['lr']:.6f}"
            f"{' [sync ON]' if use_sync else ' [sync OFF]'}")

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
