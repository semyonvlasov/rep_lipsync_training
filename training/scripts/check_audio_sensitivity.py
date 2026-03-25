#!/usr/bin/env python3
"""
Sanity checks for audio dependence of the temporal lip-sync generator.

Checks:
  1. Same face + different mel => generated mouth changes, and SyncNet prefers
     the matching audio over the swapped audio.
  2. GT + correct audio > GT + wrong audio according to the official official SyncNet.
  3. Same mel + different reference => mouth changes less than the upper face.
"""

import argparse
import importlib.util
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)

sys.path.insert(0, TRAINING_ROOT)
from data import LipSyncDataset
from models import LipSyncGenerator, SyncNet as LocalSyncNet
from scripts.dataset_roots import get_dataset_roots


def flatten_temporal(x):
    if x.dim() == 5:
        return torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
    return x


def lower_half(x):
    if x.dim() == 5:
        return x[:, :, :, x.size(3) // 2:, :]
    return x[:, :, x.size(2) // 2:, :]


def upper_half(x):
    if x.dim() == 5:
        return x[:, :, :, :x.size(3) // 2, :]
    return x[:, :, :x.size(2) // 2, :]


def prepare_syncnet_visual(face_sequences):
    lower = lower_half(face_sequences)
    lower = flatten_temporal(lower)
    lower = F.interpolate(lower, size=(48, 96), mode="bilinear", align_corners=False)

    batch_size = face_sequences.size(0)
    time_steps = face_sequences.size(2)
    lower = torch.stack(torch.split(lower, batch_size, dim=0), dim=2)
    return torch.cat([lower[:, :, i] for i in range(time_steps)], dim=1)


def prepare_local_syncnet_audio(indiv_mels):
    return torch.cat([indiv_mels[:, t] for t in range(indiv_mels.size(1))], dim=-1)


def load_official_syncnet():
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


def load_config(checkpoint_path, config_path=None):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    return ck, cfg


def load_generator(checkpoint_path, cfg, device):
    model = LipSyncGenerator(
        img_size=cfg["model"]["img_size"],
        base_channels=cfg["model"]["base_channels"],
        predict_alpha=cfg["model"]["predict_alpha"],
    ).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["generator"])
    model.eval()
    return model


def load_syncnet(checkpoint_path, device, syncnet_T):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        teacher_cfg = ck.get("config", {})
        teacher_syncnet_T = int(teacher_cfg.get("syncnet", {}).get("T", syncnet_T))
        teacher_model_cfg = teacher_cfg.get("model", {})
        model = LocalSyncNet(
            T=teacher_syncnet_T,
            audio_temporal_kernels=teacher_model_cfg.get("audio_temporal_kernels"),
        ).to(device)
        model.load_state_dict(ck["model"])
        kind = "local"
    else:
        SyncNet = load_official_syncnet()
        model = SyncNet().to(device)
        state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state_dict)
        kind = "official"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, kind


def sync_cosine_score(syncnet, syncnet_kind, mel, indiv_mels, face_sequences):
    sync_face = prepare_syncnet_visual(face_sequences)
    if syncnet_kind == "official":
        audio_emb, video_emb = syncnet(mel, sync_face)
    else:
        sync_audio = prepare_local_syncnet_audio(indiv_mels)
        video_emb, audio_emb = syncnet(sync_face, sync_audio)
    return F.cosine_similarity(audio_emb, video_emb)


def build_dataset(cfg, speaker_allowlist=None):
    roots = get_dataset_roots(cfg)
    return LipSyncDataset(
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
        min_samples_per_speaker=cfg["data"].get("min_samples_per_speaker", 100),
    )


def sample_window(dataset, speaker_idx=None, exclude_speaker=None):
    T = dataset.syncnet_T
    candidates = list(range(len(dataset.speakers)))
    if exclude_speaker is not None and len(candidates) > 1:
        candidates = [idx for idx in candidates if idx != exclude_speaker]

    for _ in range(32):
        chosen_speaker = random.choice(candidates) if speaker_idx is None else speaker_idx
        frames, mel_chunks, _ = dataset._load_speaker(dataset.speakers[chosen_speaker])
        n_frames = min(len(frames), len(mel_chunks))
        if n_frames < 3 * T:
            if speaker_idx is not None:
                break
            continue
        start = random.randint(0, n_frames - T)
        ref_start = dataset._pick_reference_start(n_frames, start, T)
        sample = dataset.make_generator_sample(chosen_speaker, start, ref_start)
        return sample, chosen_speaker, start, ref_start

    raise RuntimeError("Could not sample a valid temporal window from the dataset")


def run_generator(generator, indiv_mels, face_input):
    with torch.inference_mode():
        pred = generator(indiv_mels, face_input)
        if isinstance(pred, tuple):
            pred = pred[0]
    return pred


def save_sequence_pair(path, left, right):
    tiles = []
    for t in range(left.shape[2]):
        left_img = (left[0, :, t].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        right_img = (right[0, :, t].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        tiles.append(cv2.hconcat([left_img, right_img]))
    cv2.imwrite(path, cv2.hconcat(tiles))


def check_same_face_different_mel(generator, syncnet, syncnet_kind, dataset, device, samples, out_dir):
    mouth_deltas = []
    correct_scores = []
    swapped_scores = []
    swapped_match_scores = []

    preview_saved = False
    for _ in range(samples):
        (face_input_a, indiv_mels_a, mel_a, _), speaker_a, _, _ = sample_window(dataset)
        ( _, indiv_mels_b, mel_b, _), _, _, _ = sample_window(dataset, exclude_speaker=speaker_a)

        face_input = face_input_a.unsqueeze(0).to(device)
        indiv_mels_a = indiv_mels_a.unsqueeze(0).to(device)
        indiv_mels_b = indiv_mels_b.unsqueeze(0).to(device)
        mel_a = mel_a.unsqueeze(0).to(device)
        mel_b = mel_b.unsqueeze(0).to(device)

        pred_a = run_generator(generator, indiv_mels_a, face_input)
        pred_b = run_generator(generator, indiv_mels_b, face_input)

        mouth_deltas.append(F.l1_loss(lower_half(pred_a), lower_half(pred_b)).item())
        correct_scores.append(sync_cosine_score(syncnet, syncnet_kind, mel_a, indiv_mels_a, pred_a).mean().item())
        swapped_scores.append(sync_cosine_score(syncnet, syncnet_kind, mel_a, indiv_mels_a, pred_b).mean().item())
        swapped_match_scores.append(sync_cosine_score(syncnet, syncnet_kind, mel_b, indiv_mels_b, pred_b).mean().item())

        if out_dir and not preview_saved:
            save_sequence_pair(os.path.join(out_dir, "check1_same_face_diff_mel.png"), pred_a, pred_b)
            preview_saved = True

    return {
        "name": "same_face_different_mel",
        "pass": (np.mean(mouth_deltas) > 0.01 and np.mean(correct_scores) > np.mean(swapped_scores)),
        "mouth_delta": float(np.mean(mouth_deltas)),
        "correct_sync": float(np.mean(correct_scores)),
        "swapped_under_original_audio": float(np.mean(swapped_scores)),
        "swapped_under_swapped_audio": float(np.mean(swapped_match_scores)),
    }


def check_gt_correct_audio(syncnet, syncnet_kind, dataset, device, samples):
    correct_scores = []
    wrong_scores = []
    for _ in range(samples):
        (_, indiv_mels_a, mel_a, gt_a), speaker_a, _, _ = sample_window(dataset)
        ((_, indiv_mels_b, mel_b, _), _, _, _) = sample_window(dataset, exclude_speaker=speaker_a)
        indiv_mels_a = indiv_mels_a.unsqueeze(0).to(device)
        indiv_mels_b = indiv_mels_b.unsqueeze(0).to(device)
        mel_a = mel_a.unsqueeze(0).to(device)
        mel_b = mel_b.unsqueeze(0).to(device)
        gt_a = gt_a.unsqueeze(0).to(device)

        correct_scores.append(sync_cosine_score(syncnet, syncnet_kind, mel_a, indiv_mels_a, gt_a).mean().item())
        wrong_scores.append(sync_cosine_score(syncnet, syncnet_kind, mel_b, indiv_mels_b, gt_a).mean().item())

    return {
        "name": "gt_correct_audio_beats_wrong",
        "pass": np.mean(correct_scores) > np.mean(wrong_scores),
        "correct_sync": float(np.mean(correct_scores)),
        "wrong_sync": float(np.mean(wrong_scores)),
        "margin": float(np.mean(correct_scores) - np.mean(wrong_scores)),
    }


def check_same_mel_different_ref(generator, dataset, device, samples, out_dir):
    mouth_deltas = []
    upper_deltas = []
    preview_saved = False

    for _ in range(samples):
        speaker_idx = random.randrange(len(dataset.speakers))
        frames, mel_chunks, _ = dataset._load_speaker(dataset.speakers[speaker_idx])
        n_frames = min(len(frames), len(mel_chunks))
        T = dataset.syncnet_T
        if n_frames < 4 * T:
            continue

        start = random.randint(0, n_frames - T)
        ref_a = dataset._pick_reference_start(n_frames, start, T)
        ref_b = dataset._pick_reference_start(n_frames, start, T)
        attempts = 0
        while ref_b == ref_a and attempts < 8:
            ref_b = dataset._pick_reference_start(n_frames, start, T)
            attempts += 1

        face_a, indiv_mels, _, _ = dataset.make_generator_sample(speaker_idx, start, ref_a)
        face_b, _, _, _ = dataset.make_generator_sample(speaker_idx, start, ref_b)

        indiv_mels = indiv_mels.unsqueeze(0).to(device)
        face_a = face_a.unsqueeze(0).to(device)
        face_b = face_b.unsqueeze(0).to(device)

        pred_a = run_generator(generator, indiv_mels, face_a)
        pred_b = run_generator(generator, indiv_mels, face_b)

        mouth_deltas.append(F.l1_loss(lower_half(pred_a), lower_half(pred_b)).item())
        upper_deltas.append(F.l1_loss(upper_half(pred_a), upper_half(pred_b)).item())

        if out_dir and not preview_saved:
            save_sequence_pair(os.path.join(out_dir, "check3_same_mel_diff_ref.png"), pred_a, pred_b)
            preview_saved = True

    mouth_mean = float(np.mean(mouth_deltas)) if mouth_deltas else 0.0
    upper_mean = float(np.mean(upper_deltas)) if upper_deltas else 0.0
    return {
        "name": "same_mel_different_ref",
        "pass": upper_mean > mouth_mean,
        "mouth_delta": mouth_mean,
        "upper_delta": upper_mean,
        "ratio_mouth_over_upper": float(mouth_mean / max(upper_mean, 1e-6)),
    }


def format_result(result):
    status = "PASS" if result["pass"] else "FAIL"
    metrics = ", ".join(
        f"{key}={value:.4f}"
        for key, value in result.items()
        if key not in {"name", "pass"}
    )
    return f"[{status}] {result['name']}: {metrics}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Generator checkpoint")
    parser.add_argument("--syncnet", required=True, help="Official official SyncNet checkpoint")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--samples", type=int, default=8, help="Random trials per check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="Optional folder for preview images")
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated list of speaker dirs to evaluate on")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ck, cfg = load_config(args.checkpoint, args.config)
    device = cfg["training"]["device"]

    if args.output_dir is None:
        ck_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(ck_dir, "sanity_checks")
    os.makedirs(args.output_dir, exist_ok=True)

    generator = load_generator(args.checkpoint, cfg, device)
    syncnet, syncnet_kind = load_syncnet(args.syncnet, device, cfg["syncnet"]["T"])
    speaker_allowlist = None
    if args.speaker_list:
        with open(args.speaker_list) as f:
            speaker_allowlist = [line.strip() for line in f if line.strip()]
        print(f"Speaker snapshot: {args.speaker_list} ({len(speaker_allowlist)} entries)")
    dataset = build_dataset(cfg, speaker_allowlist=speaker_allowlist)

    results = [
        check_same_face_different_mel(generator, syncnet, syncnet_kind, dataset, device, args.samples, args.output_dir),
        check_gt_correct_audio(syncnet, syncnet_kind, dataset, device, args.samples),
        check_same_mel_different_ref(generator, dataset, device, args.samples, args.output_dir),
    ]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"SyncNet: {args.syncnet} ({syncnet_kind})")
    print(f"Samples per check: {args.samples}")
    print(f"Preview dir: {args.output_dir}")
    print("")
    for result in results:
        print(format_result(result))

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        for result in results:
            f.write(format_result(result) + "\n")


if __name__ == "__main__":
    main()
