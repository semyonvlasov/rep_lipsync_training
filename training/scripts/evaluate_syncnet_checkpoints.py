#!/usr/bin/env python3
"""
Apples-to-apples SyncNet checkpoint comparison on one deterministic eval split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from contextlib import nullcontext


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(TRAINING_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT / "models"))

from data import LipSyncDataset
from config_loader import load_config
from syncnet import SyncNet
from syncnet_mirror import SyncNetMirror
from scripts.dataset_roots import get_dataset_roots


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


def build_syncnet_model(model_type: str, syncnet_T: int, device: str):
    if model_type == "mirror":
        return SyncNetMirror().to(device)
    return SyncNet(T=syncnet_T).to(device)


def syncnet_forward(model, model_type: str, visual, audio):
    if model_type == "mirror":
        audio_emb, face_emb = model(audio, visual)
        return audio_emb, face_emb
    visual_emb, audio_emb = model(visual, audio)
    return audio_emb, visual_emb


def syncnet_loss(model_type: str, audio_emb, visual_emb, labels):
    if model_type == "mirror":
        return SyncNetMirror.cosine_loss(audio_emb, visual_emb, labels)
    return SyncNet.cosine_loss(visual_emb, audio_emb, labels)


def syncnet_acc(cos_sim, model_type: str):
    threshold = 0.5 if model_type == "mirror" else 0.0
    return (cos_sim > threshold).float()


def build_syncnet_loader(dataset, cfg: dict, batch_size: int, device: str, model_type: str, shuffle: bool, is_eval: bool):
    from torch.utils.data import DataLoader

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


def load_official_syncnet_class():
    official_models_dir = REPO_ROOT / "models" / "official_syncnet" / "models"

    def load_module(name: str, filepath: Path):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    conv_mod = load_module("official_syncnet_models.conv", official_models_dir / "conv.py")
    syncnet_src = (official_models_dir / "syncnet.py").read_text()
    syncnet_src = syncnet_src.replace("from .conv import Conv2d", "")
    syncnet_ns = {"__builtins__": __builtins__}
    exec("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n", syncnet_ns)
    syncnet_ns["Conv2d"] = conv_mod.Conv2d
    exec(syncnet_src, syncnet_ns)
    return syncnet_ns["SyncNet_color"]


def load_official_syncnet_model(checkpoint_path: str, device: str):
    SyncNetCls = load_official_syncnet_class()
    model = SyncNetCls().to(device)
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
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_syncnet_init_weights(checkpoint_path: str, model, device: str) -> dict:
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


def capture_rng_state(device: str) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_eval_rng(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def cuda_autocast(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


def evaluate_syncnet_model(model, model_type: str, loader, device: str, max_batches=None, use_amp: bool = False, seed: int | None = None):
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
                loss = syncnet_loss(model_type, audio_emb, visual_emb, labels)
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
    return {"loss": total_loss / batches, "acc": total_acc / batches, "batches": batches}


def build_pairwise_eval_items(dataset, min_frames: int):
    items = []
    for speaker_key in dataset.speakers:
        entry = dataset._entries[speaker_key]
        n_frames = dataset._entry_frame_count(entry)
        if n_frames <= (3 * dataset.syncnet_T):
            continue
        if n_frames < min_frames:
            continue
        items.append({"name": entry["name"], "key": speaker_key, "n_frames": n_frames})
    return items


def load_pairwise_eval_item(dataset, cache: dict, item: dict):
    key = item["key"]
    value = cache.get(key)
    if value is not None:
        return value
    frames, mel_chunks, _ = dataset._load_speaker(key)
    n_frames = min(int(frames.shape[0]), len(mel_chunks))
    value = (frames, mel_chunks, n_frames)
    cache[key] = value
    return value


def resize_face(frame, size: int = 96):
    import cv2

    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size))
    return frame.astype(np.float32) / 255.0


def build_pairwise_visual_window(frames, start: int, T: int, img_size: int = 96):
    window = [resize_face(frames[start + t], img_size) for t in range(T)]
    lowers = [face[img_size // 2 :, :, :] for face in window]
    visual = np.concatenate([lower.transpose(2, 0, 1) for lower in lowers], axis=0)
    return visual.astype(np.float32, copy=False)


def build_pairwise_audio(mel_chunks, start: int, T: int, model_type: str):
    if model_type == "mirror":
        return mel_chunks[start][np.newaxis, np.newaxis]
    mel_concat = np.concatenate([mel_chunks[start + t] for t in range(T)], axis=1)
    return mel_concat[np.newaxis, np.newaxis]


def score_syncnet_pair(model, model_type: str, visual_np, pos_audio_np, neg_audio_np, device: str):
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
    model_type: str,
    dataset,
    pairwise_items,
    device: str,
    samples: int,
    seed: int,
    img_size: int,
    T: int,
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

    return {
        "shifted_pairwise_acc": shifted_hits / max(1, samples),
        "shifted_margin_mean": float(np.mean(shifted_margins)),
        "foreign_pairwise_acc": foreign_hits / max(1, samples),
        "foreign_margin_mean": float(np.mean(foreign_margins)),
        "pairwise_acc_mean": float((shifted_hits + foreign_hits) / max(1, 2 * samples)),
        "margin_mean": float((np.mean(shifted_margins) + np.mean(foreign_margins)) / 2.0),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SyncNet checkpoints on one deterministic eval split")
    parser.add_argument("--config", required=True, help="SyncNet training config YAML")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Checkpoint candidate as name=/abs/or/relative/path.pth; repeatable",
    )
    parser.add_argument("--output", required=True, help="Path to JSON output")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--pairwise-eval-samples", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--pairwise-eval-seed", type=int, default=None)
    parser.add_argument("--val-speaker-list", default=None, help="Optional explicit val snapshot")
    parser.add_argument("--val-count", type=int, default=2048)
    parser.add_argument("--hdtf-tiers", default="confident")
    parser.add_argument("--talkvid-tiers", default="confident,medium")
    parser.add_argument("--official-checkpoint", default=None)
    return parser.parse_args()


def resolve_repo_path(path: str | None) -> str | None:
    if not path:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((REPO_ROOT / path_obj).resolve())


def parse_candidate(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --candidate {raw!r}; expected name=path")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid --candidate {raw!r}; expected name=path")
    return name, resolve_repo_path(path)


def parse_tiers(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_split_from_lazy_imports(
    *,
    training_root: Path,
    val_count: int,
    hdtf_tiers: list[str],
    talkvid_tiers: list[str],
) -> tuple[list[str], dict]:
    roots = []
    for tier in hdtf_tiers:
        roots.append(("hdtf", tier, training_root / "data" / "hdtf" / "processed" / "_lazy_imports" / tier))
    for tier in talkvid_tiers:
        roots.append(("talkvid", tier, training_root / "data" / "talkvid" / "processed" / "_lazy_imports" / tier))

    items: list[tuple[float, str]] = []
    source_counts: dict[str, int] = {}
    for source_name, tier_name, root in roots:
        if not root.exists():
            continue
        count = 0
        for json_path in root.rglob("*.json"):
            if json_path.name == "summary.json":
                continue
            items.append((json_path.stat().st_mtime, json_path.stem))
            count += 1
        source_counts[f"{source_name}:{tier_name}"] = count

    unique: dict[str, float] = {}
    for mtime, name in sorted(items, key=lambda x: (x[0], x[1])):
        unique[name] = mtime

    ordered = sorted(((mtime, name) for name, mtime in unique.items()), key=lambda x: (x[0], x[1]))
    if len(ordered) <= val_count:
        raise RuntimeError(f"Need more than val_count={val_count} samples, found {len(ordered)}")

    val_names = [name for _, name in ordered[-val_count:]]
    summary = {
        "total": len(ordered),
        "val_count": len(val_names),
        "hdtf_tiers": hdtf_tiers,
        "talkvid_tiers": talkvid_tiers,
        "source_counts": source_counts,
        "val_tail": val_names[-10:],
        "split_source": "rebuilt_from_lazy_imports",
    }
    return val_names, summary


def load_name_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_val_dataset(cfg: dict, val_names: list[str]) -> LipSyncDataset:
    return LipSyncDataset(
        roots=get_dataset_roots(cfg),
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="syncnet",
        syncnet_style=cfg["syncnet"].get("model_type", "local"),
        cache_size=cfg["data"].get("cache_size", 8),
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        speaker_allowlist=val_names,
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
        **build_sync_alignment_kwargs(cfg),
    )


def evaluate_candidate(
    *,
    name: str,
    checkpoint_path: str,
    cfg: dict,
    val_loader,
    val_dataset,
    pairwise_items,
    eval_batches: int,
    eval_seed: int,
    pairwise_eval_samples: int,
    pairwise_eval_seed: int,
    device: str,
) -> dict:
    model_type = cfg["syncnet"].get("model_type", "local")
    model = build_syncnet_model(model_type, cfg["syncnet"]["T"], device)
    init_info = load_syncnet_init_weights(checkpoint_path, model, device)
    official_eval = evaluate_syncnet_model(
        model,
        model_type,
        val_loader,
        device,
        max_batches=eval_batches,
        use_amp=False,
        seed=eval_seed,
    )
    our_eval = evaluate_syncnet_pairwise(
        model,
        model_type,
        val_dataset,
        pairwise_items,
        device,
        samples=pairwise_eval_samples,
        seed=pairwise_eval_seed,
        img_size=cfg["model"]["img_size"],
        T=cfg["syncnet"]["T"],
    )
    return {
        "checkpoint": checkpoint_path,
        "init_info": init_info,
        "official_eval": official_eval,
        "our_eval": our_eval,
    }


def evaluate_official(
    *,
    checkpoint_path: str,
    cfg: dict,
    val_loader,
    val_dataset,
    pairwise_items,
    eval_batches: int,
    eval_seed: int,
    pairwise_eval_samples: int,
    pairwise_eval_seed: int,
    device: str,
) -> dict:
    model = load_official_syncnet_model(checkpoint_path, device)
    official_eval = evaluate_syncnet_model(
        model,
        "mirror",
        val_loader,
        device,
        max_batches=eval_batches,
        use_amp=False,
        seed=eval_seed,
    )
    our_eval = evaluate_syncnet_pairwise(
        model,
        "mirror",
        val_dataset,
        pairwise_items,
        device,
        samples=pairwise_eval_samples,
        seed=pairwise_eval_seed,
        img_size=cfg["model"]["img_size"],
        T=cfg["syncnet"]["T"],
    )
    return {
        "checkpoint": checkpoint_path,
        "official_eval": official_eval,
        "our_eval": our_eval,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    eval_batches = int(args.eval_batches or cfg["syncnet"].get("eval_batches", 1400))
    eval_seed = int(args.eval_seed or cfg["syncnet"].get("eval_seed", 20260329))
    pairwise_eval_samples = int(args.pairwise_eval_samples or cfg["syncnet"].get("pairwise_eval_samples", 200))
    pairwise_eval_seed = int(args.pairwise_eval_seed or cfg["syncnet"].get("pairwise_eval_seed", eval_seed))

    if args.val_speaker_list:
        val_names = load_name_list(resolve_repo_path(args.val_speaker_list))
        split_summary = {
            "split_source": "provided_val_speaker_list",
            "val_speaker_list": resolve_repo_path(args.val_speaker_list),
            "val_count": len(val_names),
        }
    else:
        val_names, split_summary = build_split_from_lazy_imports(
            training_root=TRAINING_ROOT,
            val_count=int(args.val_count),
            hdtf_tiers=parse_tiers(args.hdtf_tiers),
            talkvid_tiers=parse_tiers(args.talkvid_tiers),
        )

    val_dataset = build_val_dataset(cfg, val_names)
    val_loader = build_syncnet_loader(
        val_dataset,
        cfg,
        cfg["syncnet"]["batch_size"],
        args.device,
        cfg["syncnet"].get("model_type", "local"),
        shuffle=False,
        is_eval=True,
    )
    pairwise_items = build_pairwise_eval_items(val_dataset, min_frames=12)

    official_checkpoint = resolve_repo_path(args.official_checkpoint or cfg["syncnet"]["official_checkpoint"])
    if not official_checkpoint or not os.path.exists(official_checkpoint):
        raise FileNotFoundError(f"Official checkpoint not found: {official_checkpoint}")

    results = {
        "config": resolve_repo_path(args.config),
        "device": args.device,
        "eval_batches": eval_batches,
        "eval_seed": eval_seed,
        "pairwise_eval_samples": pairwise_eval_samples,
        "pairwise_eval_seed": pairwise_eval_seed,
        "dataset_roots": get_dataset_roots(cfg),
        "split_summary": split_summary,
        "val_dataset_size": len(val_dataset),
        "val_loader_batches": len(val_loader),
        "pairwise_item_count": len(pairwise_items),
        "official_baseline": evaluate_official(
            checkpoint_path=official_checkpoint,
            cfg=cfg,
            val_loader=val_loader,
            val_dataset=val_dataset,
            pairwise_items=pairwise_items,
            eval_batches=eval_batches,
            eval_seed=eval_seed,
            pairwise_eval_samples=pairwise_eval_samples,
            pairwise_eval_seed=pairwise_eval_seed,
            device=args.device,
        ),
        "candidates": {},
    }

    for raw_candidate in args.candidate:
        name, checkpoint_path = parse_candidate(raw_candidate)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        results["candidates"][name] = evaluate_candidate(
            name=name,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            val_loader=val_loader,
            val_dataset=val_dataset,
            pairwise_items=pairwise_items,
            eval_batches=eval_batches,
            eval_seed=eval_seed,
            pairwise_eval_samples=pairwise_eval_samples,
            pairwise_eval_seed=pairwise_eval_seed,
            device=args.device,
        )

    output_path = Path(resolve_repo_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"saved={output_path}")
    print(f"val_dataset_size={results['val_dataset_size']}")
    print(f"val_loader_batches={results['val_loader_batches']}")
    print(
        "official_syncnet:"
        f" off_loss={results['official_baseline']['official_eval']['loss']:.4f}"
        f" off_acc={results['official_baseline']['official_eval']['acc']:.4f}"
        f" our_pairwise={results['official_baseline']['our_eval']['pairwise_acc_mean']:.4f}"
        f" our_margin={results['official_baseline']['our_eval']['margin_mean']:.4f}"
    )
    for name, metrics in results["candidates"].items():
        print(
            f"{name}:"
            f" off_loss={metrics['official_eval']['loss']:.4f}"
            f" off_acc={metrics['official_eval']['acc']:.4f}"
            f" our_pairwise={metrics['our_eval']['pairwise_acc_mean']:.4f}"
            f" our_margin={metrics['our_eval']['margin_mean']:.4f}"
            f" epoch={metrics['init_info'].get('epoch')}"
            f" step={metrics['init_info'].get('global_step')}"
        )


if __name__ == "__main__":
    main()
