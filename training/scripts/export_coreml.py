#!/usr/bin/env python3
"""
Export trained LipSync generator to CoreML for iOS deployment.

Usage:
    python scripts/export_coreml.py \
        --checkpoint output/training/generator/generator_epoch199.pth \
        --output lipsync_192_fp16.mlpackage --fp16
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
import coremltools as ct
import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))
from models import LipSyncGenerator
from config_loader import load_config


def load_official_wav2lip_class():
    candidate_roots = [
        REPO_ROOT.parent / "models" / "wav2lip",
        REPO_ROOT / "models" / "wav2lip",
    ]
    for root in candidate_roots:
        if not (root / "models").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        from models import Wav2Lip

        return Wav2Lip
    raise RuntimeError("Could not locate local official Wav2Lip package root")


def is_official_style_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".conv_block." in key for key in state_dict) or any(
        key.startswith("output_block.") for key in state_dict
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Config file. Required when the checkpoint does not embed config metadata.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading {args.checkpoint}...")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "generator" in ck:
        state_dict = ck["generator"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        state_dict = ck["state_dict"]
    else:
        state_dict = ck

    if not isinstance(state_dict, dict):
        raise KeyError(
            "Could not locate generator weights in checkpoint. "
            "Expected one of: checkpoint['generator'], checkpoint['state_dict'], or a raw state_dict."
        )

    official_style = is_official_style_state_dict(state_dict)

    cfg = ck.get("config") if isinstance(ck, dict) else None
    if args.config:
        cfg = load_config(args.config)
    if cfg is None and official_style:
        cfg = {"model": {"img_size": 96, "predict_alpha": False, "base_channels": 32}}
    if cfg is None:
        raise KeyError(
            "Checkpoint does not contain embedded config. "
            "Pass --config with the training yaml used for this model."
        )
    img_size = cfg["model"]["img_size"]
    predict_alpha = cfg["model"]["predict_alpha"]

    if official_style:
        Wav2Lip = load_official_wav2lip_class()
        model = Wav2Lip()
        predict_alpha = False
        img_size = 96
    else:
        model = LipSyncGenerator(
            img_size=img_size,
            base_channels=cfg["model"]["base_channels"],
            predict_alpha=predict_alpha,
        )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model: img_size={img_size}, predict_alpha={predict_alpha}")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.1f}M")

    # Trace
    mel = torch.randn(1, 1, 80, 16)
    face = torch.randn(1, 6, img_size, img_size)

    with torch.no_grad():
        if predict_alpha:
            out_face, out_alpha = model(mel, face)
            print(f"Output: face {out_face.shape}, alpha {out_alpha.shape}")
        else:
            out_face = model(mel, face)
            print(f"Output: face {out_face.shape}")

    # For tracing, wrap to return a single tensor or tuple
    class ExportWrapper(torch.nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, mel, face):
            if getattr(self.gen, "predict_alpha", False):
                f, a = self.gen(mel, face)
                return f, a
            return self.gen(mel, face)

    wrapper = ExportWrapper(model)
    wrapper.eval()

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (mel, face))

    # Convert to CoreML
    print("Converting to CoreML...")
    t0 = time.time()

    outputs = [ct.TensorType(name="face_output")]
    if predict_alpha:
        outputs.append(ct.TensorType(name="alpha_output"))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=(1, 1, 80, 16)),
            ct.TensorType(name="face", shape=(1, 6, img_size, img_size)),
        ],
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    print(f"Conversion done in {time.time() - t0:.1f}s")

    # Save
    if args.output is None:
        suffix = "_fp16" if args.fp16 else ""
        args.output = f"lipsync_{img_size}{suffix}.mlpackage"

    mlmodel.save(args.output)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(args.output) for f in fn
    ) / (1024 * 1024)

    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")
    print(f"Inputs:  mel (1,1,80,16), face (1,6,{img_size},{img_size})")
    print(f"Outputs: face ({img_size}x{img_size})"
          + (f", alpha ({img_size}x{img_size})" if predict_alpha else ""))

    # Validate
    print("\nValidating CoreML...")
    mel_np = mel.numpy()
    face_np = face.numpy()
    pred = mlmodel.predict({"mel": mel_np, "face": face_np})
    cml_face = pred["face_output"]
    pt_face = out_face.numpy()
    diff = np.abs(pt_face - cml_face).max()
    print(f"Max abs diff (face): {diff:.6f}")
    if predict_alpha and "alpha_output" in pred:
        cml_alpha = pred["alpha_output"]
        diff_a = np.abs(out_alpha.numpy() - cml_alpha).max()
        print(f"Max abs diff (alpha): {diff_a:.6f}")
    print("OK" if diff < 0.1 else "WARNING: large diff!")


if __name__ == "__main__":
    main()
