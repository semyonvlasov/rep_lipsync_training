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
import torch
import coremltools as ct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LipSyncGenerator
from config_loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Config file (overrides checkpoint config)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading {args.checkpoint}...")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if args.config:
        cfg = load_config(args.config)
    img_size = cfg["model"]["img_size"]
    predict_alpha = cfg["model"]["predict_alpha"]

    model = LipSyncGenerator(
        img_size=img_size,
        base_channels=cfg["model"]["base_channels"],
        predict_alpha=predict_alpha,
    )
    model.load_state_dict(ck["generator"])
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
            if self.gen.predict_alpha:
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
