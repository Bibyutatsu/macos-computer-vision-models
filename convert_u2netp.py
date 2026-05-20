"""
Convert U²-Netp salient segmentation ONNX → CoreML (.mlpackage).

Source: models/u2netp.onnx
        opset 11, fixed input [1, 3, 320, 320]
        7 outputs: first is [1,1,320,320] (fixed); last 5 have symbolic dims
Output: models/u2netp.mlpackage
        Single output: primary saliency mask [1,1,320,320]

coremltools 7+ dropped the ONNX frontend. The conversion path is:
  ONNX  →  trim aux outputs  →  onnx2torch  →  torch.jit.trace  →  coremltools

The 5 auxiliary deep-supervision outputs (nodes 1961–1965) carry symbolic
dimension names and are unused by SnapGrade (subject_seg.py reads only
output[0]). We drop them before converting to avoid shape-inference issues.

Note: snapgrade/metrics/subject_seg.py currently loads via onnxruntime.
After conversion, add a CoreML loading branch in _load() mirroring the
pattern in snapgrade/metrics/objects.py:113.

Usage (from the SnapGrade repo root, using its venv):
    uv run python ../macos-computer-vision-models/convert_u2netp.py
    uv run python ../macos-computer-vision-models/convert_u2netp.py --force
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent


def _trim_to_primary_output(src: Path, dst: Path) -> None:
    """Drop auxiliary deep-supervision outputs; keep only the primary mask."""
    import onnx

    m = onnx.load(str(src))
    # output[0] is node 1959, shape [1,1,320,320] — the main saliency mask.
    # outputs[1:] are deep-supervision side outputs with symbolic dims.
    del m.graph.output[1:]
    onnx.save(m, str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert U²-Netp salient segmentation ONNX → CoreML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "models" / "u2netp.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "u2netp.mlpackage",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not args.force:
        print(f"Model already exists: {args.output}")
        print("Pass --force to regenerate.")
        sys.exit(0)

    import coremltools as ct
    import onnx2torch
    import torch

    warnings.filterwarnings("ignore")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        trimmed = Path(tf.name)

    try:
        print("Trimming auxiliary deep-supervision outputs …")
        _trim_to_primary_output(args.input, trimmed)

        print(f"Loading {args.input.name} via onnx2torch …")
        model = onnx2torch.convert(str(trimmed))
        model.eval()

        dummy = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            out = model(dummy)
        print(f"Forward pass OK — output shape: {tuple(out.shape)}")

        print("Tracing …")
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy, strict=False)

        print("Converting to CoreML …")
        # coremltools normalises "input.1" → "input_1" in the saved spec.
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input.1", shape=dummy.shape, dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        print(f"\nERROR: conversion failed — {exc}", file=sys.stderr)
        trimmed.unlink(missing_ok=True)
        sys.exit(1)

    trimmed.unlink(missing_ok=True)

    mlmodel.short_description = (
        "U²-Netp salient object segmentation. "
        "Input: [1,3,320,320] float32 (ImageNet-normalised). "
        "Output: primary saliency mask [1,1,320,320], values in [0,1]."
    )
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")
    print("\nTo activate, set:")
    print(f'  export SNAPGRADE_U2NETP_MODEL="{args.output.resolve()}"')


if __name__ == "__main__":
    main()
