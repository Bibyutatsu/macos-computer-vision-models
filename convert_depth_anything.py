"""
Convert Depth-Anything-V2-Small ONNX → CoreML (.mlpackage).

Source: models/depth_anything_v2_small.onnx
        opset 14, dynamic input ['batch_size', 3, 'height', 'width']
Output: models/depth_anything_v2_small.mlpackage
        Fixed input [1, 3, 518, 518]
        Output depth map [1, 518, 518]  (higher value = nearer)

Input is fixed to 518×518 (= 14 × 37), matching the ViT patch size of 14
and SnapGrade's _IN = 518 constant in snapgrade/metrics/depth.py.

coremltools 7+ dropped the ONNX frontend. The conversion path is:
  ONNX  →  onnx2torch.convert()  →  torch.jit.trace  →  coremltools

Tracing the ViT with a fixed input resolves the dynamic shapes. The Where /
Equal / Erf (GELU) ops convert cleanly through the TorchScript path.

Note: snapgrade/metrics/depth.py currently loads via onnxruntime.
After conversion, add a CoreML loading branch in _load() and update
_model_path() to prefer depth_anything_v2_small.mlpackage when present.

Usage (from the SnapGrade repo root, using its venv):
    uv run python ../macos-computer-vision-models/convert_depth_anything.py
    uv run python ../macos-computer-vision-models/convert_depth_anything.py --force
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

# 518 = 14 × 37 — must be a multiple of the ViT patch size (14).
# Matches depth.py's _IN constant.
_IN = 518


def _patch_cubic_resize(src: Path, dst: Path) -> int:
    """Replace cubic Resize nodes with linear (bilinear). Returns patch count.

    coremltools doesn't support upsample_bicubic2d in the TorchScript frontend.
    For relative-depth estimation the quality difference is imperceptible.
    """
    import onnx

    m = onnx.load(str(src))
    patched = 0
    for node in m.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"cubic":
                attr.s = b"linear"
                patched += 1
    onnx.save(m, str(dst))
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Depth-Anything-V2-Small ONNX → CoreML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "models" / "depth_anything_v2_small.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "depth_anything_v2_small.mlpackage",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=_IN,
        help="Input spatial size (must be a multiple of 14, the ViT patch size).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.size % 14 != 0:
        print(
            f"ERROR: --size {args.size} is not a multiple of 14 (ViT patch size).",
            file=sys.stderr,
        )
        sys.exit(1)

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

    h = w = args.size

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        patched_path = Path(tf.name)

    n = _patch_cubic_resize(args.input, patched_path)
    if n:
        print(f"Patched {n} cubic→bilinear Resize node(s) (coremltools doesn't support bicubic)")

    print(f"Loading {args.input.name} via onnx2torch …")
    model = onnx2torch.convert(str(patched_path))
    model.eval()

    dummy = torch.zeros(1, 3, h, w)
    print(f"Forward pass with {h}×{w} input …")
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward pass OK — output shape: {tuple(out.shape)}")

    print("Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)

    print("Converting to CoreML …")
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="pixel_values", shape=dummy.shape, dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        print(f"\nERROR: CoreML conversion failed — {exc}", file=sys.stderr)
        patched_path.unlink(missing_ok=True)
        sys.exit(1)

    patched_path.unlink(missing_ok=True)

    mlmodel.short_description = (
        f"Depth-Anything-V2-Small monocular depth estimator (DINOv2 ViT-S). "
        f"Input: [1,3,{h},{w}] float32 (ImageNet-normalised). "
        f"Output: predicted_depth [1,{h},{w}] — relative depth, higher = nearer."
    )
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")
    print("\nTo activate, set:")
    print(f'  export SNAPGRADE_DEPTH_MODEL="{args.output.resolve()}"')


if __name__ == "__main__":
    main()
