"""
Convert YuNet face detector ONNX → CoreML (.mlpackage).

Source: models/face_detection_yunet_2023mar.onnx
        opset 11, fixed input [1, 3, 640, 640]
Output: models/yunet.mlpackage
        12 output heads (cls/obj/bbox/kps at strides 8, 16, 32)

coremltools 7+ dropped the ONNX frontend. The conversion path is:
  ONNX  →  onnx2torch.convert()  →  torch.jit.trace  →  coremltools

Note: SnapGrade currently loads YuNet via cv2.FaceDetectorYN.create() which
requires the ONNX file. This CoreML package is for future ANE-accelerated
inference; snapgrade/metrics/subject.py will need a separate update to switch
from the OpenCV DNN path to CoreML inference.

Usage (from the SnapGrade repo root, using its venv):
    uv run python ../macos-computer-vision-models/convert_yunet.py
    uv run python ../macos-computer-vision-models/convert_yunet.py --force
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YuNet face detector ONNX → CoreML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "yunet.mlpackage",
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

    print(f"Loading {args.input.name} via onnx2torch …")
    model = onnx2torch.convert(str(args.input))
    model.eval()

    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        outs = model(dummy)
    n_out = len(outs) if isinstance(outs, (list, tuple)) else 1
    print(f"Forward pass OK — {n_out} output heads")

    print("Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)

    print("Converting to CoreML …")
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=dummy.shape, dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        print(f"\nERROR: CoreML conversion failed — {exc}", file=sys.stderr)
        sys.exit(1)

    mlmodel.short_description = (
        "YuNet face detector (OpenCV Zoo, 2023mar). "
        "Input: [1,3,640,640] float32. "
        "Outputs: 12 multi-scale heads (cls/obj/bbox/kps at strides 8,16,32)."
    )
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
