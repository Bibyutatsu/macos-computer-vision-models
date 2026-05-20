"""
Convert YOLO26n ONNX → CoreML (.mlpackage).

Source: models/yolo26n.onnx
        opset 13, fixed input [1, 3, 640, 640]
        Output: [1, 300, 6]  (xyxy + score + class_id, NMS-free end-to-end)
Output: models/yolo26n.mlpackage

coremltools 7+ dropped the ONNX frontend. The conversion path is:
  ONNX  →  onnx2torch.convert()  →  torch.jit.trace  →  coremltools

The torch 2.12 / coremltools 9.0 tracing warning is cosmetic for this model —
the traced TorchScript is generated cleanly by onnx2torch.

Note: snapgrade/metrics/objects.py CoreML branch (line 95) currently expects
the Ultralytics NMS-baked format (confidence + coordinates). This CoreML model
emits NMS-free [1,300,6] — the same layout as the ONNX path handled at
line 185. After deploying this model, update objects.py to add a branch that
handles the NMS-free format for .mlpackage inputs.

Usage (from the SnapGrade repo root, using its venv):
    uv run python ../macos-computer-vision-models/convert_yolo26_coreml.py
    uv run python ../macos-computer-vision-models/convert_yolo26_coreml.py --force
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
        description="Convert YOLO26n ONNX → CoreML (NMS-free [1,300,6] output)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "models" / "yolo26n.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "yolo26n.mlpackage",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        print("Run convert_yolo26.py first to export the ONNX.", file=sys.stderr)
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
        out = model(dummy)
    print(f"Forward pass OK — output shape: {tuple(out.shape)}")

    print("Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)

    print("Converting to CoreML …")
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="images", shape=dummy.shape, dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        print(f"\nERROR: CoreML conversion failed — {exc}", file=sys.stderr)
        sys.exit(1)

    mlmodel.short_description = (
        "YOLO26n object detector (Ultralytics, COCO 80 classes). "
        "Input: [1,3,640,640] float32, values in [0,1]. "
        "Output: [1,300,6] — xyxy + score + class_id, NMS-free."
    )
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")
    print("\nTo activate, set:")
    print(f'  export SNAPGRADE_YOLO_MODEL="{args.output.resolve()}"')
    print(
        "\nIMPORTANT: update snapgrade/metrics/objects.py — the CoreML branch "
        "currently expects Ultralytics NMS-baked format. This model emits the "
        "same NMS-free [1,300,6] layout as the ONNX path (line 185)."
    )


if __name__ == "__main__":
    main()
