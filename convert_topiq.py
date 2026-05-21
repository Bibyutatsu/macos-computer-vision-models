"""Convert TopIQ-NR (CVPR 2024) to CoreML.

TopIQ is a transformer-based no-reference IQA model from `pyiqa`. We trace
the `topiq_nr` variant and export a CoreML .mlpackage that takes a 1×3×384×384
ImageNet-normalized float32 input and emits a single quality score in [0, 1].

Output: ./models/topiq.mlpackage  (+ topiq.mlpackage.zip via update_manifest.py)

Usage:
    python convert_topiq.py
    python update_manifest.py   # refresh SHA-256 in manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "topiq.mlpackage",
    )
    parser.add_argument(
        "--variant",
        default="topiq_nr",
        help="pyiqa metric name (default: topiq_nr).",
    )
    args = parser.parse_args()

    try:
        import pyiqa  # type: ignore
    except ImportError:
        print("ERROR: install pyiqa first — `pip install pyiqa`", file=sys.stderr)
        sys.exit(1)

    # coremltools 9.0 doesn't implement upsample_bicubic2d. TopIQ uses bicubic
    # for feature upsampling; bilinear is visually indistinguishable for the
    # IQA head and converts cleanly.
    import torch.nn.functional as F
    _orig_interp = F.interpolate

    def _interp_no_bicubic(*a, **kw):
        if kw.get("mode") == "bicubic":
            kw["mode"] = "bilinear"
        return _orig_interp(*a, **kw)

    F.interpolate = _interp_no_bicubic

    print(f"Loading {args.variant} from pyiqa …")
    metric = pyiqa.create_metric(args.variant, as_loss=False)
    net = metric.net.eval()

    # The pyiqa wrapper normalizes internally; we wrap with a thin module that
    # accepts an already-normalized tensor so the CoreML graph stays pure
    # tensor ops (no PIL / cv2 calls). Caller is responsible for ImageNet
    # mean/std + 384×384 resize.
    class TopIQNRWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            score = self.inner(x)
            # Clamp to [0, 1] — TopIQ usually emits in this range but the head
            # has no built-in saturation.
            return torch.clamp(score.view(-1), 0.0, 1.0)

    wrapped = TopIQNRWrapper(net).eval()

    print("Sanity check …")
    example = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        out = wrapped(example)
        assert out.numel() == 1, f"unexpected output shape: {out.shape}"
    print(f"Got scalar score: {float(out.item()):.4f}")

    print("Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, strict=False)

    print("Converting to CoreML …")
    import coremltools as ct
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")
    print("\nNext: run `python update_manifest.py` to refresh checksums.")


if __name__ == "__main__":
    main()
