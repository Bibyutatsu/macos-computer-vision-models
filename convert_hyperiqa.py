"""Convert HyperIQA (CVPR 2020) to CoreML.

HyperIQA is a no-reference IQA model with a ResNet50 backbone + a small
hyper-network head. The architecture is pure conv + linear ops — no exotic
upsampling or multi-element int casts that trip coremltools 9.0 — so it
converts cleanly via the direct torch.jit.trace → ct.convert path when torch
is pinned to 2.7.

Output:
    models/hyperiqa.mlpackage
        input:  (1, 3, 224, 224) float32, ImageNet-normalized
        output: (1,) float32 — quality in [0, 1]

Usage (using the dedicated .convert-venv pinned to torch 2.7):
    cd macos-computer-vision-models
    SSL_CERT_FILE=$(./.convert-venv/bin/python -c "import certifi; print(certifi.where())") \\
        ./.convert-venv/bin/python convert_hyperiqa.py
    ./.convert-venv/bin/python update_manifest.py
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
        "--output", type=Path, default=REPO_ROOT / "models" / "hyperiqa.mlpackage",
    )
    args = parser.parse_args()

    try:
        import pyiqa  # type: ignore
    except ImportError:
        print("ERROR: install pyiqa first — `pip install pyiqa`", file=sys.stderr)
        sys.exit(1)

    # pyiqa.archs.arch_util.uniform_crop calls F.interpolate(scale_factor=[Tensor, Tensor])
    # which torch 2.7+ rejects. Replace it with a size-based variant that's
    # equivalent for our fixed 224×224 input.
    import pyiqa.archs.arch_util as _au

    def _uniform_crop(images, output_size: int, num_crop: int):
        # Stock returns a single tensor of shape (B*num_crop, C, H, W) when
        # called with a single input. We're forcing num_crop=1 with a fixed
        # 224×224 input, so the result is just the input itself (or a fixed
        # resize if shape differs).
        x = images[0] if isinstance(images, (list, tuple)) else images
        if x.shape[-1] != output_size or x.shape[-2] != output_size:
            from torch.nn.functional import interpolate
            x = interpolate(x, size=(output_size, output_size), mode="bilinear", align_corners=False)
        return x

    _au.uniform_crop = _uniform_crop
    # Patch the symbol that hypernet_arch already imported.
    import pyiqa.archs.hypernet_arch as _hn
    _hn.uniform_crop = _uniform_crop

    print("Loading hyperiqa from pyiqa …")
    metric = pyiqa.create_metric("hyperiqa", as_loss=False)
    net = metric.net.eval()
    # HyperIQA's stock forward does 25-crop averaging via F.interpolate with
    # scale_factor=list — torch 2.7+ no longer accepts list scale_factors here.
    # Force single-crop (input is already 224×224, no resize needed) so the
    # graph is purely Conv/Linear and traces cleanly.
    if hasattr(net, "num_crop"):
        net.num_crop = 1

    class HyperIQAWrapper(torch.nn.Module):
        """Accepts a pre-normalized 224×224 tensor (caller does ImageNet
        mean/std + resize). Clamps the head's score to [0, 1]."""

        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            score = self.inner(x)
            return torch.clamp(score.view(-1), 0.0, 1.0)

    wrapped = HyperIQAWrapper(net).eval()

    print("Sanity check …")
    example = torch.zeros(1, 3, 224, 224)
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
