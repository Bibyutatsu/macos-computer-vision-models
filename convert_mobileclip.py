"""Convert Apple MobileCLIP-S0 to CoreML (image + text towers).

Apple ships official PyTorch checkpoints via `ml-mobileclip`. This script
loads them, traces each tower separately, and exports two .mlpackages:

    models/mobileclip_s0_image.mlpackage   — input: (1, 3, 256, 256) float32
                                              output: (1, 512) float32
    models/mobileclip_s0_text.mlpackage    — input: (1, 77) int32 tokens
                                              output: (1, 512) float32

Both outputs are L2-normalized so callers can cosine-rank with a plain dot.

Usage:
    pip install ml-mobileclip torch coremltools
    python convert_mobileclip.py
    python update_manifest.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
CKPT_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/"
    "mobileclip_s0.pt"
)


def _curl_download(url: str, dst: Path) -> None:
    curl = shutil.which("curl")
    if curl:
        subprocess.run([curl, "-fL", "--retry", "3", "-o", str(dst), url], check=True)
    else:
        urllib.request.urlretrieve(url, dst)


def _download_checkpoint(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 10_000_000:
        print(f"Checkpoint cached: {dst}")
        return
    print(f"Downloading MobileCLIP-S0 (~30 MB) → {dst} …")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    _curl_download(CKPT_URL, tmp)
    tmp.rename(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "mobileclip_s0.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "models",
    )
    args = parser.parse_args()

    try:
        import mobileclip  # type: ignore
    except ImportError:
        print("ERROR: `pip install ml-mobileclip` first.", file=sys.stderr)
        sys.exit(1)

    # coremltools 8.x/9.0 _cast does `dtype(x.val)` which raises on a
    # 1-element numpy array with ndim > 0 (newer numpy refuses). Patch to use
    # .item() so the scalar coercion works regardless of shape.
    import coremltools.converters.mil.frontend.torch.ops as _ctops
    from coremltools.converters.mil.mil import Builder as _mb

    def _cast_patched(context, node, dtype, dtype_name):
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = x.val
            if hasattr(val, "size") and val.size == 1:
                val = val.item()
            try:
                res = _mb.const(val=dtype(val), name=node.name)
            except TypeError:
                # Fall back to dynamic cast for non-coercible constants.
                res = _mb.cast(x=x, dtype=dtype_name, name=node.name)
        elif len(x.shape) > 0:
            x = _mb.squeeze(x=x, name=node.name + "_item")
            res = _mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = _mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _ctops._cast = _cast_patched

    _download_checkpoint(args.checkpoint)

    print("Loading MobileCLIP-S0 …")
    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=str(args.checkpoint),
    )
    model.eval()

    class ImageTower(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            # Caller (snapgrade) L2-normalizes Python-side. In-graph normalize
            # gets traced into a CoreML rsqrt that produces near-zero outputs
            # on real images (root cause TBD — likely a graph constant-fold bug).
            return self.inner.encode_image(x)

    class TextTower(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, tokens):
            return self.inner.encode_text(tokens.long())

    img_wrap = ImageTower(model).eval()
    txt_wrap = TextTower(model).eval()

    import coremltools as ct

    print("Tracing + converting image tower …")
    img_example = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        img_traced = torch.jit.trace(img_wrap, img_example, strict=False)
    img_ml = ct.convert(
        img_traced,
        inputs=[ct.TensorType(name="image", shape=img_example.shape, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
    )
    img_out = args.out_dir / "mobileclip_s0_image.mlpackage"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    img_ml.save(str(img_out))
    print(f"  → {img_out}")

    print("Tracing + converting text tower …")
    txt_example = torch.zeros(1, 77, dtype=torch.int32)
    with torch.no_grad():
        txt_traced = torch.jit.trace(txt_wrap, txt_example, strict=False)
    txt_ml = ct.convert(
        txt_traced,
        inputs=[ct.TensorType(name="tokens", shape=txt_example.shape, dtype=np.int32)],
        minimum_deployment_target=ct.target.macOS13,
    )
    txt_out = args.out_dir / "mobileclip_s0_text.mlpackage"
    txt_ml.save(str(txt_out))
    print(f"  → {txt_out}")

    print("\nNext: run `python update_manifest.py` to refresh checksums.")


if __name__ == "__main__":
    main()
