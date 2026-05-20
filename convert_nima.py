"""
Convert the titu1994 NIMA MobileNetV1 Keras checkpoint to CoreML.

Input:  ~/.blurdetector/models/nima_mobilenet.h5
Output: ~/.blurdetector/models/nima.mlpackage

No TensorFlow required — weights are read directly from the h5 file
and mapped into a PyTorch MobileNetV1 graph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent



# ---------------------------------------------------------------------------
# MobileNetV1 architecture
# ---------------------------------------------------------------------------

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_relu = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_relu(self.dw_bn(self.dw(x)))
        x = self.pw_relu(self.pw_bn(self.pw(x)))
        return x


class MobileNetV1NIMA(nn.Module):
    # (in_ch, out_ch, stride)
    _CFG = [
        (32,   64,  1),
        (64,  128,  2),
        (128, 128,  1),
        (128, 256,  2),
        (256, 256,  1),
        (256, 512,  2),
        (512, 512,  1), (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 512, 1),
        (512, 1024, 2),
        (1024, 1024, 1),
    ]

    def __init__(self):
        super().__init__()
        self.conv1    = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList([
            DepthwiseSeparable(ic, oc, s) for ic, oc, s in self._CFG
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc   = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return torch.softmax(self.fc(x), dim=1)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def _np(f, *keys):
    node = f
    for k in keys:
        node = node[k]
    return np.array(node)


def load_conv(pt_conv: nn.Conv2d, f, layer: str, key: str = "kernel:0"):
    # Keras: (kH, kW, in, out)  →  PyTorch: (out, in, kH, kW)
    # Depthwise Keras: (kH, kW, in, 1)  →  PyTorch: (in, 1, kH, kW)
    # h5 layout is always layer/layer/weight_key
    w = _np(f, layer, layer, key)
    if pt_conv.groups > 1:
        # depthwise: (kH, kW, C, 1) → (C, 1, kH, kW)
        w = w.transpose(2, 3, 0, 1)
    else:
        # standard/pointwise: (kH, kW, in, out) → (out, in, kH, kW)
        w = w.transpose(3, 2, 0, 1)
    pt_conv.weight.data = torch.from_numpy(w.copy())


def load_bn(pt_bn: nn.BatchNorm2d, f, layer: str):
    pt_bn.weight.data       = torch.from_numpy(_np(f, layer, layer, "gamma:0").copy())
    pt_bn.bias.data         = torch.from_numpy(_np(f, layer, layer, "beta:0").copy())
    pt_bn.running_mean.data = torch.from_numpy(_np(f, layer, layer, "moving_mean:0").copy())
    pt_bn.running_var.data  = torch.from_numpy(_np(f, layer, layer, "moving_variance:0").copy())


def load_weights(model: MobileNetV1NIMA, h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        load_conv(model.conv1,    f, "conv1")
        load_bn(model.conv1_bn,   f, "conv1_bn")

        for i, blk in enumerate(model.blocks, start=1):
            load_conv(blk.dw,    f, f"conv_dw_{i}", "depthwise_kernel:0")
            load_bn(blk.dw_bn,   f, f"conv_dw_{i}_bn")
            load_conv(blk.pw,    f, f"conv_pw_{i}")
            load_bn(blk.pw_bn,   f, f"conv_pw_{i}_bn")

        # Dense → Linear: Keras (in, out) → PyTorch (out, in)
        w = _np(f, "dense_1", "dense_1", "kernel:0").T.copy()
        b = _np(f, "dense_1", "dense_1", "bias:0").copy()
        model.fc.weight.data = torch.from_numpy(w)
        model.fc.bias.data   = torch.from_numpy(b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert NIMA MobileNetV1 Keras checkpoint to CoreML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "nima_mobilenet.h5",
        help="Path to input Keras .h5 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "nima.mlpackage",
        help="Path to output CoreML .mlpackage bundle",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: checkpoint not found at {args.input}")
        sys.exit(1)

    print("Building MobileNetV1-NIMA …")
    model = MobileNetV1NIMA()

    print(f"Loading weights from {args.input} …")
    load_weights(model, args.input)
    model.eval()

    # Quick sanity check: output should be a probability distribution summing to 1
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 10), f"unexpected output shape: {out.shape}"
        total = out.sum().item()
        assert abs(total - 1.0) < 1e-4, f"output doesn't sum to 1: {total}"
    print(f"Sanity check passed — output sums to {total:.6f}")

    print("Tracing model …")
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    print("Converting to CoreML …")
    import coremltools as ct
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=example.shape,
            dtype=np.float32,
        )],
        minimum_deployment_target=ct.target.macOS13,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"\nSaved → {args.output}")
    print(f"\nTo enable aesthetic scoring, add this to your shell profile:")
    print(f'  export BLURDETECTOR_NIMA_MODEL="{args.output.resolve()}"')


if __name__ == "__main__":
    main()
