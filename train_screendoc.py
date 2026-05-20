"""
Train a MobileNetV3-Small screenshot/document/photo classifier and export to CoreML.

No external dataset required — training data is generated synthetically.  The
synthetic images capture the statistical signatures that distinguish each class:
  screenshot  →  few distinct colors, hard rectangular edges, UI-palette tones
  document    →  high brightness, low saturation, regular horizontal text lines
  photo       →  smooth color gradients, natural noise, high color diversity

  uv run python train_screendoc.py [--samples N] [--epochs N]

Outputs:
  ~/.blurdetector/models/screendoc.mlpackage

To activate, add to your shell profile:
  export BLURDETECTOR_SCREENDOC_MODEL="$HOME/.blurdetector/models/screendoc.mlpackage"
"""

from __future__ import annotations

import argparse
import os
import random
import ssl
import sys
from pathlib import Path

# macOS framework Python often lacks a CA bundle; point ssl at certifi when available.
try:
    import certifi as _certifi
    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=_certifi.where()
    )
except ImportError:
    pass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

REPO_ROOT   = Path(__file__).resolve().parent
IMG_SIZE    = 224
CLASSES     = ("screenshot", "document", "photo")   # must match screendoc.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _gen_screenshot(rng: np.random.Generator) -> np.ndarray:
    """Solid-color UI blocks; intentionally few distinct hues."""
    n_colors = int(rng.integers(3, 10))
    # UI palettes lean towards grays, blues, whites
    palette = []
    for _ in range(n_colors):
        base = int(rng.integers(0, 256))
        tint = rng.integers(-30, 30, 3).clip(-base, 255 - base)
        palette.append(tuple(int(np.clip(base + t, 0, 255)) for t in tint))

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), palette[0])
    draw = ImageDraw.Draw(img)

    # Random rectangles (windows, buttons, panels)
    for _ in range(int(rng.integers(4, 18))):
        x1 = int(rng.integers(0, IMG_SIZE - 10))
        y1 = int(rng.integers(0, IMG_SIZE - 10))
        x2 = int(min(x1 + rng.integers(10, IMG_SIZE // 2 + 1), IMG_SIZE))
        y2 = int(min(y1 + rng.integers(5, IMG_SIZE // 3 + 1), IMG_SIZE))
        draw.rectangle([x1, y1, x2, y2], fill=palette[int(rng.integers(n_colors))])

    # Thin horizontal strip (menu/toolbar)
    if rng.random() > 0.4:
        y = int(rng.integers(15, 50))
        draw.rectangle([0, y, IMG_SIZE, y + int(rng.integers(18, 32))],
                       fill=palette[int(rng.integers(n_colors))])

    return np.array(img, dtype=np.uint8)


def _gen_document(rng: np.random.Generator) -> np.ndarray:
    """White/off-white background with dark horizontal text-row stripes."""
    bg = int(rng.integers(210, 256))
    img = np.full((IMG_SIZE, IMG_SIZE, 3), bg, dtype=np.uint8)

    margin = IMG_SIZE // 8
    line_h = int(rng.integers(8, 18))
    y = margin
    while y < IMG_SIZE - margin:
        thickness = int(rng.integers(1, 4))
        darkness  = int(rng.integers(20, 90))
        x_start = margin + int(rng.integers(0, 6))
        x_end   = IMG_SIZE - margin - int(rng.integers(0, 20))
        x_end   = max(x_end, x_start + 10)
        img[y: y + thickness, x_start:x_end] = darkness
        y += line_h + int(rng.integers(3, 9))

    # Optional column gutter
    if rng.random() > 0.6:
        cx = IMG_SIZE // 2 + int(rng.integers(-20, 20))
        img[:, cx: cx + 2] = int(rng.integers(150, 200))

    return img


def _gen_photo(rng: np.random.Generator) -> np.ndarray:
    """Smooth color gradient + per-pixel Gaussian noise → natural image stats."""
    # Build a tiny low-res color field and upscale (Perlin-like smoothness)
    base = int(rng.integers(4, 9))
    seed_rgb = rng.integers(0, 256, (base, base, 3), dtype=np.uint8)
    img = np.array(
        Image.fromarray(seed_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
        dtype=np.float32,
    )
    # Natural images have ~15-25 DN std-dev noise at ISO 400-1600
    noise = rng.normal(0, float(rng.uniform(10, 30)), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


_GENERATORS = [_gen_screenshot, _gen_document, _gen_photo]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    def __init__(self, samples_per_class: int, seed: int = 42):
        self._n = samples_per_class
        self._rng = np.random.default_rng(seed)
        self._items: list[tuple[int, np.ndarray]] = []
        for cls_idx, gen in enumerate(_GENERATORS):
            for _ in range(samples_per_class):
                self._items.append((cls_idx, gen(self._rng)))
        random.Random(seed).shuffle(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        cls_idx, img = self._items[idx]
        # [0,1] float tensor, CHW
        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return x, cls_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ScreenDocNet(nn.Module):
    """MobileNetV3-Small with ImageNet normalisation baked in.

    Inference input: float32 NCHW, values in [0, 1].
    The baked-in normalisation lets us reuse ImageNet pretrained weights while
    keeping the CoreML inference call simple (just /255, no explicit mean/std).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Normalisation is applied inside forward so the traced CoreML model
        # handles it automatically; inference code only needs to divide by 255.
        self.register_buffer("mean", _MEAN.clone())
        self.register_buffer("std",  _STD.clone())

        import torchvision.models as tvm
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Replace the final linear layer; keep the adaptive pool + dropout.
        backbone.classifier[-1] = nn.Linear(1024, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [0, 1] NCHW
        x = (x - self.mean) / self.std
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(samples_per_class: int, epochs: int) -> ScreenDocNet:
    print(f"Generating {samples_per_class * 3} synthetic images "
          f"({samples_per_class} per class) …")
    ds_train = SyntheticDataset(samples_per_class, seed=0)
    ds_val   = SyntheticDataset(max(samples_per_class // 5, 100), seed=99)
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=64, shuffle=False, num_workers=0)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Training on {device}.")

    model = ScreenDocNet().to(device)

    # Phase 1: freeze backbone, train only the classifier head (fast warm-up)
    for p in model.backbone.features.parameters():
        p.requires_grad_(False)

    opt  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(loader, train_mode: bool) -> tuple[float, float]:
        model.train(train_mode)
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss   = loss_fn(logits, y)
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * len(y)
                correct    += (logits.argmax(1) == y).sum().item()
                n          += len(y)
        return total_loss / n, correct / n

    warmup = min(epochs // 2, 10)
    print(f"\n— Phase 1: head-only warm-up ({warmup} epochs) —")
    for ep in range(1, warmup + 1):
        tr_loss, tr_acc = run_epoch(dl_train, True)
        va_loss, va_acc = run_epoch(dl_val,   False)
        print(f"  ep {ep:3d}/{warmup}  train {tr_acc:.1%}  val {va_acc:.1%}")

    # Phase 2: unfreeze everything and fine-tune end-to-end with lower LR
    for p in model.backbone.features.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup)

    remaining = epochs - warmup
    print(f"\n— Phase 2: end-to-end fine-tune ({remaining} epochs) —")
    best_acc, best_state = 0.0, None
    for ep in range(1, remaining + 1):
        tr_loss, tr_acc = run_epoch(dl_train, True)
        va_loss, va_acc = run_epoch(dl_val,   False)
        sched.step()
        print(f"  ep {ep:3d}/{remaining}  train {tr_acc:.1%}  val {va_acc:.1%}")
        if va_acc >= best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest validation accuracy: {best_acc:.1%}")
    return model.cpu()


# ---------------------------------------------------------------------------
# CoreML export
# ---------------------------------------------------------------------------

def export(model: ScreenDocNet, out_model_path: Path) -> None:
    import coremltools as ct

    model.eval()
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
    )
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_model_path))
    print(f"Saved → {out_model_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000,
                    help="synthetic images per class (default 2000)")
    ap.add_argument("--epochs",  type=int, default=30,
                    help="total training epochs (default 30)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing model")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "screendoc.mlpackage",
        help="Path to output CoreML .mlpackage bundle",
    )
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        print(f"Model already exists: {args.output}")
        print("Pass --force to retrain.")
        sys.exit(0)

    if args.epochs < 2:
        print("--epochs must be at least 2")
        sys.exit(1)

    model = train(args.samples, args.epochs)

    print("\nExporting to CoreML …")
    export(model, args.output)

    print("\nTo activate screendoc classification, add to your shell profile:")
    print(f'  export BLURDETECTOR_SCREENDOC_MODEL="{args.output.resolve()}"')


if __name__ == "__main__":
    main()
