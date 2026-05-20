"""
Convert CSAILVision Places365-ResNet18 to CoreML.

Downloads the pretrained checkpoint directly from CSAIL's HTTP server (no SSL
friction) and labels from GitHub via curl (same strategy as models.py).

  uv run python convert_places365.py

Outputs:
  ~/.blurdetector/models/places365.mlpackage
  ~/.blurdetector/models/places365_labels.txt

To activate scene classification, add to your shell profile:
  export BLURDETECTOR_SCENE_MODEL="$HOME/.blurdetector/models/places365.mlpackage"
  export BLURDETECTOR_SCENE_LABELS="$HOME/.blurdetector/models/places365_labels.txt"
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
import torch.nn as nn
import torchvision.models as tvm

REPO_ROOT = Path(__file__).resolve().parent


# HTTP (not HTTPS) — avoids macOS CA-store issues entirely.
CKPT_URL    = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
LABELS_URL  = (
    "https://raw.githubusercontent.com/CSAILVision/places365/"
    "master/categories_places365.txt"
)


def _curl_download(url: str, dst: Path) -> None:
    """Download via curl (preferred on macOS — avoids CA-bundle friction)."""
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-fL", "--retry", "3", "-o", str(dst), url],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, dst)


def _download_checkpoint(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"Checkpoint already cached: {dst}")
        return
    print(f"Downloading ResNet18-Places365 (~45 MB) → {dst} …")
    tmp = dst.with_suffix(".part")
    _curl_download(CKPT_URL, tmp)
    tmp.rename(dst)


def _download_labels(dst: Path) -> None:
    if dst.exists():
        print(f"Labels already present: {dst}")
        return
    print(f"Downloading Places365 labels → {dst} …")
    tmp = dst.with_suffix(".part")
    _curl_download(LABELS_URL, tmp)
    tmp.rename(dst)


def _parse_labels(path: Path) -> list[str]:
    labels = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "/a/airfield 0"  →  "airfield"
        name = line.split()[0].lstrip("/")  # "a/airfield"
        name = name.split("/", 1)[-1]       # "airfield"
        labels.append(name.replace("_", " "))
    return labels


def _build_model(ckpt_path: Path) -> nn.Module:
    print("Building ResNet18 (365-class head) …")
    model = tvm.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 365)

    print("Loading Places365 weights …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Checkpoints from DataParallel training have a "module." prefix.
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CSAILVision Places365-ResNet18 to CoreML."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory to save the converted model and labels",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="places365.mlpackage",
        help="Filename for the converted CoreML model",
    )
    parser.add_argument(
        "--labels-name",
        type=str,
        default="places365_labels.txt",
        help="Filename for the scene labels",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "resnet18_places365.pth.tar",
        help="Path to the PyTorch ResNet18-Places365 checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output model exists",
    )
    args = parser.parse_args()

    out_model = args.output_dir / args.model_name
    out_labels = args.output_dir / args.labels_name
    ckpt_path = args.checkpoint

    if out_model.exists() and not args.force:
        print(f"Model already exists: {out_model}")
        print("Delete it first or pass --force to regenerate.")
        sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    _download_checkpoint(ckpt_path)
    _download_labels(out_labels)

    labels = _parse_labels(out_labels)
    if len(labels) != 365:
        print(f"WARNING: expected 365 labels, got {len(labels)}")
    else:
        print(f"Loaded {len(labels)} class labels.")

    model = _build_model(ckpt_path)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 365), f"unexpected output shape: {out.shape}"
    print("Sanity check passed.")

    print("Tracing model …")
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    print("Converting to CoreML …")
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
    )

    mlmodel.save(str(out_model))
    print(f"\nSaved model  → {out_model}")
    print(f"Saved labels → {out_labels}")
    print("\nTo activate, add to your shell profile:")
    print(f'  export BLURDETECTOR_SCENE_MODEL="{out_model.resolve()}"')
    print(f'  export BLURDETECTOR_SCENE_LABELS="{out_labels.resolve()}"')


if __name__ == "__main__":
    main()
