"""Regenerate models/manifest.json with SHA-256 of every hosted artifact.

Run after adding/replacing any model:

    python update_manifest.py

The digests here must match snapgrade/models_manifest.json in the SnapGrade
repo — that bundled copy is what the client verifies against. After running
this, copy the relevant entries into the client manifest (or diff to confirm
they already agree).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# Map registry name (as used by SnapGrade) -> hosted artifact filename.
ARTIFACTS = {
    "yunet": "face_detection_yunet_2023mar.onnx",
    "yunet_coreml": "yunet.mlpackage.zip",
    "face_landmarker": "face_landmarker.task",
    "u2netp": "u2netp.onnx",
    "u2netp_coreml": "u2netp.mlpackage.zip",
    "yolo26n": "yolo26n.onnx",
    "yolo26n_coreml": "yolo26n.mlpackage.zip",
    "yolov8n": "yolov8n.mlpackage.zip",
    "nima": "nima.mlpackage.zip",
    "places365": "places365.mlpackage.zip",
    "places365_labels": "places365_labels.txt",
    "depth": "depth_anything_v2_small.onnx",
    "depth_coreml": "depth_anything_v2_small.mlpackage.zip",
    "hyperiqa": "hyperiqa.mlpackage.zip",
    "topiq": "topiq.mlpackage.zip",
    "mobileclip_image": "mobileclip_s0_image.mlpackage.zip",
    "mobileclip_text": "mobileclip_s0_text.mlpackage.zip",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    models: dict[str, str] = {}
    for name, filename in ARTIFACTS.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"  skip (missing): {filename}")
            continue
        models[name] = _sha256(path)
        print(f"  {name}: {models[name]}")
    out = {
        "_comment": "SHA-256 of each hosted model artifact. Keep in sync with "
        "snapgrade/models_manifest.json in the SnapGrade repo.",
        "models": models,
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {MODELS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
