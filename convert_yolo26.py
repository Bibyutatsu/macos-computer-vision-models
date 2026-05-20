"""Convert Ultralytics YOLO26n to ONNX for SnapGrade object detection.

YOLO26 (released 2026-01-14) succeeds YOLOv8n: better small-object accuracy and
faster inference, same 80 COCO classes. We export to ONNX (not CoreML) because
the current toolchain (torch 2.12 + coremltools 9.0) can't trace the CoreML
graph — coremltools is only validated through torch 2.7. ONNX runs on
onnxruntime, which SnapGrade already depends on. Revisit CoreML/ANE export when
coremltools gains torch 2.12 support.

Exported with nms=False, so the raw output keeps YOLOv8's [1, 84, 8400] layout
(4 box + 80 class) and SnapGrade's existing Python NMS in objects.py applies
unchanged.

  uv run python convert_yolo26.py

Outputs:
  models/yolo26n.onnx               (committed, hosted as-is — no zip needed)
  ~/.snapgrade/models/yolo26n.onnx  (local cache, ready to use)

After running, refresh checksums:
  uv run python update_manifest.py
and copy the yolo26n digest into snapgrade/models_manifest.json.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_MODELS = Path(__file__).resolve().parent / "models"
CACHE_DIR = Path.home() / ".snapgrade" / "models"

IMG_SIZE = 640  # YOLO26n default; SnapGrade letterboxes to this.


def main() -> None:
    from ultralytics import YOLO

    REPO_MODELS.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO26n (downloads yolo26n.pt on first run) …")
    model = YOLO("yolo26n.pt")

    print("Exporting to ONNX (nms=False, raw [1,84,8400]) …")
    exported = Path(model.export(format="onnx", nms=False, imgsz=IMG_SIZE, opset=13))

    target = REPO_MODELS / "yolo26n.onnx"
    shutil.copy2(exported, target)
    print(f"  -> {target}")

    cache_target = CACHE_DIR / "yolo26n.onnx"
    shutil.copy2(target, cache_target)
    print(f"  -> {cache_target}")

    print("\nDone. Now: `uv run python update_manifest.py` and sync the digest "
          "into snapgrade/models_manifest.json.")


if __name__ == "__main__":
    main()
