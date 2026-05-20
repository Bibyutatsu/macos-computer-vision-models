# macOS Image Quality Assessment & Computer Vision Models

A centralized repository hosting pre-compiled CoreML models, ONNX weights, and conversion/training methodologies optimized for Apple Silicon and Intel Macs.

These models power the [SnapGrade](https://github.com/Bibyutatsu/SnapGrade) photo triage pipeline, providing local, private, and high-performance assessment of aesthetic quality, scene classification, screenshot/document filtering, face landmarks, and object detection.

---

## 📦 Models Included

All compiled models and serialized weights are stored in the [models/](models/) directory:

| Model / Feature | Format / Path | Size | Target / Usage | Origin / Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **NIMA** | [`models/nima.mlpackage`](models/nima.mlpackage) | 6.0 MB | Aesthetic scoring (scale 1–10) | MobileNetV1 (AVA dataset) |
| **Places365** | [`models/places365.mlpackage`](models/places365.mlpackage) | 44.8 MB | Scene classification (365 classes) | ResNet18 (CSAILVision) |
| **YOLO26n** | [`models/yolo26n.onnx`](models/yolo26n.onnx) | 9.5 MB | Object detection (80 classes), NMS-free | Ultralytics YOLO26 Nano |
| **YOLO26n CoreML** | [`models/yolo26n.mlpackage`](models/yolo26n.mlpackage) | 4.9 MB | ANE-accelerated object detection | converted via onnx2torch |
| **YOLOv8n** *(legacy)* | [`models/yolov8n.mlpackage`](models/yolov8n.mlpackage) | 6.2 MB | Superseded by YOLO26n; kept for fallback | Ultralytics YOLOv8 Nano |
| **U2Netp** | [`models/u2netp.onnx`](models/u2netp.onnx) | 4.6 MB | Salient object / subject segmentation | U2-Net Portable |
| **U2Netp CoreML** | [`models/u2netp.mlpackage`](models/u2netp.mlpackage) | 2.4 MB | ANE-accelerated saliency mask | converted via onnx2torch |
| **Depth-Anything-V2-S** | *(not tracked — 94 MB, download from HuggingFace)* | 94 MB | Monocular relative depth estimation | DINOv2 ViT-S (onnx-community) |
| **Depth-Anything CoreML** | [`models/depth_anything_v2_small.mlpackage`](models/depth_anything_v2_small.mlpackage) | 47 MB | ANE-accelerated depth estimation | converted via onnx2torch |
| **YuNet** | [`models/face_detection_yunet_2023mar.onnx`](models/face_detection_yunet_2023mar.onnx) | 232 KB | Fast, multi-scale face detection | OpenCV Zoo YuNet |
| **YuNet CoreML** | [`models/yunet.mlpackage`](models/yunet.mlpackage) | 200 KB | ANE-accelerated face detection | converted via onnx2torch |
| **Face Landmarker**| [`models/face_landmarker.task`](models/face_landmarker.task) | 3.8 MB | Eye/Blink detection & face mesh | MediaPipe FaceLandmarker |

---

## 🛠️ Methodologies & Conversion Pipelines

### 1. NIMA (Neural Image Assessment)
* **Goal**: Predict the aesthetic distribution of an image on a 1–10 scale.
* **Pipeline** (`convert_nima.py`):
  1. Reads Keras weights directly from titu1994's pre-trained `.h5` checkpoint using `h5py`.
  2. Constructs a native PyTorch `MobileNetV1` matching the Keras graph block-for-block (avoiding heavy TensorFlow dependencies on macOS).
  3. Transposes weights from Keras layout:
     * Standard Convolution: `(kH, kW, in_channels, out_channels)` ➜ `(out_channels, in_channels, kH, kW)`
     * Depthwise Convolution: `(kH, kW, channels, 1)` ➜ `(channels, 1, kH, kW)`
     * Linear: `(in_features, out_features)` ➜ `(out_features, in_features)`
  4. Traces the model graph with `torch.jit.trace` and exports to CoreML via `coremltools`.

```bash
# Re-run NIMA conversion:
python convert_nima.py --input checkpoints/nima_mobilenet.h5 --output models/nima.mlpackage
```

---

### 2. Places365 Scene Classifier
* **Goal**: Identify the semantic context of a scene (e.g., beach, forest, office).
* **Pipeline** (`convert_places365.py`):
  1. Downloads the standard ResNet18-Places365 weights via PyTorch Hub (`torch.hub.load`).
  2. Downloads the official 365-category label list and sanitizes it by stripping directory prefixes (e.g. `/a/airfield` ➜ `airfield`) and replacing underscores with spaces.
  3. Traces the ResNet18 graph with a dummy input `(1, 3, 224, 224)`.
  4. Converts to a channels-first CoreML model targeting macOS 13+.

```bash
# Re-run Places365 conversion:
python convert_places365.py
```

---

### 3. Screenshot / Document Classification — *retired*
The synthetic-trained `screendoc` MobileNetV3-Small model has been **removed**.
SnapGrade now classifies screenshot / document / photo with Apple Vision
(`VNRecognizeTextRequest` for OCR text density, `VNDetectDocumentSegmentationRequest`
for document quads) combined with EXIF camera-presence and colour-diversity
heuristics — no model download required. See `snapgrade/metrics/content_type.py`.

---

### 4. YOLO26n Object Detector
* **Goal**: Identify 80 COCO classes (people, pets, vehicles, common objects) to inform subject detection and organization.
* **Pipeline** (`convert_yolo26.py` → ONNX, `convert_yolo26_coreml.py` → CoreML):
  1. `convert_yolo26.py`: Loads Ultralytics' pre-trained `yolo26n.pt` and exports to ONNX with `nms=False`. YOLO26 is NMS-free end-to-end, yielding `[1, 300, 6]` output (`x1, y1, x2, y2, score, class_id`).
  2. `convert_yolo26_coreml.py`: Converts the ONNX to CoreML via `onnx2torch` → `torch.jit.trace` → `coremltools`. Output is the same NMS-free `[1, 300, 6]` tensor.

```bash
# Export ONNX (requires ultralytics):
uv run python convert_yolo26.py

# Convert ONNX → CoreML (run from SnapGrade repo root):
uv run python ../macos-computer-vision-models/convert_yolo26_coreml.py
```

---

### 5. Depth-Anything-V2-Small
* **Goal**: Monocular relative-depth estimation, used to separate foreground from background and flag the "subject out of focus" failure mode (sharp background, soft foreground) that a plain sharpness score misses.
* **Source ONNX**: Pre-exported from [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) on HuggingFace. Not tracked in git (94 MB); download manually and place at `models/depth_anything_v2_small.onnx`.
* **Conversion** (`convert_depth_anything.py`):
  1. Patches `Resize` nodes with `mode=cubic` → `mode=linear` in the ONNX graph (coremltools doesn't support `upsample_bicubic2d`; quality impact is imperceptible for relative depth).
  2. Fixes the dynamic input to `[1, 3, 518, 518]` (518 = 14 × 37, matching the ViT patch size and SnapGrade's `_IN = 518`).
  3. Converts via `onnx2torch` → `torch.jit.trace(strict=False)` → `coremltools`. Output: `[1, 518, 518]` depth map (higher = nearer).

```bash
# Convert ONNX → CoreML (run from SnapGrade repo root):
uv run python ../macos-computer-vision-models/convert_depth_anything.py
```

---

### 6. U2Netp Salient Segmentation
* **Goal**: Segment the salient / primary subject region; output used as a bbox mask for subject-aware sharpness scoring.
* **Conversion** (`convert_u2netp.py`):
  1. Trims the ONNX graph to keep only the primary output (node 1959, shape `[1,1,320,320]`). The 5 auxiliary deep-supervision side outputs (nodes 1961–1965) have symbolic dimension names that trip coremltools.
  2. Converts via `onnx2torch` → `torch.jit.trace(strict=False)` → `coremltools`. Input key: `input_1`; output key: `var_2227`.

```bash
# Convert ONNX → CoreML (run from SnapGrade repo root):
uv run python ../macos-computer-vision-models/convert_u2netp.py
```

---

### 7. YuNet Face Detector
* **Goal**: Fast, multi-scale face detection producing bounding boxes and 5 facial keypoints.
* **Conversion** (`convert_yunet.py`):
  1. Converts via `onnx2torch` → `torch.jit.trace(strict=False)` → `coremltools`. Produces 12 output heads (cls/obj/bbox/kps at strides 8, 16, 32).

```bash
# Convert ONNX → CoreML (run from SnapGrade repo root):
uv run python ../macos-computer-vision-models/convert_yunet.py
```

---

### 8. Downstream / Built-in Models
The remaining models are imported from official distribution points and hosted here for direct downloading:
* **YOLOv8n** *(legacy)*: Superseded by YOLO26n above; the CoreML package is kept one release for fallback on installs that haven't re-downloaded.
* **Face Landmarker**: Google MediaPipe's Task bundle containing models for 478 face landmarks, blendshapes, and blink estimation.

---

## Conversion toolchain

All ONNX → CoreML conversions use the same pipeline since coremltools 7+ removed the ONNX frontend:

```
ONNX → onnx2torch.convert() → torch.jit.trace(strict=False) → coremltools.convert()
```

Run conversions from the SnapGrade repo root so they pick up the SnapGrade `.venv`:

```bash
cd ~/Projects/BlurDetector
uv run python ../macos-computer-vision-models/convert_<model>.py
```

After converting, zip and regenerate the manifest:

```bash
cd models
zip -r yunet.mlpackage.zip yunet.mlpackage
# repeat for other models ...
cd ..
uv run python ../macos-computer-vision-models/update_manifest.py
```

---

## 🚀 Quickstart

### 1. Requirements

Install core dependencies (specifically requires `coremltools` on macOS):
```bash
pip install -r requirements.txt
```

### 2. Basic Inference (NIMA example)

```python
import coremltools as ct
import numpy as np
from PIL import Image

# Load the CoreML model
model = ct.models.MLModel("models/nima.mlpackage")

# Preprocess image (Resize to 224x224, normalize, transpose to CHW)
img = Image.open("photo.jpg").resize((224, 224))
arr = np.asarray(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
arr  = ((arr - mean) / std).transpose(2, 0, 1)[None, ...]  # (1, 3, 224, 224)

# Predict
out = model.predict({"input": arr})
probs = list(out.values())[0].ravel()

# Calculate aesthetic score (1-10 expected value)
score = float((probs * np.arange(1, 11)).sum())
print(f"Aesthetic Score: {score:.2f} / 10")
```

### 3. Usage with SnapGrade

To use these models in SnapGrade, clone this repository or download the models, and configure the path environment variables:

```bash
export SNAPGRADE_NIMA_MODEL="/path/to/models/nima.mlpackage"
export SNAPGRADE_SCENE_MODEL="/path/to/models/places365.mlpackage"
export SNAPGRADE_SCENE_LABELS="/path/to/models/places365_labels.txt"
export SNAPGRADE_YOLO_MODEL="/path/to/models/yolo26n.mlpackage"
export SNAPGRADE_U2NETP_MODEL="/path/to/models/u2netp.mlpackage"
export SNAPGRADE_DEPTH_MODEL="/path/to/models/depth_anything_v2_small.mlpackage"
```

---

## 📜 License
* Conversion scripts and custom training code are licensed under the MIT License.
* Individual model weights are governed by their respective upstream licenses (e.g., AVA dataset terms, CSAILVision, Ultralytics YOLOv8, MediaPipe).
