# macOS Image Quality Assessment & Computer Vision Models

A centralized repository hosting pre-compiled CoreML models, ONNX weights, and conversion/training methodologies optimized for Apple Silicon and Intel Macs.

These models power the [BlurDetector](https://github.com/Bibyutatsu/BlurDetector) photo triage pipeline, providing local, private, and high-performance assessment of aesthetic quality, scene classification, screenshot/document filtering, face landmarks, and object detection.

---

## 📦 Models Included

All compiled models and serialized weights are stored in the [models/](models/) directory:

| Model / Feature | Format / Path | Size | Target / Usage | Origin / Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **NIMA** | [`models/nima.mlpackage`](models/nima.mlpackage) | 6.0 MB | Aesthetic scoring (scale 1–10) | MobileNetV1 (AVA dataset) |
| **Places365** | [`models/places365.mlpackage`](models/places365.mlpackage) | 44.8 MB | Scene classification (365 classes) | ResNet18 (CSAILVision) |
| **Screendoc** | [`models/screendoc.mlpackage`](models/screendoc.mlpackage) | 2.8 MB | Screenshot vs. Document vs. Photo | Custom MobileNetV3-Small |
| **YOLOv8n** | [`models/yolov8n.mlpackage`](models/yolov8n.mlpackage) | 6.2 MB | General object detection (80 classes) | Ultralytics YOLOv8 Nano |
| **U2Netp** | [`models/u2netp.onnx`](models/u2netp.onnx) | 4.6 MB | Salient object / subject segmentation | U2-Net Portable |
| **YuNet** | [`models/face_detection_yunet_2023mar.onnx`](models/face_detection_yunet_2023mar.onnx) | 232 KB | Fast, multi-scale face detection | OpenCV Zoo YuNet |
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

### 3. Screendoc Classifier
* **Goal**: Categorize images into **screenshot**, **document**, or **photo** to filter out non-memories.
* **Pipeline** (`train_screendoc.py`):
  Because no standard dataset exists for this task, the classifier is trained on programmatically generated synthetic images representing the statistical and visual signatures of each class:
  * **Screenshot Generator**: Draws flat rectangular UI blocks, buttons, and bars with limited, high-contrast color palettes (dominated by grays, whites, and blues).
  * **Document Generator**: Creates high-brightness, low-saturation off-white canvases containing dark, thin, horizontal line segments resembling text lines.
  * **Photo Generator**: Generates smooth color gradients by resizing tiny low-resolution seed grids using bicubic interpolation, then adds random per-pixel Gaussian noise to simulate sensor grain.
  
  **Training Architecture**:
  * Backbone: `MobileNetV3-Small` pre-trained on ImageNet.
  * Baked-in Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` are registered as model buffers, allowing the exported CoreML model to accept simple `[0, 1]` inputs without external normalization logic.
  * Phase 1: Freeze backbone features and warm up the classification head at `lr=1e-3`.
  * Phase 2: Unfreeze backbone and perform end-to-end fine-tuning at `lr=2e-4` with a Cosine Annealing learning rate scheduler.

```bash
# Re-run Screendoc training and CoreML export:
python train_screendoc.py --samples 2000 --epochs 30
```

---

### 4. Downstream / Built-in Models
The remaining models are imported from official distribution points and hosted here for direct downloading:
* **YOLOv8n**: Derived from Ultralytics' pre-trained weights, exported directly to CoreML format with baked-in Non-Maximum Suppression (NMS) for Neural Engine (ANE) acceleration. Used for identifying 80 classes of objects (especially people, pets, and vehicles).
* **U2Netp**: Pre-trained portable salient object segmentation model, used for bounding box isolation.
* **YuNet**: OpenCV Zoo's lightweight face detector, optimized for fast and accurate face detection at multiple scales.
* **Face Landmarker**: Google MediaPipe's Task bundle containing models for 478 face landmarks, blendshapes, and blink estimation.

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

### 3. Usage with BlurDetector

To use these models in BlurDetector, clone this repository or download the models, and configure the path environment variables:

```bash
export BLURDETECTOR_NIMA_MODEL="/path/to/models/nima.mlpackage"
export BLURDETECTOR_SCENE_MODEL="/path/to/models/places365.mlpackage"
export BLURDETECTOR_SCENE_LABELS="/path/to/models/places365_labels.txt"
export BLURDETECTOR_SCREENDOC_MODEL="/path/to/models/screendoc.mlpackage"
```

---

## 📜 License
* Conversion scripts and custom training code are licensed under the MIT License.
* Individual model weights are governed by their respective upstream licenses (e.g., AVA dataset terms, CSAILVision, Ultralytics YOLOv8, MediaPipe).
