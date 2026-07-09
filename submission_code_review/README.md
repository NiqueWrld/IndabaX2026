# IndabaX South Africa Cocoa Contamination Hackathon — Reproducible Solution

This repository contains the code to reproduce the 2nd-place private leaderboard submission (mAP@IoU=0.5: **0.763846699**) for the IndabaX South Africa Cocoa Contamination Hackathon.

## Solution summary

- **Model:** YOLOv8s (Ultralytics), pretrained on COCO
- **Input size:** 1024×1024
- **Training:** 40 epochs, batch size 16, seed 42
- **Inference:** single-scale TTA (`augment=True`), confidence threshold 0.001, IoU-NMS 0.6
- **Hardware used during development:** NVIDIA RTX A4000 16 GB, CUDA 12.4
- **Training time:** ~under 9 hours on the above GPU
- **Inference time:** ~under 3 hours on the above GPU for 1,626 test images

## Repository structure

```
.
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── data.yaml                 # YOLO dataset configuration
├── train.py                  # training script
├── infer.py                  # inference script
├── dataset/                  # provided train/val/test images and labels
└── runs/detect/train/weights/best.pt   # trained model weights
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the dataset is in `dataset/` with the structure expected by `data.yaml`:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    └── val/
```

## Reproduce training

```bash
python train.py
```

This writes the best weights to:

```
runs/detect/train/weights/best.pt
```

A fixed random seed (`seed=42`) is used so training is deterministic given the same hardware and software versions.

## Reproduce inference / submission

```bash
python infer.py
```

This generates `BenchmarkSubmission.csv` in the required Zindi format:

```
Image_ID class confidence ymin xmin ymax xmax
```

The script ensures every test image appears in the output, adding a placeholder `healthy` box with confidence 0.01 for images with no model predictions.

## Reproducibility notes

- All packages are open-source and publicly available.
- The pretrained `yolov8s.pt` checkpoint is openly distributed by Ultralytics.
- No external datasets, AutoML tools, or paid services are used.
- The same random seed is set for training.
- The submission scores reported were obtained with `ultralytics==8.4.90` and `torch==2.6.0+cu124`. Minor score differences may occur with different PyTorch / CUDA versions or different GPUs due to floating-point numerics.

## Edge-device suitability

The competition requires models suitable for low-resource smartphones. YOLOv8s is a compact architecture that can be exported to ONNX or TensorFlow Lite for mobile deployment:

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='tflite')
```

Training and inference stay within the competition limits (T4 GPU, 9 h training, 3 h inference).

## Mobile app bonus

A separate React + Capacitor Android demo is available in the `mobile_app/`
directory of the main repository. It ONNX-exports the trained model and runs
on-device inference using the phone camera. This is provided as optional
extra material and is not required to reproduce the competition submission.
