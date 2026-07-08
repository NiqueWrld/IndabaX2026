"""
Train a YOLO object detector for the Cocoa Contamination hackathon.

Designed for the competition's resource limits (single T4, <=9h train).
Runs on Kaggle/Colab (T4) or any CUDA machine. Falls back to CPU (slow).

Usage:
  python train.py                       # default: yolo11s, 640px
  python train.py --model yolo11m --imgsz 768 --epochs 120

Export to ONNX (edge-device requirement) happens automatically at the end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11s.pt",
                   help="pretrained weights (yolo11n/s/m.pt). n=fastest, m=most accurate")
    p.add_argument("--data", default=str(ROOT / "yolo" / "cocoa.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto (60%% VRAM)")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--name", default="cocoa_yolo11s")
    args = p.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        project=str(ROOT / "runs"),   # pin output dir (ignore global ultralytics settings)
        name=args.name,
        seed=42,               # reproducibility (competition rule)
        cache="ram",           # 48 GB / T4 RAM: cache images for speed
        cos_lr=True,
        close_mosaic=10,
        # augmentation for generalisation to unseen diseases
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.2,
        mosaic=1.0, mixup=0.1, scale=0.5,
        amp=True,              # mixed precision -> faster on T4
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")

    # Export to ONNX for the edge-device / TFLite requirement.
    YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print("exported ONNX next to best.pt")


if __name__ == "__main__":
    main()
