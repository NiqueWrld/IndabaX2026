"""
Training script for the IndabaX South Africa Cocoa Contamination Hackathon.

This reproduces the winning single-model solution (private LB 0.763846699).
Model: YOLOv8s trained on the provided train/val split.
"""

import torch
from ultralytics import YOLO
from pathlib import Path

SEED = 42
DATA_YAML = Path(__file__).parent / 'data.yaml'
PROJECT = Path(__file__).parent / 'runs' / 'detect'
NAME = 'train'


def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    model = YOLO('yolov8s.pt')  # pretrained COCO weights, openly available
    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=1024,
        batch=16,
        device=device,
        patience=20,
        seed=SEED,
        cache='disk',
        workers=8,
        cos_lr=True,
        close_mosaic=15,
        mixup=0.10,
        copy_paste=0.10,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
    )

    # Validate best weights
    best = PROJECT / NAME / 'weights' / 'best.pt'
    model = YOLO(best)
    model.val()
    print(f'Best weights saved to: {best}')


if __name__ == '__main__':
    main()
