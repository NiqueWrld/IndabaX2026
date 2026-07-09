"""
Export the trained YOLOv8 model to ONNX format for mobile deployment.

Usage:
    python scripts/export_onnx.py

The exported model is saved to mobile_app/public/cocoa-disease.onnx
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT.parent / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
OUTPUT = ROOT / 'public' / 'cocoa-disease.onnx'


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {WEIGHTS}. "
            "Run train.py first to generate the model."
        )

    print(f"Loading weights from {WEIGHTS}")
    model = YOLO(WEIGHTS)

    print(f"Exporting ONNX model to {OUTPUT}")
    model.export(format='onnx', imgsz=1024, simplify=True)

    # Ultralytics saves the ONNX next to the .pt file; copy it to the mobile app
    exported = WEIGHTS.with_suffix('.onnx')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(exported.read_bytes())
    print(f"Saved mobile ONNX model: {OUTPUT}")


if __name__ == '__main__':
    main()
