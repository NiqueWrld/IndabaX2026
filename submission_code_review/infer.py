"""
Inference script for the IndabaX South Africa Cocoa Contamination Hackathon.

This reproduces the winning submission CSV (private LB 0.763846699)
using the trained YOLOv8s model.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).parent
TEST_IMAGES_DIR = ROOT / 'dataset' / 'images' / 'test'
SUBMISSION_PATH = ROOT / 'BenchmarkSubmission.csv'
WEIGHTS = ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'

CLASS_NAMES = ['anthracnose', 'cssvd', 'healthy']


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f'Model weights not found: {WEIGHTS}\n'
            'Run train.py first or place the trained weights at the expected path.'
        )

    model = YOLO(WEIGHTS)
    image_files = sorted(os.listdir(TEST_IMAGES_DIR))
    all_data = []

    for image_file in tqdm(image_files, desc='Generating predictions'):
        img_path = TEST_IMAGES_DIR / image_file
        results = model(
            str(img_path),
            imgsz=1024,
            conf=0.001,
            iou=0.6,
            max_det=300,
            augment=True,
            verbose=False,
        )
        r = results[0]

        if r.boxes and len(r.boxes):
            for box, cls, confv in zip(r.boxes.xyxy.tolist(),
                                       r.boxes.cls.tolist(),
                                       r.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                all_data.append({
                    'Image_ID': image_file,
                    'class': CLASS_NAMES[int(cls)],
                    'confidence': float(confv),
                    'ymin': round(y1),
                    'xmin': round(x1),
                    'ymax': round(y2),
                    'xmax': round(x2),
                })
        else:
            # Zindi requires every test Image_ID to appear in the submission
            all_data.append({
                'Image_ID': image_file,
                'class': 'healthy',
                'confidence': 0.01,
                'ymin': 100,
                'xmin': 100,
                'ymax': 100,
                'xmax': 100,
            })

    sub = pd.DataFrame(all_data)
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f'Saved submission: {SUBMISSION_PATH}')
    print(f'Rows: {len(sub)} | Unique IDs: {sub.Image_ID.nunique()}')
    print(sub['class'].value_counts())


if __name__ == '__main__':
    main()
