"""
Cocoa Contamination Hackathon - full pipeline as a plain Python script.
Converted from Cocoa_Disease_Starter_Notebook.ipynb (Colab bits removed, paths localized).

Usage:
    python app.py prep     # build train/val split + data.yaml
    python app.py train    # fine-tune YOLO
    python app.py infer    # predict on test set -> BenchmarkSubmission.csv
    python app.py all      # prep + train + infer
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).parent
INPUT_DATA_DIR = ROOT / 'indabax-south-africa-cocoa-contamination-hackathon20260703-3588-a14vp'
DATASETS_DIR = ROOT / 'dataset'

TRAIN_IMAGES_DIR = DATASETS_DIR / 'images' / 'train'
TRAIN_LABELS_DIR = DATASETS_DIR / 'labels' / 'train'
TEST_IMAGES_DIR = DATASETS_DIR / 'images' / 'test'
VAL_IMAGES_DIR = DATASETS_DIR / 'images' / 'val'
VAL_LABELS_DIR = DATASETS_DIR / 'labels' / 'val'

YAML_PATH = ROOT / 'data.yaml'
RUNS_DIR = ROOT / 'runs'
BEST_WEIGHTS = RUNS_DIR / 'detect' / 'train' / 'weights' / 'best.pt'
SUBMISSION_PATH = INPUT_DATA_DIR / 'BenchmarkSubmission.csv'

SEED = 42


def load_csvs():
    train = pd.read_csv(INPUT_DATA_DIR / 'Train.csv')
    test = pd.read_csv(INPUT_DATA_DIR / 'Test.csv')
    ss = pd.read_csv(INPUT_DATA_DIR / 'SampleSubmission.csv')
    return train, test, ss


def count_files(directory):
    return sum(len(files) for _, _, files in os.walk(directory))


# ----------------------------------------------------------------- prep ----
def prep():
    # Unpack dataset.zip only if the dataset folder isn't already extracted
    if not TRAIN_IMAGES_DIR.exists():
        zip_path = INPUT_DATA_DIR / 'dataset.zip'
        print(f'Unpacking {zip_path} ...')
        shutil.unpack_archive(zip_path, DATASETS_DIR)

    train, test, ss = load_csvs()

    # Class map: {'anthracnose': 0, 'cssvd': 1, 'healthy': 2}
    train['class'] = train['class'].str.strip()
    class_map = {cls: i for i, cls in enumerate(sorted(train['class'].unique()))}
    train['class_id'] = train['class'].map(class_map)
    print('Class map:', class_map)

    # Train/val split on unique image IDs
    train_names, val_names = train_test_split(
        train['Image_ID'].unique(), test_size=0.15, random_state=SEED)
    print(f'{len(train_names)} train images, {len(val_names)} val images')

    # Move val images + labels out of the train folders (idempotent)
    VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image_name in tqdm(val_names, desc='Moving val files'):
        src_img = TRAIN_IMAGES_DIR / image_name
        if not src_img.exists():
            continue  # already moved on a previous run
        shutil.move(src_img, VAL_IMAGES_DIR / image_name)
        label_name = Path(image_name).stem + '.txt'
        src_lbl = TRAIN_LABELS_DIR / label_name
        if src_lbl.exists():
            shutil.move(src_lbl, VAL_LABELS_DIR / label_name)
        moved += 1
    print(f'Moved {moved} images to val.')
    print(f'train imgs: {count_files(TRAIN_IMAGES_DIR)}, val imgs: {count_files(VAL_IMAGES_DIR)}, '
          f'test imgs: {count_files(TEST_IMAGES_DIR)}')

    # data.yaml for YOLO
    class_names = sorted(train['class'].unique().tolist())
    data_yaml = {
        'path': str(DATASETS_DIR.absolute()),
        'train': str(TRAIN_IMAGES_DIR.absolute()),
        'val': str(VAL_IMAGES_DIR.absolute()),
        'test': str(TEST_IMAGES_DIR.absolute()),
        'nc': len(class_names),
        'names': class_names,
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print('Wrote', YAML_PATH)
    print(data_yaml)


# ---------------------------------------------------------------- train ----
def train_model():
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    print('Training on device:', device)

    model = YOLO('yolov8n.pt')
    model.train(
        data=str(YAML_PATH),
        epochs=10,
        imgsz=1024,
        batch=24,        # RTX A4000 16GB: batch 8 only used ~3.4GB
        device=device,
        patience=5,
        seed=SEED,
        cache='disk',    # RAM cache x12 workers duplicates memory on Windows -> OOM
        workers=8,
        project=str(RUNS_DIR / 'detect'),
        name='train',
        exist_ok=True,
    )

    # Validate best weights
    model = YOLO(BEST_WEIGHTS)
    model.val()


# ---------------------------------------------------------------- infer ----
def infer():
    model = YOLO(BEST_WEIGHTS)
    image_files = os.listdir(TEST_IMAGES_DIR)

    all_data = []
    for image_file in tqdm(image_files, desc='Predicting'):
        results = model(str(TEST_IMAGES_DIR / image_file), verbose=False)
        r = results[0]
        if r.boxes and len(r.boxes):
            for box, cls, conf in zip(r.boxes.xyxy.tolist(),
                                      r.boxes.cls.tolist(),
                                      r.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                all_data.append({
                    'Image_ID': image_file,
                    'class': r.names[int(cls)],
                    'confidence': conf,
                    'ymin': y1, 'xmin': x1, 'ymax': y2, 'xmax': x2,
                })
        else:
            all_data.append({
                'Image_ID': image_file, 'class': 'None', 'confidence': None,
                'ymin': None, 'xmin': None, 'ymax': None, 'xmax': None,
            })

    sub = pd.DataFrame(all_data)
    sub.to_csv(SUBMISSION_PATH, index=False)
    print('Saved', SUBMISSION_PATH)
    print(sub['class'].value_counts())


# ------------------------------------------------- optional visualization ----
def plot_samples(n=5):
    import matplotlib.pyplot as plt

    def load_annotations(label_path):
        boxes = []
        for line in Path(label_path).read_text().splitlines():
            class_id, xc, yc, w, h = map(float, line.split())
            boxes.append((class_id, xc, yc, w, h))
        return boxes

    for image_name in os.listdir(TRAIN_IMAGES_DIR)[:n]:
        image_path = TRAIN_IMAGES_DIR / image_name
        label_path = TRAIN_LABELS_DIR / (Path(image_name).stem + '.txt')
        if not label_path.exists():
            print(f'No annotations for {image_name}')
            continue
        boxes = load_annotations(label_path)
        image = np.array(Image.open(image_path))
        h, w = image.shape[:2]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for class_id, xc, yc, bw, bh in boxes:
            xmin, ymin = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            xmax, ymax = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              edgecolor='red', facecolor='none', linewidth=2))
            plt.text(xmin, ymin - 10, f'Class {int(class_id)}',
                     color='red', fontsize=8, weight='bold')
        plt.title(image_name)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('prep', 'all'):
        prep()
    if step in ('train', 'all'):
        train_model()
    if step in ('infer', 'all'):
        infer()
    if step == 'plot':
        plot_samples()
