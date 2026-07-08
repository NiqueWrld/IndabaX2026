"""
Cocoa Contamination Hackathon - full pipeline as a plain Python script.
Converted from Cocoa_Disease_Starter_Notebook.ipynb (Colab bits removed, paths localized).

Usage:
    python app.py prep              # build train/val split + data.yaml
    python app.py train             # fine-tune yolov8m (strongest available)
    python app.py train_v8m         # fine-tune yolov8m
    python app.py train_v8s         # fine-tune yolov8s
    python app.py train_v8n         # fine-tune yolov8n
    python app.py train_all         # train v8m + v8s + v8n
    python app.py infer             # single-model inference with TTA
    python app.py infer_ensemble    # multi-model + multi-scale TTA + WBF
    python app.py all               # prep + train + infer
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

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except Exception:
    WBF_AVAILABLE = False

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

# Class order must match data.yaml / YOLO model.names
CLASS_NAMES = ['anthracnose', 'cssvd', 'healthy']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# Models we can train/ensemble.  yolo11m.pt is NOT in this workspace,
# so default to the strongest available checkpoint: yolov8m.pt.
# workers=0 + cache='ram' is the Windows-stable config that avoids dataloader deadlocks.
MODEL_CONFIGS = {
    'v8m': {'weights': 'yolov8m.pt', 'epochs': 50, 'batch': 12, 'imgsz': 832, 'name': 'train_v8m'},
    'v8s': {'weights': 'yolov8s.pt', 'epochs': 60, 'batch': 16, 'imgsz': 832, 'name': 'train_v8s'},
    'v8n': {'weights': 'yolov8n.pt', 'epochs': 60, 'batch': 64, 'imgsz': 640, 'name': 'train_v8n'},
}


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

    # Train/val split on unique image IDs, stratified by each image's
    # dominant class so val has the same class balance as train
    img_cls = train.groupby('Image_ID')['class_id'].agg(lambda s: s.mode()[0])
    train_names, val_names = train_test_split(
        img_cls.index.values, test_size=0.15, random_state=SEED,
        stratify=img_cls.values)
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
def train_model(config_key='v8m'):
    """Train a single YOLO variant. config_key chooses MODEL_CONFIGS."""
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    cfg = MODEL_CONFIGS[config_key]
    print(f"Training {config_key} on device: {device}", cfg)

    model = YOLO(cfg['weights'])
    model.train(
        data=str(YAML_PATH),
        epochs=cfg['epochs'],
        imgsz=cfg.get('imgsz', 832),
        batch=cfg['batch'],
        device=device,
        patience=20,
        seed=SEED,
        cache='ram',
        workers=0,   # Windows-stable: avoid dataloader multiprocessing deadlocks
        cos_lr=True,
        close_mosaic=15,
        mixup=0.10,
        copy_paste=0.10,
        project=str(RUNS_DIR / 'detect'),
        name=cfg['name'],
        exist_ok=True,
    )

    # Validate best weights
    best = RUNS_DIR / 'detect' / cfg['name'] / 'weights' / 'best.pt'
    model = YOLO(best)
    model.val()


# ---------------------------------------------------------------- infer helpers ----
def predict_one(model, image_path, imgsz=1024, augment=True, conf=0.01, iou=0.6):
    """Run a single model on one image and return list of dict predictions."""
    results = model(
        str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=300,
        augment=augment,
        verbose=False,
    )
    r = results[0]
    preds = []
    if r.boxes and len(r.boxes):
        for box, cls, confv in zip(r.boxes.xyxy.tolist(),
                                   r.boxes.cls.tolist(),
                                   r.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            preds.append({
                'class_id': int(cls),
                'class': r.names[int(cls)],
                'confidence': float(confv),
                'ymin': round(y1), 'xmin': round(x1),
                'ymax': round(y2), 'xmax': round(x2),
            })
    return preds


def boxes_to_normalized(preds, w, h):
    """Convert absolute xyxy predictions to normalized xyxy + labels + scores."""
    boxes, scores, labels = [], [], []
    for p in preds:
        x1, y1, x2, y2 = p['xmin'], p['ymin'], p['xmax'], p['ymax']
        boxes.append([max(0, x1 / w), max(0, y1 / h),
                      min(1, x2 / w), min(1, y2 / h)])
        scores.append(p['confidence'])
        labels.append(p['class_id'])
    return boxes, scores, labels


def normalized_to_rows(boxes, labels, scores, image_file):
    rows = []
    for (x1, y1, x2, y2), lbl, confv in zip(boxes, labels, scores):
        rows.append({
            'Image_ID': image_file,
            'class': CLASS_NAMES[int(lbl)],
            'confidence': float(confv),
            'ymin': round(y1), 'xmin': round(x1),
            'ymax': round(y2), 'xmax': round(x2),
        })
    return rows


# ---------------------------------------------------------------- infer ----
def infer():
    model = YOLO(BEST_WEIGHTS)
    image_files = sorted(os.listdir(TEST_IMAGES_DIR))

    all_data = []
    for image_file in tqdm(image_files, desc='Predicting'):
        preds = predict_one(model, TEST_IMAGES_DIR / image_file,
                            imgsz=1024, augment=True)
        if preds:
            for p in preds:
                row = {k: v for k, v in p.items() if k != 'class_id'}
                row['Image_ID'] = image_file
                all_data.append(row)
        else:
            all_data.append({
                'Image_ID': image_file, 'class': 'healthy', 'confidence': 0.01,
                'ymin': 100, 'xmin': 100, 'ymax': 100, 'xmax': 100,
            })

    sub = pd.DataFrame(all_data)
    sub.to_csv(SUBMISSION_PATH, index=False)
    print('Saved', SUBMISSION_PATH)
    print(sub['class'].value_counts())


# ---------------------------------------------------------- ensemble infer ----
def infer_ensemble():
    """
    Ensemble multiple trained models with multi-scale TTA + Weighted Boxes Fusion.
    Expected checkpoints (train them first with the commands below):
        runs/detect/train_v8m/weights/best.pt
        runs/detect/train_v8s/weights/best.pt
        runs/detect/train_v8n/weights/best.pt   (optional)
    """
    if not WBF_AVAILABLE:
        raise RuntimeError('ensemble_boxes is required for WBF. Install: pip install ensemble-boxes')

    model_paths = [
        RUNS_DIR / 'detect' / 'train_v8m' / 'weights' / 'best.pt',
        RUNS_DIR / 'detect' / 'train_v8s' / 'weights' / 'best.pt',
        RUNS_DIR / 'detect' / 'train_v8n' / 'weights' / 'best.pt',
        RUNS_DIR / 'detect' / 'train' / 'weights' / 'best.pt',  # legacy v8s run
    ]
    model_paths = [p for p in model_paths if p.exists()]
    if len(model_paths) < 2:
        raise FileNotFoundError(
            'Need at least two trained models for ensemble. '
            'Run: python app.py train_v8m  (and/or train_v8s, train_v8n)'
        )

    models = [YOLO(str(p)) for p in model_paths]
    print(f'Ensembling {len(models)} models: {[p.name for p in model_paths]}')

    # Multi-scale TTA sizes. 736/800/1024/1216 covers a wider scale range
    # without blowing up runtime too much.
    scales = [736, 800, 1024, 1216]
    conf = 0.002   # very low: mAP ranks by confidence; WBF cleans up noise
    iou = 0.5
    weights = [2.0] * len(models)   # equal model weighting; increase for best single model

    image_files = sorted(os.listdir(TEST_IMAGES_DIR))
    all_data = []

    for image_file in tqdm(image_files, desc='Ensemble predicting'):
        img_path = TEST_IMAGES_DIR / image_file
        img = Image.open(img_path)
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []

        for model in models:
            for scale in scales:
                preds = predict_one(model, img_path, imgsz=scale,
                                    augment=True, conf=conf, iou=iou)
                if preds:
                    b, s, l = boxes_to_normalized(preds, w, h)
                    all_boxes.append(b)
                    all_scores.append(s)
                    all_labels.append(l)

        if all_boxes:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                weights=weights * len(scales),
                iou_thr=0.55,
                skip_box_thr=0.001,
            )
            # Convert normalized -> absolute pixel coords for submission
            fused_boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h]
                           for b in fused_boxes]
            rows = normalized_to_rows(fused_boxes, fused_labels, fused_scores, image_file)
            all_data.extend(rows)
        else:
            all_data.append({
                'Image_ID': image_file, 'class': 'healthy', 'confidence': 0.01,
                'ymin': 100, 'xmin': 100, 'ymax': 100, 'xmax': 100,
            })

    sub = pd.DataFrame(all_data)
    sub = ensure_all_ids(sub)
    sub.to_csv(SUBMISSION_PATH, index=False)
    print('Saved', SUBMISSION_PATH)
    print(sub['class'].value_counts())


def ensure_all_ids(sub_df):
    """Zindi scorer errors if any test Image_ID is missing."""
    test_ids = {f for f in os.listdir(TEST_IMAGES_DIR)}
    missing = test_ids - set(sub_df['Image_ID'])
    if missing:
        rows = [{
            'Image_ID': mid, 'class': 'healthy', 'confidence': 0.01,
            'ymin': 100, 'xmin': 100, 'ymax': 100, 'xmax': 100,
        } for mid in missing]
        sub_df = pd.concat([sub_df, pd.DataFrame(rows)], ignore_index=True)
    return sub_df


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
    if step == 'train':
        # Default to the strongest available model
        train_model('v8m')
    if step == 'train_v8m':
        train_model('v8m')
    if step == 'train_v8s':
        train_model('v8s')
    if step == 'train_v8n':
        train_model('v8n')
    if step == 'train_all':
        for key in ['v8m', 'v8s', 'v8n']:
            train_model(key)
    if step in ('infer', 'all'):
        infer()
    if step == 'infer_ensemble':
        infer_ensemble()
    if step == 'plot':
        plot_samples()
