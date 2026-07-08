# Cocoa Contamination Hackathon — Object Detection

Pipeline for the Zindi IndabaX SA Cocoa Contamination Hackathon (mAP@0.5, T4/9h limit).

## Data
- `data/Train.csv` — boxes: `ymin,xmin,ymax,xmax` (pixels), classes: `anthracnose`, `cssvd`, `healthy`
- `data/Test.csv` — 1626 test images
- `data/dataset/images/{train,test}/` — extracted from `dataset.zip` (~9.4 GB)

## Workflow

1. Extract the images zip so images land in `data/dataset/images/...`.
2. Prepare YOLO labels + split:
   ```
   python prepare_yolo.py
   ```
3. Train (on Kaggle/Colab **T4** — matches the competition limit):
   ```
   pip install -r requirements.txt
   python train.py --model yolo11s.pt --imgsz 640 --epochs 100
   ```
4. Build submission:
   ```
   python infer.py --weights runs/detect/cocoa_yolo11s/weights/best.pt
   ```

## Notes
- Local machine (Ryzen 9 9900X, 48 GB, no active CUDA GPU) → use for **data prep/EDA**; train on Kaggle/Colab T4.
- `seed=42` set everywhere for reproducible leaderboard scores (competition rule).
- ONNX export runs automatically (edge-device requirement).
- Inference uses low `conf` and keeps **raw** confidences — do not threshold (rule).
