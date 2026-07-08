"""
Run inference on the test set and build a Zindi submission CSV.

Submission format (from the competition):
  Image_ID,class,confidence,ymin,xmin,ymax,xmax
  ID_xxx.jpg,healthy,0.87,130,12,340,300

Every test image must appear at least once. Images with no detections get a
single low-confidence placeholder row so they are still represented.

Usage:
  python infer.py --weights runs/detect/cocoa_yolo11s/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
CLASSES = ["anthracnose", "cssvd", "healthy"]


def read_test_ids(test_csv: Path) -> list[str]:
    with open(test_csv, newline="") as f:
        return [row["Image_ID"] for row in csv.DictReader(f)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--test-csv", default=str(ROOT / "data" / "Test.csv"))
    p.add_argument("--img-dir", default=str(ROOT / "data" / "images" / "test"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05,   # keep low-ish; do NOT hard-threshold
                   help="confidence floor; raw scores preserved for mAP")
    p.add_argument("--max-det", type=int, default=100, help="max boxes per image")
    p.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = p.parse_args()

    model = YOLO(args.weights)
    img_dir = Path(args.img_dir)
    test_ids = read_test_ids(Path(args.test_csv))

    rows: list[list] = []
    seen: set[str] = set()

    for img_id in test_ids:
        path = img_dir / img_id
        if not path.exists():
            continue
        r = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf,
                          max_det=args.max_det, verbose=False, augment=True)[0]
        has_box = False
        for box in r.boxes:
            cls = CLASSES[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            # submission order: ymin,xmin,ymax,xmax
            rows.append([img_id, cls, round(conf, 6),
                         round(y1), round(x1), round(y2), round(x2)])
            has_box = True
        if has_box:
            seen.add(img_id)

    # placeholder for images with zero detections
    for img_id in test_ids:
        if img_id not in seen:
            rows.append([img_id, "healthy", 0.01, 0, 0, 1, 1])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"])
        w.writerows(rows)

    print(f"wrote {len(rows)} rows for {len(test_ids)} test images -> {args.out}")


if __name__ == "__main__":
    main()
