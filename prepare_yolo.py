"""
Build a train/val split for the Cocoa Contamination hackathon using the
YOLO labels that already ship inside dataset.zip.

Input (already extracted):
  data/images/train/*.jpg     training images (5529)
  data/labels/train/*.txt     YOLO labels (class cx cy w h, normalised)
  data/images/test/*.jpg      test images (1626)

Output:
  yolo/images/{train,val}     images (hard-linked if possible, else copied)
  yolo/labels/{train,val}     matching labels
  yolo/cocoa.yaml             Ultralytics dataset config

Run:  python prepare_yolo.py
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC_IMG = ROOT / "data" / "images" / "train"
SRC_LBL = ROOT / "data" / "labels" / "train"
OUT = ROOT / "yolo"
VAL_FRACTION = 0.15
SEED = 42

# class index order matches the provided labels (0=anthracnose, 1=cssvd, 2=healthy)
CLASSES = ["anthracnose", "cssvd", "healthy"]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hard link: no extra disk, instant
    except OSError:
        shutil.copy2(src, dst)


def find_image(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = SRC_IMG / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    if not SRC_LBL.exists():
        raise SystemExit(f"Labels not found at {SRC_LBL}. Extract dataset.zip into data/ first.")

    # only keep images that have a matching label file
    stems = sorted(p.stem for p in SRC_LBL.glob("*.txt"))
    random.Random(SEED).shuffle(stems)
    n_val = int(len(stems) * VAL_FRACTION)
    splits = {"val": stems[:n_val], "train": stems[n_val:]}

    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for split, ids in splits.items():
        for stem in ids:
            img = find_image(stem)
            lbl = SRC_LBL / f"{stem}.txt"
            if img is None or not lbl.exists():
                continue
            link_or_copy(img, OUT / "images" / split / img.name)
            link_or_copy(lbl, OUT / "labels" / split / lbl.name)
            counts[split] += 1

    (OUT / "cocoa.yaml").write_text(
        f"path: {OUT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    print(f"train images: {counts['train']}  val images: {counts['val']}")
    print(f"classes: {CLASSES}")
    print(f"wrote config: {OUT / 'cocoa.yaml'}")


if __name__ == "__main__":
    main()
