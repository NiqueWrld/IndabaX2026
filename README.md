# Cocoa Contamination Hackathon — Issues & Solutions Log

**Competition:** IndabaX South Africa Cocoa Contamination Hackathon (Zindi)
**Task:** Object detection on cocoa leaf images — 3 classes (`anthracnose`, `cssvd`, `healthy`), metric mAP@IoU 0.5
**Final result:** #2 on the private leaderboard, score **0.763846699** (submission ID `JzAfjwx4`)
- Public best: 0.7539
- Model: YOLOv8s trained at 1024×1024, single-scale TTA inference at confidence 0.001

---

## Pipeline

Everything runs from a single reproducible script, [app.py](app.py):

```
python app.py prep     # train/val split (seed 42) + data.yaml
python app.py train    # fine-tune YOLOv8 (seed 42)
python app.py infer    # predict test set -> submission CSV
python app.py all      # all of the above
```

Hardware: Ryzen 9 9900X, 48 GB RAM, RTX A4000 16 GB (CUDA, torch 2.6.0+cu124).

---

## Issues found & how we solved them

### 1. Starter notebook was Colab-only
**Symptom:** `ModuleNotFoundError: No module named 'google.colab'` when running locally.
**Fix:** Converted the whole notebook into a plain Python script (`app.py`) with local
paths, removing Drive mounts, `!pip` cells and `/content/...` paths.
Side note: the 19 MB starter notebook could not be edited reliably in place —
cell edits silently reverted, so the script conversion was also the practical fix.

### 2. Training outputs written to the wrong folder
**Symptom:** Ultralytics saved runs to a stale global path (`...\R3-1\runs\detect`)
left over from a previous project, so inference could not find `best.pt`.
**Fix:** Pin the output location explicitly with `project=runs/detect, name=train,
exist_ok=True` in every `model.train()` call.

### 3. Slow training / caching stalls
**Symptom:** First runs underused the GPU (batch 8 used only 3.4/16 GB VRAM);
`cache=True` (RAM) with many dataloader workers duplicated memory on Windows and
appeared to hang.
**Fix:** Increased batch size to fill VRAM, switched to `cache='disk'`, kept
`workers=8`. Epoch time dropped to ~1 min at 1024 px on the A4000.

### 4. Submission scored exactly 0.000 (the big one)
**Symptom:** Perfectly formatted submissions scored `0.000000` on Zindi.
Five other participants were stuck at 0.0 too.

**What we ruled out** (all verified programmatically):
- Column names/order vs `SampleSubmission.csv` — identical
- All 1626 test `Image_ID`s present with exact case (incl. 81 uppercase `.JPG`)
- Coordinate scale — same pixel space as `Train.csv` ground truth
- Stale test set — re-downloaded `SampleSubmission.csv` was byte-identical
- File encoding — no BOM, clean line endings, valid floats, no inverted boxes

**Root cause (two-part):**
1. The starter notebook writes placeholder rows for images with no detections:
   `class="None"` with empty confidence/coordinates. Zindi's scorer cannot parse
   these and the whole submission scores **0.000** (not an error!).
2. Dropping those rows instead is also wrong — the scorer **requires every test
   ID to be present** and rejects the file with
   `Error. Image ID ID_xxxxxx.JPG not found in submission file`.

**Fix:** every no-detection image gets one *valid* low-confidence dummy row,
mirroring the sample file's shape:

```
Image_ID, class=healthy, confidence=0.01, ymin=100, xmin=100, ymax=100, xmax=100
```

Score went from 0.000 → 0.558 with the identical model.

### 5. Climbing the leaderboard
| Change | Local val mAP50 | Public score |
|---|---|---|
| YOLOv8n, 10 epochs, 1024 px | 0.666 | 0.558 |
| YOLOv8s, 40 epochs, 1024 px | 0.723 | 0.667 |
| + test-time augmentation (`augment=True` at predict) | — | **0.698 (rank 3)** |
| + lower confidence threshold (0.001) | — | 0.7535 |
| conf threshold 0.002 | — | 0.7539 |
| **Private leaderboard (JzAfjwx4)** | — | **0.7638 (#2)** |

Key insight: the competition metric is mAP@IoU 0.5, which ranks by confidence.
Submitting more low-confidence boxes (down to 0.001) improved the public score
materially over the default 0.25 threshold.

TTA (multi-scale + flip) was a cheap early win; the biggest gain came from
tuning the inference confidence threshold.

---

## Mobile app bonus

A React + Capacitor Android demo is included in [`mobile_app/`](mobile_app/):

- Loads an ONNX export of the trained model
- Uses the device camera to capture cocoa leaves
- Runs inference in the app and displays detected classes

See [`mobile_app/README.md`](mobile_app/README.md) for build instructions.

---

## Submission-format checklist (for future Zindi detection comps)

- [ ] Columns exactly: `Image_ID,class,confidence,ymin,xmin,ymax,xmax`
- [ ] **Every** test ID present, exact case-sensitive filename incl. extension
- [ ] No NaN / empty / `"None"` values anywhere
- [ ] One valid dummy row for images with no detections
- [ ] Raw (unthresholded) confidences — required by Zindi rules
- [ ] Pixel coordinates of the original image size, `min < max`
- [ ] Fixed seeds everywhere (`seed=42`) for reproducible code review
