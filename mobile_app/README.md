# Cocoa Disease Detector — Mobile App (Bonus)

A React + Capacitor mobile demo that runs the trained YOLOv8 model on-device to detect cocoa leaf diseases.

## What it does

- Take a photo of a cocoa leaf using the device camera
- Run inference with an ONNX-exported YOLOv8 model
- Display detected diseases: **anthracnose**, **cssvd**, or **healthy**

## Project structure

```
mobile_app/
├── public/
│   └── cocoa-disease.onnx          # exported model (already included)
├── scripts/
│   └── export_onnx.py              # re-export best.pt to ONNX
├── src/
│   ├── App.tsx                     # camera + inference UI
│   ├── index.css                   # mobile-first styles
│   └── main.tsx                    # React entry point
├── capacitor.config.ts             # Capacitor configuration
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Quick start (web preview)

```bash
cd mobile_app
npm install
npm run dev
```

Then open the local URL in your browser. The web preview can use a file input instead of the device camera.

## Build the Android app

### Prerequisites

- Node.js 18+
- npm
- Android Studio
- Android SDK

### Steps

1. Install dependencies and build the web app:

```bash
cd mobile_app
npm install
npm run build
```

2. Add the Android platform (only needed once):

```bash
npx cap add android
```

3. Copy the built web assets into the Android project:

```bash
npx cap sync
```

4. Open Android Studio:

```bash
npx cap open android
```

5. Build and run on a device or emulator from Android Studio.

## Re-export the model

If you retrain the model, update the mobile ONNX file:

```bash
cd mobile_app
npm run export-onnx
```

This runs `scripts/export_onnx.py`, which loads `runs/detect/train/weights/best.pt` and writes a new `public/cocoa-disease.onnx`.

## Notes on model size and performance

- The included model (`cocoa-disease.onnx`) is a YOLOv8s exported at 1024×1024 resolution and is about **43 MB**.
- For production deployment on low-end smartphones, consider:
  - Switching to **YOLOv8n** (nano) for a smaller, faster model
  - Reducing input resolution to **640×640**
  - Quantizing the ONNX model to **INT8** with ONNX Runtime tools
- This demo prioritizes accuracy matching the competition submission; performance optimizations are left as future work.

## Tech stack

- React + TypeScript
- Vite
- Capacitor (Android)
- ONNX Runtime Web
- Ultralytics YOLOv8
