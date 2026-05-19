# Sign Language Recognition — MediaPipe + GRU

Real-time Indian/American Sign Language recognition using **MediaPipe Hand Landmarks** and a **GRU neural network**. Supports both left and right hands simultaneously with mirror normalization.

## Recognized Signs
`Hello` · `ILoveYou` · `No` · `Please` · `Thanks` · `Yes`

## Architecture
```
Webcam → MediaPipe Hand Landmarker (21 × 3D keypoints)
       → Wrist-relative normalization + left-hand X mirroring
       → 15-frame sliding window
       → Stacked GRU (128 → 64) + Dropout + BatchNorm
       → Softmax (6 classes)
```

## Setup

```bash
git clone https://github.com/<your-username>/Sign-Language-Recognition.git
cd Sign-Language-Recognition
pip install -r requirements.txt
```

## Usage

### Step 1 — Collect your own landmark data
```bash
python collect_data.py
```
- Hold each sign in front of the camera
- Press **SPACE** to start recording, **Q** to move to next class
- Works with **both hands** (left-hand X is auto-mirrored)

### Step 2 — Train the GRU model
```bash
python train_model.py
```

### Step 3 — Run real-time inference
```bash
python run.py
```

## Key Features
- **Mirror normalization** — left and right hand produce identical features
- **Dual-hand tracking** — independent inference per hand, shown simultaneously
- **Temporal windowing** — 15-frame GRU captures motion, not just static poses
- **Spatial filtering** — ignores hands covering less than 1.5% of frame area
- **Stable predictions** — hold-frame logic prevents flickering labels
- **Dynamic prioritization** — in single-output mode, prefers hand closest to frame center

## Requirements
See `requirements.txt`

## Project Structure
```
├── collect_data.py      # landmark data collection
├── train_model.py       # GRU model training
├── run.py               # real-time inference
├── requirements.txt
├── data/                # generated after collection + training
│   ├── landmarks.csv
│   ├── sign_model.keras
│   └── label_classes.npy
└── yolov5/              # YOLOv5 (used for original best.pt weights)
```
