# Sign Language Recognition — WSASL

A skeleton-based isolated sign language recognition system using **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**. This project extracts body and hand keypoints from sign language videos, normalizes the skeleton data, and trains graph-based models to classify signs.

---

## Project Structure

```
.
├── main.py                  # Multi-model comparison runner + chart generation
├── preprocessing/           # Video → skeleton keypoint pipeline
│   ├── README.md            # Preprocessing module documentation
│   ├── extract.py           # Step 1: Extract keypoints (MediaPipe or YOLO)
│   ├── normalize.py         # Step 2: Normalize + compute bone/motion streams
│   └── resample.py          # Step 3: Resample all streams to 64 frames
├── training/                # Model training
│   ├── README.md            # Training pipeline documentation
│   ├── dataset.py           # Dataset class + data augmentation
│   └── train.py             # Four-stream ST-GCN training loop
├── models/                  # Neural network architectures
│   ├── README.md            # Models & architecture documentation
│   ├── graph.py             # Skeleton graph, adaptive adjacency, DropGraph
│   ├── tgcn.py              # Core graph convolution (ConvTemporalGraphical)
│   ├── st_gcn.py            # Single-stream ST-GCN backbone
│   ├── lstm.py              # TCN + Temporal Attention (SignLSTM)
│   ├── stgcn.py             # Three-stream ST-GCN (late fusion)
│   ├── st_gcn_twostream.py  # Four-stream ST-GCN with early/late fusion
│   └── stgcn_2stream_ported.py  # Two-stream ST-GCN port
├── evaluation/              # Evaluation and inference
│   ├── README.md            # Evaluation module documentation
│   └── evaluate.py          # Classification report + single-video inference
├── docs/                    # Architecture documentation
├── dataset/                 # Input videos organized by class folder
├── output/                  # Processed .npy arrays + results + charts
└── data/                    # Model weight caches (MediaPipe .task, YOLO .pt)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Organize Dataset

Place sign language videos in:

```
dataset/
├── hello/
│   ├── video1.mp4
│   └── video2.mp4
├── thank_you/
│   └── ...
└── ...
```

### 3. Run Preprocessing Pipeline

```bash
# Step 1: Extract skeleton keypoints
python preprocessing/extract.py

# Step 2: Normalize + compute bone/motion streams
python preprocessing/normalize.py

# Step 3: Resample to 64 frames
python preprocessing/resample.py
```

### 4. Train

```bash
# Train the four-stream ST-GCN (best model)
python training/train.py

# Or run all three comparison models
python main.py --compare-5
```

### 5. Evaluate

```bash
# Full dataset evaluation (classification report)
python evaluation/evaluate.py

# Single video inference
python evaluation/evaluate.py --video path/to/video.mp4
```

---

## Models

| Model                  | Streams                           | Fusion    | Description              |
| ---------------------- | --------------------------------- | --------- | ------------------------ |
| `SignLSTM` (TCN)       | 1 (joint)                         | —         | TCN + Temporal Attention |
| `STGCN`                | 3 (joint+bone+motion)             | Late      | Three-stream ST-GCN      |
| `TwoStreamPortedSTGCN` | 2 (joint+bone)                    | Late      | Ported Yan et al. 2018   |
| `FourStreamSTGCN`      | 4 (joint+motion+bone+bone_motion) | **Early** | **Current best**         |

---

## Feature Streams

After preprocessing, four parallel data streams are available:

| File                | Description                 | Shape          |
| ------------------- | --------------------------- | -------------- |
| `X_final.npy`       | Normalized joint positions  | `(N, 64, 102)` |
| `X_bones.npy`       | Bone vectors (child−parent) | `(N, 64, 102)` |
| `X_motion.npy`      | Frame-to-frame joint delta  | `(N, 64, 102)` |
| `X_bone_motion.npy` | Frame-to-frame bone delta   | `(N, 64, 102)` |

Each sample is **51 joints × 2 (x,y) = 102 features** per frame, for **64 frames**.

---

## Skeleton Layout (51 joints)

```
Joints 0–8 : Upper body
    0 = nose
    1,2 = left/right shoulder
    3,4 = left/right elbow
    5,6 = left/right wrist
    7,8 = left/right hip

Joints 9–29 : Left hand (wrist root = 9)
Joints 30–50: Right hand (wrist root = 30)
```

---

## Extraction Backends

| Backend             | Library       | GPU         | Features                   |
| ------------------- | ------------- | ----------- | -------------------------- |
| MediaPipe (default) | `mediapipe`   | ❌ CPU only | 150 (33 pose + 21+21 hand) |
| YOLO                | `ultralytics` | ✅ CUDA     | 118 (17 body + 21+21 hand) |

```bash
# Use YOLO with GPU
python preprocessing/extract.py --backend yolo --device cuda
```

---

## main.py — Multi-Model Comparison

```bash
python main.py --multi-stream-stgcn   # 3-stream ST-GCN
python main.py --2stream-stgcn        # 2-stream ST-GCN (ported)
python main.py --4stream-fusion       # 4-stream early fusion
python main.py --compare-5            # run all models
python main.py --compare-5 --results-only  # charts from cache only
python main.py --compare-5 --epochs 100    # shorter run
```

Charts are saved to `output/charts/`. Results are cached in `output/model_results.json`.

## Generating Charts From Cached Results

If you already have per-model JSON result files in `output/` (for example after running `main.py` or training scripts), you can generate the same per-model 4-panel charts and the comparison overview using the lightweight script at `evaluation/generate_charts.py`.

Required JSON files (place in `output/`):

- `results_3stream.json` — Multi-stream ST-GCN (3-stream)
- `results_4stream_early.json` — 4-stream early fusion
- `results_4stream_late.json` — 4-stream late fusion

Run:

```bash
python evaluation/generate_charts.py
```

Output (saved to `output/charts/`):

- `multi_stream_stgcn_results.png`
- `4stream_fusion_results.png`
- `4stream_late_fusion_results.png`
- `comparison_overview.png`

The script will also produce per-model confusion matrix PNGs and training-history PNGs if the JSON files include the required arrays (`all_preds`, `all_labels`, and `fold_histories`).

---

## Documentation

We provide extensive documentation for both the project components and the model architectures.

### Component Documentation

Each major module contains its own `README.md` detailing its implementation and usage:

- [**Models** (`models/README.md`)](models/README.md) - Neural network architectures, graph formulations, and fusion strategies.
- [**Preprocessing** (`preprocessing/README.md`)](preprocessing/README.md) - Video extraction, normalization, and stream generation.
- [**Training** (`training/README.md`)](training/README.md) - Dataset handling, augmentations, and training loops.
- [**Evaluation** (`evaluation/README.md`)](evaluation/README.md) - Inference scripts, metrics, and visualization.

### GUI Demo

We provide a simple Streamlit GUI for running single-video inference, dataset-video inference, or dataset-sample inference.

- [**GUI Guide** (`docs/gui.md`)](docs/gui.md) — how to run the GUI and required inputs.

### Architecture Walkthroughs

See `docs/` for detailed architecture walkthroughs:

| File                                | Description                            |
| ----------------------------------- | -------------------------------------- |
| `2stream_stgcn.md`                  | Two-stream ST-GCN architecture         |
| `2stream_stgcn_walkthrough.md`      | Two-stream training walkthrough        |
| `4stream_fusion.md`                 | Four-stream early fusion architecture  |
| `4stream_fusion_walkthrough.md`     | Four-stream training walkthrough       |
| `multi_stream_stgcn.md`             | Three-stream ST-GCN architecture       |
| `multi_stream_stgcn_walkthrough.md` | Three-stream training walkthrough      |
| `project_walkthrough_beginner.md`   | Full project walkthrough for beginners |
