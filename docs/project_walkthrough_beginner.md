# Project Walkthrough (Beginner, Step-by-Step)

This guide explains the whole repository in simple terms, assuming you are new to coding. It also explains what each code file does and how data moves through the system.

## 1) What this project is
This project recognizes isolated sign language words from videos. It turns each video into a sequence of body and hand keypoints (numbers), then trains deep-learning models to classify the sign.

**High-level pipeline:**
1. Organize videos into class folders.
2. Extract skeleton keypoints from videos.
3. Clean and normalize the keypoints.
4. Resample all videos to the same number of frames.
5. Train models on the processed data.
6. Evaluate and compare models.

## 2) Folder map in plain words
- `dataset/`: raw video files, organized by class names (one folder per sign).
- `data/`: helper files and pre-trained keypoint extractor models.
- `preprocessing/`: scripts to extract and clean keypoints.
- `models/`: neural network model definitions.
- `training/`: training code (data loaders, training loop).
- `evaluation/`: evaluation and inference code.
- `output/`: processed arrays, cached results, charts.
- `docs/`: documentation (you are reading one of these).

## 3) Data shape glossary (simple)
These shapes appear everywhere:
- `N`: number of samples (videos).
- `T`: number of frames in a video (fixed to 64 after resampling).
- `V`: number of joints (51 after filtering).
- `C`: number of channels (2: x and y).
- `F`: number of features per frame (2 * V).

Common arrays:
- `X_raw.npy`: `(N, T, 150)` raw keypoints (pose + hands).
- `X_normalized.npy`: `(N, T, 102)` normalized 51-joint stream.
- `X_final.npy`: `(N, 64, 102)` normalized and resampled joint stream.
- `X_bones.npy`: `(N, 64, 102)` bone vectors.
- `X_motion.npy`: `(N, 64, 102)` joint motion (frame-to-frame).
- `X_bone_motion.npy`: `(N, 64, 102)` bone motion.
- `y.npy`: `(N,)` integer class labels.

## 4) Step-by-step: how you use the project

### Step A) Organize dataset
File: `data/organize_dataset.py`

**Goal:** Move or copy videos into folders by label (class).

**What the code does (simple):**
- Loads the WLASL JSON file that lists all videos and labels.
- Optionally selects only the top K classes (for small experiments).
- For each video id, it copies/moves the `.mp4` into `dataset/<label>/`.
- Prints a summary of how many files were copied, skipped, or filtered.

**Why it matters:**
The extractor expects a folder per class. Without this, you cannot build the dataset.

### Step B) Extract keypoints from videos
File: `preprocessing/extract.py`

**Goal:** Convert video frames into numeric keypoints.

**Two backends:**
- MediaPipe (CPU): pose + both hands.
- YOLO (GPU or CPU): body + both hands.

**What happens in the code:**
1. The script checks the backend and downloads models if needed.
2. It lists all class folders in `dataset/`.
3. For each video, it reads frames and extracts keypoints.
4. Each video becomes a `(max_frames, feature_size)` array.
5. It saves `X_raw.npy` and `y.npy` in `output/`.

**Important functions in this file:**
- `resolve_device`: chooses CPU or GPU and compatible backend.
- `download_mp_models` / `download_yolo_models`: ensure models exist.
- `extract_mp`: MediaPipe extraction (150 features per frame).
- `extract_yolo`: YOLO extraction (118 features per frame).
- `build_dataset`: main loop that produces `X_raw.npy` and `label_map.json`.

### Step C) Clean + normalize + compute extra streams
File: `preprocessing/normalize.py`

**Goal:** Make the data consistent and create extra “streams”.

**What happens in the code:**
1. `clean_missing_keypoints`: fills missing joints by interpolation.
2. `filter_joints`: keeps only upper body + both hands (51 joints).
3. `normalize_skeleton`:
   - Centers the body (move to origin).
   - Scales by a body distance so sizes are consistent.
   - Makes hands relative to their wrist roots.
4. `smooth_dataset`: optional smoothing along time.
5. `compute_bone_vectors`: convert joints to bone vectors.
6. `compute_motion`: difference between consecutive frames.

**Outputs saved:**
- `X_normalized.npy`, `X_bones.npy`, `X_motion.npy`, `X_bone_motion.npy`.

### Step D) Resample to fixed length
File: `preprocessing/resample.py`

**Goal:** Make every video have exactly 64 frames.

**What happens in the code:**
- `valid_length` detects how many real (non-zero) frames a sequence has.
- `temporal_resample` linearly interpolates frames to length 64.
- Resampled outputs are saved. For joint stream: `X_final.npy`.

### Step E) Train models
Two main entry points:
- `training/train.py` (specialized 4-stream training)
- `main.py` (5-model comparison runner)

#### 1) `training/train.py`
**Goal:** Train the 4-stream early-fusion ST-GCN (best model).

**What happens:**
- Loads all four streams with `load_data`.
- Splits into train/val/test.
- Runs K-fold training with early stopping.
- Uses SWA (stochastic weight averaging) at the end.
- Saves `models/sign_stgcn.pth`.

**Key helpers inside:**
- `LabelSmoothingLoss`: makes training less overconfident.
- `augment_skeleton`: applies many random transforms to data.
- `FourStreamDataset`: returns four streams per sample.
- `train_fold`: a full training loop for one fold.

#### 2) `main.py`
**Goal:** Compare multiple models and produce charts.

**What happens:**
- Loads data streams.
- Runs one or more models (3-stream, 2-stream, 4-stream).
- Does K-fold training with shared logic in `_run_kfold`.
- Generates charts for accuracy and confusion matrices.
- Caches results in `output/model_results.json`.

**Key sections in `main.py`:**
- Dataset classes for 2/3/4 streams.
- Training engine (`train_one_fold` + `_run_kfold`).
- Model runners (`run_multi_stream_stgcn`, `run_4stream_late_fusion`, `run_4stream_fusion`).
- Chart creation (`plot_model_results`, `plot_comparison_overview`).

### Step F) Evaluate or run inference
File: `evaluation/evaluate.py`

**Goal:** Evaluate the LSTM/TCN model or predict a single video.

**What happens:**
- Loads `models/sign_lstm.pth`.
- Converts a video to keypoints (same pipeline as training).
- Runs the model and prints the predicted label.
- Or evaluates the whole dataset and prints a classification report.

## 5) Models folder explained

### `models/graph.py`
Defines the skeleton graph (nodes and edges). It creates adjacency matrices for the ST-GCN models. It also supports:
- `AdaptiveAdjacency`: learnable edges added to the fixed graph.
- `drop_graph`: randomly drops joints during training (regularization).

### `models/tgcn.py`
Implements the basic graph convolution used inside ST-GCN. It does a temporal convolution and then multiplies by the adjacency matrix to mix information between joints.

### `models/st_gcn.py`
The improved ST-GCN backbone used by the 2-stream and 4-stream models.
- Builds the graph and adjacency.
- Applies DropGraph.
- Uses 10 ST-GCN blocks with reduced channel sizes.
- Can return either logits or features (for early fusion).

### `models/stgcn.py`
A simpler, custom 3-stream ST-GCN with its own fixed adjacency.
Each stream is independent and their logits are summed.

### `models/stgcn_2stream_ported.py`
A legacy port of the original two-stream ST-GCN. It runs joint and bone streams and sums their logits.

### `models/st_gcn_twostream.py`
The four-stream model architecture. It supports two modes:
- **Late fusion:** Runs 4 streams independently and sums the final logits (the `4stream-late-fusion` model).
- **Early fusion:** Concatenates 128-dimensional features from 4 streams before a shared dense classifier (the `4stream-fusion` model).

### `models/lstm.py`
Despite the name, this is a TCN + attention model used for an older baseline. It does:
1. Input projection.
2. Four temporal convolution blocks with dilation.
3. Attention pooling across time.
4. Final classifier.

## 6) How the data becomes a prediction (simple story)
1. A video is read frame by frame.
2. Each frame becomes a list of keypoints (x, y values).
3. The keypoints are cleaned, normalized, and resampled.
4. The processed data is passed to a model.
5. The model outputs logits (scores for each class).
6. The highest score is the predicted sign.

## 7) What to read next
If you want a very detailed, line-by-line explanation of each model, use:
- `docs/multi_stream_stgcn_walkthrough.md`
- `docs/4stream_late_fusion_walkthrough.md`
- `docs/4stream_fusion_walkthrough.md`
