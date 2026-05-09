# preprocessing/

This directory contains the **three-step preprocessing pipeline** that converts raw sign language video files into clean, normalized NumPy arrays ready for model training.

The pipeline must be run **in order**: `extract.py` → `normalize.py` → `resample.py`.

---

## Pipeline Overview

```
Raw videos (dataset/<class>/*.mp4)
        │
        ▼
  [1] extract.py       →  output/X_raw.npy, output/y.npy, output/label_map.json
        │
        ▼
  [2] normalize.py     →  output/X_normalized.npy, X_bones.npy, X_motion.npy, X_bone_motion.npy
        │
        ▼
  [3] resample.py      →  output/X_final.npy (and in-place resampling of other streams)
        │
        ▼
  Ready for training (train.py / main.py)
```

---

## File Reference

| File | Step | Description |
|---|---|---|
| `extract.py` | 1 | Extract skeleton keypoints from each video frame |
| `normalize.py` | 2 | Normalize, smooth, and compute derived feature streams |
| `resample.py` | 3 | Resample all streams to a fixed length (64 frames) |

---

## extract.py — Step 1: Skeleton Keypoint Extraction

**What it does:**  
Opens each video file, runs a pose estimation model frame-by-frame, and saves the resulting keypoint arrays to disk.

**Two backend options:**

| Backend | Library | GPU support | Features per frame |
|---|---|---|---|
| MediaPipe (default) | `mediapipe` | ❌ CPU only | 150 (33 pose × 2 + 21+21 hand × 2) |
| YOLO | `ultralytics` | ✅ CUDA | 118 (17 body × 2 + 21+21 hand × 2) |

**MediaPipe feature layout (150):**
```
Indices 0–65   : 33 pose joints × 2 (x, y)
Indices 66–107 : 21 left hand joints × 2
Indices 108–149: 21 right hand joints × 2
```

**YOLO feature layout (118):**
```
Indices 0–33   : 17 body joints × 2 (COCO format)
Indices 34–75  : 21 right hand joints × 2
Indices 76–117 : 21 left hand joints × 2
```

**Usage:**
```bash
# All classes, MediaPipe, CPU (default)
python preprocessing/extract.py

# First 20 classes only (for quick testing)
python preprocessing/extract.py 20

# YOLO backend with GPU acceleration
python preprocessing/extract.py --backend yolo --device cuda

# Custom frame limit
python preprocessing/extract.py --frames 64
```

**Output files (saved to `output/`):**
- `X_raw.npy` — shape `(N, max_frames, feature_size)`, float32
- `y.npy` — shape `(N,)`, int64 class labels
- `label_map.json` — `{"class_name": index, ...}` mapping

**Key functions:**

| Function | Description |
|---|---|
| `resolve_device(preferred, backend)` | Handles MediaPipe GPU limitation (auto-switches to YOLO on CUDA) |
| `download_mp_models()` | Downloads MediaPipe .task files if not cached |
| `make_mp_detectors()` | Creates MediaPipe PoseLandmarker + HandLandmarker instances |
| `extract_mp(video, pose_det, hand_det, max_frames)` | Per-frame MediaPipe extraction |
| `download_yolo_models()` | Downloads YOLO body + hand models if not cached |
| `make_yolo_models()` | Loads YOLO body + hand models |
| `extract_yolo(video, body, hand, device, max_frames)` | Per-frame YOLO extraction |
| `build_dataset(data_dir, save_dir, ...)` | Main orchestrator: scans class folders, extracts, saves |

---

## normalize.py — Step 2: Normalization and Feature Engineering

**What it does:**  
Takes the raw `X_raw.npy` file and applies a multi-step normalization pipeline to produce clean, normalized skeleton sequences. Then derives three additional feature streams (bone vectors, joint motion, bone motion) for the multi-stream ST-GCN models.

**Processing steps:**

1. **Clean missing keypoints** (`clean_missing_keypoints`)  
   Some frames have joints set to `(0, 0)` because the detector failed. Per-joint linear temporal interpolation fills these gaps using surrounding valid frames.

2. **Filter joints** (`filter_joints`)  
   Reduces from 75 joints (150 features) to 51 joints (102 features) by discarding irrelevant lower-body joints (knees, ankles, face details), keeping only 9 upper-body joints + both hands.

3. **Normalize per frame** (`normalize_skeleton`)  
   For each frame independently:
   - **Center**: Subtract mid-shoulder point from all joints → position-invariant
   - **Scale**: Divide by shoulder width → size-invariant across different signers
   - **Relative hands**: Re-anchor each hand relative to its wrist root → hand shape only

4. **Temporal smoothing** (`smooth_skeleton`)  
   Gaussian filter along the time axis (σ=0.8 frames) removes high-frequency detector noise while preserving the overall sign motion.

5. **Bone vectors** (`compute_bone_vectors`)  
   For each bone edge (parent, child): `bone_vector = child_pos - parent_pos`  
   Captures limb orientation independently of global position.

6. **Joint motion** (`compute_motion`)  
   `motion[t] = joint[t+1] - joint[t]`  
   Captures joint velocity / direction of movement.

7. **Bone motion** (`compute_motion` applied to bone vectors)  
   `bone_motion[t] = bone[t+1] - bone[t]`  
   Captures angular velocity / rate of change of limb orientation.

**Key constants:**
- `UPPER_BODY_JOINTS`: Raw MediaPipe indices of the 9 kept upper-body joints
- `BONE_EDGES`: (parent, child) pairs defining the skeleton's bone structure
- `FILT_LEFT_HAND_OFFSET` / `FILT_RIGHT_HAND_OFFSET`: Feature indices of hand wrist roots

**Output files (saved to `output/`):**
- `X_normalized.npy` — normalized joint positions
- `X_bones.npy` — bone vectors
- `X_motion.npy` — joint velocity
- `X_bone_motion.npy` — bone angular velocity

**Usage:**
```bash
python preprocessing/normalize.py
```

---

## resample.py — Step 3: Temporal Resampling to Fixed Length

**What it does:**  
Ensures all skeleton sequences are exactly 64 frames long, regardless of original video duration. This is required because neural networks need fixed-size inputs.

**Why linear interpolation (not repeat/truncate):**  
- Truncation loses information from long videos.
- Repetition creates artificial patterns.
- Linear interpolation produces smooth, continuous sequences that preserve the temporal shape of each sign.

**Process per video:**
1. Find `valid_length`: the last non-zero frame (skips trailing zero-padding from extract.py).
2. Interpolate each feature channel independently from the valid portion to 64 frames using `np.interp`.

**Files processed:**

| Input file | Output file | Description |
|---|---|---|
| `X_normalized.npy` | `X_final.npy` | Joint positions (primary training stream) |
| `X_raw.npy` | `X_final.npy` | Fallback if X_normalized not found |
| `X_bones.npy` | `X_bones.npy` | Bone vectors (in-place) |
| `X_motion.npy` | `X_motion.npy` | Joint motion (in-place) |
| `X_bone_motion.npy` | `X_bone_motion.npy` | Bone motion (in-place) |

**Key functions:**

| Function | Description |
|---|---|
| `valid_length(skeleton)` | Find last non-zero frame index (effective sequence length) |
| `temporal_resample(skeleton, target_len)` | Linear interpolation to target_len frames |
| `resample_dataset(X, target_len)` | Apply to entire dataset |

**Usage:**
```bash
python preprocessing/resample.py
```

---

## Complete Pipeline Example

```bash
# Step 1: Extract keypoints (MediaPipe, all classes)
python preprocessing/extract.py

# Step 2: Normalize and compute derived streams
python preprocessing/normalize.py

# Step 3: Resample all streams to 64 frames
python preprocessing/resample.py

# Now run training:
python training/train.py
# Or run comparison:
python main.py --compare-5
```

---

## Data Format Reference

After running the full pipeline, the `output/` directory will contain:

```
output/
├── X_raw.npy           (N, 64, 150) or (N, 64, 118)  — raw keypoints
├── X_normalized.npy    (N, T, 102)                    — normalized joints (before resampling)
├── X_final.npy         (N, 64, 102)                   — joint stream ← primary input
├── X_bones.npy         (N, 64, 102)                   — bone stream
├── X_motion.npy        (N, 64, 102)                   — motion stream
├── X_bone_motion.npy   (N, 64, 102)                   — bone motion stream
├── y.npy               (N,)                            — class labels
└── label_map.json                                      — class name to index mapping
```

All `.npy` files use float32 (inputs) or int64 (labels). `N` = total number of video samples.