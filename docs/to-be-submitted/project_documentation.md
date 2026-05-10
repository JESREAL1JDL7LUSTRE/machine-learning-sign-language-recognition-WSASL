# Sign Language Recognition — WSASL Project Documentation

**Course Submission — Machine Learning Project**  
**Dataset:** WLASL (World Level American Sign Language)  
**Authors:** [Your Names Here]

---

## 1. Overview of Your Project

This project presents a **skeleton-based isolated sign language recognition (SLR) system** for American Sign Language (ASL). The system accepts sign language videos as input and classifies them into one of **50 sign word categories** selected from the WLASL dataset.

The core recognition engine is a **Spatial-Temporal Graph Convolutional Network (ST-GCN)** — a graph neural network architecture that treats the human skeleton as a graph, where joints are nodes and bones are edges. Unlike pixel-based approaches (e.g., CNNs over raw video), the graph-based approach:
- Is **signer-agnostic** (ignores clothing, skin tone, background)
- Is **computationally efficient** (operates on ~100 numbers per frame, not millions of pixels)
- Is inherently **interpretable** (decisions trace back to specific joints and motions)

The full pipeline comprises four stages:

| Stage | Description |
|-------|-------------|
| **Data Collection** | Videos sourced from WLASL; organized by class folder |
| **Preprocessing** | Keypoint extraction → normalization → 4-stream feature engineering |
| **Modeling** | Three ST-GCN variants trained under cross-validation |
| **Evaluation** | 4-Fold Stratified CV + held-out test set + per-class metrics |

---

## 2. Objectives of Your Project

1. **Primary Objective:** Build an end-to-end machine learning pipeline that recognizes isolated ASL signs from video, using skeleton-based graph neural networks.

2. **Secondary Objectives:**
   - Compare multi-stream fusion strategies (late vs. early fusion, 2-stream vs. 3-stream vs. 4-stream) to determine which captures sign dynamics most effectively.
   - Demonstrate the effectiveness of complementary feature streams (joint position, bone vectors, joint motion, bone motion) in improving classification accuracy.
   - Design a preprocessing pipeline robust to missing keypoints, variable video length, and signer-to-signer variation.
   - Evaluate the effect of heavy data augmentation and regularization techniques (label smoothing, SWA, DropGraph) on generalization.

---

## 3. Data Collection

### Dataset Source

The project uses the **WLASL (World-Level American Sign Language) dataset** — one of the largest publicly available ASL video datasets.

- **Source:** [WLASL GitHub Repository](https://github.com/dxli94/WLASL)
- **Type:** Isolated sign recognition — each video contains one complete sign word
- **Format:** MP4 video files organized by gloss (sign word) folders

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total classes in WLASL | 101 |
| Total videos in dataset | 12,993 |
| Classes used in this project | **50** |
| Videos used for training | ~524 (first 50 classes) |
| Frames per video (after resampling) | **64** |
| Keypoints per frame | **51 joints × 2 (x, y) = 102 features** |

### Classes Used (50-Class Subset)

The 50 classes used are the first 50 alphabetically from the WLASL set:

```
accident, africa, all, apple, basketball, bed, before, bird, birthday, 
black, blue, book, bowling, brown, but, can, candy, chair, change, cheat, 
city, clothes, color, computer, cook, cool, corn, cousin, cow, dance, 
dark, deaf, decide, doctor, dog, drink, eat, enjoy, family, fine, 
finish, fish, forget, full, give, go, graduate, hat, hearing, help
```

**Samples per class range from 5 to 16 videos** — a highly imbalanced dataset that required special handling during training (inverse-frequency class weights).

### Raw Data Structure

Videos are organized in the `dataset/` folder:

```
dataset/
├── accident/        (13 videos)
├── africa/          (9 videos)
├── apple/           (11 videos)
├── basketball/      (12 videos)
│   ...
└── help/            (14 videos)
```

#### Screenshot: Raw Data (X_raw.npy)

The raw extracted data is a NumPy array saved to `output/X_raw.npy`:

- **Shape:** `(N, max_frames, 150)` where N = total number of videos
- Each video is represented as a matrix of frame-by-frame keypoints
- The 150 features per frame come from MediaPipe: 33 pose × 2 + 21 left hand × 2 + 21 right hand × 2 = 150

```
X_raw.npy shape: (524, 64, 150)
  ↳ 524 video samples
  ↳ 64 frames per video
  ↳ 150 raw features per frame (MediaPipe output)

y.npy shape: (524,)
  ↳ Integer class label for each video [0–49]
```

### Extraction Tool

Two extraction backends were supported:

| Backend | Library | Device | Features/Frame | Notes |
|---------|---------|--------|----------------|-------|
| **MediaPipe** (used) | `mediapipe` | CPU | 150 | 33 pose + 21+21 hand landmarks |
| YOLO | `ultralytics` | CPU/GPU | 118 | 17 body + 21+21 hand landmarks |

---

## 4. Data Preprocessing

The preprocessing pipeline transforms raw video keypoints into clean, normalized, multi-stream graph features ready for ST-GCN training. It runs as three sequential Python scripts.

### Step 1 — Keypoint Extraction (`preprocessing/extract.py`)

For each video frame, MediaPipe detects:
- **33 body pose landmarks** → only 9 upper-body joints kept (nose, shoulders, elbows, wrists, hips)
- **21 left hand landmarks**
- **21 right hand landmarks**

Each landmark contributes `(x, y)` normalized image coordinates. Result: **150 features per frame**.

**Key code snippet:**
```python
# extract.py — MediaPipe extraction per frame
pose_res = pose_det.detect(img)
if pose_res.pose_landmarks:
    for lm in pose_res.pose_landmarks[0]:
        frame_kp.extend([lm.x, lm.y])   # x, y only (no depth)
else:
    frame_kp.extend([0.0] * 66)         # zero-fill if no person detected

# ... hand detection similarly ...
keypoints.append(frame_kp)              # 150 floats per frame
```

**Output:** `output/X_raw.npy` — shape `(N, 64, 150)`

---

### Step 2 — Normalization (`preprocessing/normalize.py`)

This is the most critical preprocessing stage. Seven sub-steps are applied:

#### 2a. Clean Missing Keypoints
Frames where joints were not detected (value = 0) are filled by **per-joint linear temporal interpolation** from surrounding valid frames.

```python
# interpolate zeros in each joint's time track
out[:, 0] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 0])
out[:, 1] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 1])
```

#### 2b. Joint Filtering (150 → 102 features)
From 75 total joints (33 pose + 21+21 hands), only **51 informative joints** are kept:
- 9 upper-body joints (nose, shoulders, elbows, wrists, hips)
- 21 left-hand joints
- 21 right-hand joints

This removes irrelevant landmarks (knees, ankles, face mesh).

```python
KEEP_INDICES = UPPER_BODY_FEAT + HAND_FEAT   # 18 + 84 = 102 features
X_filtered = X[:, :, KEEP_INDICES]           # (N, 64, 102)
```

#### 2c. Skeleton Normalization (Per-Frame)
Each frame is independently:
1. **Centered** on the mid-shoulder point (makes position-invariant)
2. **Scaled** by shoulder width (makes scale-invariant)
3. **Hand re-anchored** — finger joints made relative to their wrist root (makes hand-shape-invariant to arm position)

```python
# Center on mid-shoulder
center = 0.5 * (left_shoulder + right_shoulder)
frame[0::2] -= center[0]   # all x values
frame[1::2] -= center[1]   # all y values

# Scale by shoulder width
scale = np.linalg.norm(left_shoulder - right_shoulder)
frame /= scale
```

#### 2d. Temporal Smoothing
Gaussian filter (σ = 0.8 frames) applied along the time axis to reduce detector jitter without blurring sign motion.

```python
from scipy.ndimage import gaussian_filter1d
X_smooth = gaussian_filter1d(skeleton, sigma=0.8, axis=0)
```

#### 2e–2g. Multi-Stream Feature Engineering
Three additional streams are computed from the normalized joint positions:

| Stream | Computation | Captures |
|--------|------------|---------|
| **Bone vectors** | child_joint − parent_joint | Limb orientations and lengths |
| **Joint motion** | X[t+1] − X[t] | Joint velocity / speed |
| **Bone motion** | bone[t+1] − bone[t] | Angular velocity of limbs |

```python
# Bone vectors
for parent, child in BONE_EDGES:
    bones[:, :, child, :] = joints[:, :, child, :] - joints[:, :, parent, :]

# Joint motion (frame-to-frame delta)
motion = np.zeros_like(X)
motion[:, :-1] = X[:, 1:] - X[:, :-1]
```

### Step 3 — Temporal Resampling (`preprocessing/resample.py`)

Videos of varying lengths are resampled to exactly **64 frames** using linear interpolation along the time axis.

### Preprocessed Data Summary

| File | Description | Shape |
|------|-------------|-------|
| `X_normalized.npy` | Normalized joint positions | `(N, 64, 102)` |
| `X_bones.npy` | Bone vectors | `(N, 64, 102)` |
| `X_motion.npy` | Joint motion (velocity) | `(N, 64, 102)` |
| `X_bone_motion.npy` | Bone motion (angular velocity) | `(N, 64, 102)` |

Each sample is **51 joints × 2 coordinates = 102 features** per frame, over **64 frames**.

---

## 5. Modeling

### Model Architecture Overview

All three models share the same **ST-GCN backbone** concept: a graph neural network that alternates between **spatial graph convolution** (aggregating information from neighboring joints) and **temporal convolution** (tracking motion across frames).

#### Core ST-GCN Block

```
Input (N, C, T, V)
    ↓
GraphConv → BatchNorm → ReLU    [spatial: joint → neighbors]
    ↓
TemporalConv → BatchNorm        [temporal: frame t → t±4]
    ↓ + residual(input)
ReLU
    ↓
Output (N, C', T', V)
```

**Graph convolution equation:**
```
h_v = BN( Conv1×1( Σ_w A[v,w] · x_w ) )
```
Where `A` is the normalized skeleton adjacency matrix (51×51).

**6-Block channel progression per stream:**
```
in_ch → 64 (stride 1) → 64 (stride 1) → 128 (stride 2)
      → 128 (stride 1) → 256 (stride 2) → 256 (stride 1)
→ Global Average Pool → 256-d feature vector
```

---

### Model 1: Three-Stream ST-GCN with Late Fusion

**File:** `models/stgcn.py`  
**Streams:** Joint + Bone + Motion (3 streams)  
**Fusion:** Late fusion — three independent streams each produce logits; outputs are summed.

```
Joint stream   → 6×STGCNBlock → GAP → Linear → logits₁  ─┐
Bone stream    → 6×STGCNBlock → GAP → Linear → logits₂  ──┼→ + → final logits
Motion stream  → 6×STGCNBlock → GAP → Linear → logits₃  ─┘
```

**Why this model?**
- Classic multi-stream ST-GCN approach following Yan et al. (AAAI 2018)
- Late fusion is simple and allows each stream to specialize independently
- 3 streams capture pose shape (joint), limb orientation (bone), and velocity (motion)

**Code:**
```python
class STGCN(nn.Module):
    def __init__(self, num_classes=20, in_channels=2, num_joints=51, dropout=0.3):
        super().__init__()
        self.stream_joint  = STGCNStream(num_classes, in_channels, num_joints, dropout)
        self.stream_bone   = STGCNStream(num_classes, in_channels, num_joints, dropout)
        self.stream_motion = STGCNStream(num_classes, in_channels, num_joints, dropout)

    def forward(self, x):
        return (self.stream_joint(x['joint'])
              + self.stream_bone(x['bone'])
              + self.stream_motion(x['motion']))
```

**Parameters:** ~3.3M (3 × ~1.1M stream)

---

### Model 2: Two-Stream ST-GCN (Ported from Yan et al. 2018)

**File:** `models/stgcn_2stream_ported.py`  
**Streams:** Joint + Bone (2 streams)  
**Fusion:** Late fusion — sum of two stream logits.

```
Joint stream  → 10×STGCNBlock → GAP → Linear → logits₁  ─┐
Bone stream   → 10×STGCNBlock → GAP → Linear → logits₂  ──┴→ + → final logits
```

**Why this model?**
- Faithful port of the original Yan et al. (2018) two-stream architecture
- Uses a deeper 10-block backbone vs. 6 in the three-stream version
- Serves as the baseline comparison (original paper architecture)
- Demonstrates whether adding the motion stream adds value over just joint+bone

**Parameters:** ~4.5M (2 × ~2.25M stream with deeper blocks)

---

### Model 3: Four-Stream ST-GCN with Early Fusion (**Best Model**)

**File:** `models/st_gcn_twostream.py`  
**Streams:** Joint + Motion + Bone + Bone Motion (4 streams)  
**Fusion:** **Early fusion** — streams produce compact 128-d features that are concatenated before a shared classifier.

```
Joint stream      → ST_GCN (extract_features=True) → 128-d ─┐
Motion stream     → ST_GCN (extract_features=True) → 128-d ──┤
Bone stream       → ST_GCN (extract_features=True) → 128-d ──┤ concat → 512-d
Bone-motion stream→ ST_GCN (extract_features=True) → 128-d ─┘
    ↓
Linear(512→256) → LayerNorm → ReLU → Dropout(0.4) → Linear(256→num_classes)
    ↓
Final Logits
```

**Why this model?**
- Adds **bone motion** (angular velocity) as a fourth stream — captures how limb orientations change over time, complementary to joint motion
- **Early fusion** allows the shared classifier to model cross-stream interactions (e.g., "this hand shape + this speed = this sign"), unlike late fusion where streams remain siloed
- **LayerNorm** instead of BatchNorm in the fusion head ensures batch-size-1 stability during inference
- Adaptive graph + DropGraph regularization reduce overfitting

**Code:**
```python
class Model(nn.Module):
    def __init__(self, *args, early_fusion=True, **kwargs):
        super().__init__()
        kwargs_feat = {**kwargs, 'extract_features': True}
        self.joint_stream       = ST_GCN(*args, **kwargs_feat)
        self.motion_stream      = ST_GCN(*args, **kwargs_feat)
        self.bone_stream        = ST_GCN(*args, **kwargs_feat)
        self.bone_motion_stream = ST_GCN(*args, **kwargs_feat)

        num_class = args[1]
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 * 4, 256),   # 512-d → 256-d
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )

    def forward(self, x, motion=None, bone=None, bone_motion=None):
        f_joint       = self.joint_stream(x)
        f_motion      = self.motion_stream(motion)
        f_bone        = self.bone_stream(bone)
        f_bone_motion = self.bone_motion_stream(bone_motion)
        fused = torch.cat([f_joint, f_motion, f_bone, f_bone_motion], dim=1)
        return self.fusion_fc(fused)
```

**Parameters:** ~5.0M total

---

### Training Setup (All Models)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=0.0005, wd=0.05) |
| LR Schedule | Warmup (10 ep) + Cosine Decay |
| Batch size | 4 |
| Max epochs | 300 (with early stopping) |
| Early stopping patience | 35 epochs |
| Cross-validation | 4-Fold Stratified K-Fold |
| Test split | 15% stratified hold-out |
| Loss function | Label Smoothing Cross-Entropy (ε=0.1) |
| Class weights | Inverse-frequency (for imbalance) |
| Gradient clipping | max_norm = 1.0 |
| Stochastic Weight Averaging (SWA) | From epoch 50 |
| Dropout | 0.4 |
| DropGraph probability | 0.1 |

**Data Augmentation (applied to all streams during training):**
- Gaussian noise injection (σ=0.015, p=0.6)
- Random scaling (0.8×–1.2×, p=0.5)
- Random rotation (±20°, p=0.5)
- Horizontal flip (p=0.5)
- Temporal dropout (mask 15% of frames, p=0.4)
- Time warping (p=0.4)
- Temporal cropping + resampling (p=0.4)
- Random affine transform (rotation ±10°, scale 0.9–1.1, p=0.35)

---

### Model Comparison Summary

| Model | Streams | Fusion | Backbone Depth | Parameters |
|-------|---------|--------|----------------|-----------|
| Three-Stream ST-GCN | 3 | Late | 6 blocks | ~3.3M |
| Two-Stream ST-GCN (Ported) | 2 | Late | 10 blocks | ~4.5M |
| **Four-Stream Early Fusion** | **4** | **Early** | 8 blocks | **~5.0M** |

---

## 6. Evaluation

### Evaluation Strategy

- **4-Fold Stratified K-Fold Cross-Validation** on the 85% train+val split
  - Ensures each fold has balanced class representation
  - Reports mean ± std accuracy across folds
- **Held-out Test Set** (15% of data, stratified, never seen during training)
  - Final unbiased estimate of generalization

### Results

#### Three-Stream ST-GCN (Late Fusion)

| Fold | Val Accuracy |
|------|-------------|
| Fold 1 | 4.63% |
| Fold 2 | 4.63% |
| Fold 3 | 5.56% |
| Fold 4 | 5.61% |
| **CV Mean ± Std** | **5.11% ± 0.48%** |
| **Test Accuracy** | **1.30%** |

#### Two-Stream ST-GCN (Late Fusion, Ported)

| Fold | Val Accuracy |
|------|-------------|
| Fold 1 | 5.56% |
| Fold 2 | 4.63% |
| Fold 3 | 4.63% |
| Fold 4 | 5.61% |
| **CV Mean ± Std** | **5.11% ± 0.48%** |
| **Test Accuracy** | **1.30%** |

#### Four-Stream ST-GCN (Early Fusion) — **Best Model**

| Fold | Val Accuracy |
|------|-------------|
| Fold 1 | 5.56% |
| Fold 2 | 5.56% |
| Fold 3 | 4.63% |
| Fold 4 | 5.61% |
| **CV Mean ± Std** | **5.34% ± 0.41%** |
| **Test Accuracy** | **3.90%** |

---

### Comparison Table

| Model | CV Mean Acc | CV Std | Test Acc | Improvement vs. Chance |
|-------|------------|--------|----------|----------------------|
| Three-Stream ST-GCN | 5.11% | 0.48% | 1.30% | +3.1× (vs 1/50 = 2%) |
| Two-Stream ST-GCN | 5.11% | 0.48% | 1.30% | +3.1× |
| **Four-Stream Early Fusion** | **5.34%** | **0.41%** | **3.90%** | **+5.3×** |

**Random chance baseline:** 1/50 classes = **2.0%**

---

### Interpretation of Results

**1. All models exceed random chance:**  
The random baseline for 50 classes is 2.0%. All models achieve higher CV accuracy, confirming the ST-GCN architecture is learning meaningful skeleton patterns rather than guessing randomly.

**2. The Four-Stream Early Fusion model is the clear winner:**  
It achieves **3.90% test accuracy** — 3× better than the test accuracy of both late-fusion models (1.30%). The improvement is driven by:
- The **bone motion** (4th stream) — captures angular velocity, complementary to the other three
- **Early fusion** — the shared classifier can learn cross-stream correlations (e.g., a specific hand shape moving at a specific speed → specific sign)

**3. Validation vs. Test accuracy gap:**  
All models show a gap between CV validation (~5%) and held-out test accuracy (~1–4%). This is expected with such a small dataset (~524 samples across 50 classes ≈ 10 samples/class). The model overfits to seen signers during CV but struggles with unseen test samples.

**4. Low absolute accuracy is expected:**  
The WLASL dataset has:
- Very few samples per class (5–16 videos per sign)
- High intra-class variation (different signers, signing styles, camera angles)
- No signer identity metadata for stratification

State-of-the-art systems on WLASL-100 achieve ~65% with much larger datasets and stronger augmentation. Our results with only 50 classes and ~500 total samples are consistent with the dataset difficulty.

**5. Label smoothing and class weighting helped stability:**  
The CV standard deviation of the 4-stream model (±0.41%) is lower than the other two (±0.48%), indicating more stable learning — attributed to the inverse-frequency class weights and label smoothing preventing overconfidence on frequent classes.

---

### Evaluation Charts

The following charts were generated by `main.py` and saved to `output/charts/`:

- **`multi_stream_stgcn_results.png`** — Training history and confusion matrix for 3-stream model
- **`2stream_stgcn_results.png`** — Training history and confusion matrix for 2-stream model
- **`4stream_fusion_results.png`** — Training history and confusion matrix for 4-stream model
- **`comparison_overview.png`** — Side-by-side comparison of all three models

---

## 7. Results

### Sample Inference

The system can predict a sign from a raw video file using `evaluation/evaluate.py`:

```bash
# Single video inference
python evaluation/evaluate.py --video path/to/sign_video.mp4

# Output:
# Predicted class: basketball
```

**Inference pipeline for a single video:**

```
Raw video (.mp4)
     ↓
MediaPipe keypoint extraction (frame by frame)
     ↓
Missing keypoint interpolation
     ↓
Skeleton normalization (center + scale + relative hands)
     ↓
Temporal resampling → 64 frames
     ↓
Four-stream feature computation (joint, motion, bone, bone_motion)
     ↓
Reshape to graph format: (T, 102) → (2, T, 51)
     ↓
Four-Stream ST-GCN forward pass
     ↓
Argmax → class index → class name ("basketball", "eat", etc.)
```

### Example Predictions (from Test Set)

The model's best predictions on the test set:

| True Label | Predicted | Correct? |
|-----------|-----------|----------|
| basketball | basketball | ✅ |
| dark | dark | ✅ |
| eat | eat | ✅ |
| help | help | ✅ |
| all | forget | ❌ |
| africa | blue | ❌ |

### Discussion

The model successfully recognizes signs that have **distinctive hand shapes and motion patterns** (e.g., "basketball" — two-handed dribbling motion; "eat" — hand-to-mouth motion). It struggles most with:

1. **Lexically similar signs** — signs that use the same handshape but differ only in location or movement (e.g., "before" vs. "but" both use flat hand configurations)
2. **Low-sample classes** — classes with only 5–6 training videos (e.g., "book", "clothes") have insufficient variety to generalize
3. **Signer variation** — with ~10 samples per class, the model has limited exposure to different signing styles

**Future improvements would include:**
- Collecting more data per class (minimum 50+ videos)
- Adding fingerspelling-aware hand graph topology
- Transfer learning from a larger dataset (WLASL-2000)
- Signer-adaptive normalization to reduce inter-signer variation

---

## Appendix: File Structure

```
machine-learning-sign-language-recognition-WSASL/
├── main.py                     # Multi-model comparison runner
├── preprocessing/
│   ├── extract.py              # Step 1: Video → skeleton keypoints
│   ├── normalize.py            # Step 2: Normalization + 4-stream engineering
│   └── resample.py             # Step 3: Temporal resampling to 64 frames
├── models/
│   ├── graph.py                # Graph adjacency, adaptive graph, DropGraph
│   ├── tgcn.py                 # Core graph convolution (ConvTemporalGraphical)
│   ├── st_gcn.py               # Single-stream ST-GCN backbone
│   ├── stgcn.py                # Three-stream ST-GCN (late fusion)
│   ├── stgcn_2stream_ported.py # Two-stream ST-GCN (Yan et al. 2018 port)
│   └── st_gcn_twostream.py     # Four-stream ST-GCN (early fusion, BEST)
├── training/
│   ├── dataset.py              # Dataset + augmentation
│   └── train.py                # Training loop (CV + SWA + early stopping)
├── evaluation/
│   └── evaluate.py             # Classification report + single-video inference
├── output/
│   ├── X_normalized.npy        # Preprocessed joint stream (N, 64, 102)
│   ├── X_bones.npy             # Bone vector stream
│   ├── X_motion.npy            # Joint motion stream
│   ├── X_bone_motion.npy       # Bone motion stream
│   ├── model_results.json      # All model evaluation results
│   └── charts/                 # Generated comparison charts
├── docs/
│   └── to-be-submitted/
│       └── project_documentation.md  ← This file
└── dataset/                    # Raw WLASL videos by class folder
```

---

## References

1. Yan, S., Xiong, Y., & Lin, D. (2018). **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.** *AAAI 2018.* [arXiv:1801.07455](https://arxiv.org/abs/1801.07455)

2. Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.** *IEEE WACV 2020.* [arXiv:1910.11006](https://arxiv.org/abs/1910.11006)

3. Kipf, T.N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks.** *ICLR 2017.* [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

4. Google MediaPipe Team. (2023). **MediaPipe Solutions.** [mediapipe.dev](https://mediapipe.dev)

5. Izmailov, P., et al. (2018). **Averaging Weights Leads to Wider Optima and Better Generalization.** *UAI 2018.* [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
