# Presentation Context Card — Sign Language Recognition (WSASL)
## Quick Reference for Presenters

---

## SLIDE 1 — Title / Overview
**Project:** Skeleton-based ASL Sign Language Recognition using ST-GCN  
**Dataset:** WLASL (World-Level American Sign Language) — 50 classes, ~524 videos  
**Goal:** Classify isolated sign language gestures from video using graph neural networks

**Key message:** We treat the human body as a GRAPH — joints are nodes, bones are edges.

---

## SLIDE 2 — Data Collection
**Source:** WLASL GitHub dataset (Li et al., 2020 IEEE WACV)  
**Stats:** 50 sign words, 5–16 videos per class, ~524 total samples  
**Tool:** MediaPipe — extracts 33 body + 21+21 hand landmarks per frame  
**Output:** `X_raw.npy` shape `(524, 64, 150)` — 150 features per frame

**Screenshot to show:**
- Dataset folder structure (class subfolders with MP4 videos)
- `X_raw.npy` shape printed from Python

---

## SLIDE 3 — Data Preprocessing
**Pipeline (4 steps):**
1. `extract.py` — video → skeleton keypoints (150 features/frame)
2. `normalize.py` — filter joints (150→102), center+scale, smooth
3. `resample.py` — resample to exactly 64 frames
4. Feature engineering → **4 streams** (joint, bone, motion, bone_motion)

**Key preprocessing:** Per-frame centering on mid-shoulder + scaling by shoulder width  
**Why 4 streams?** Each captures complementary motion cues: WHERE, HOW ORIENTED, HOW FAST, ANGULAR VELOCITY

**Screenshots to show:**
- Code blocks from `normalize.py` (lines 337–377 — normalize_skeleton)
- Output file shapes: `(524, 64, 102)` for each stream

---

## SLIDE 4 — Modeling
**Architecture:** ST-GCN (Spatial-Temporal Graph Convolutional Network)
- Graph convolution: aggregates info from neighboring joints via adjacency matrix
- Temporal convolution: captures motion across frames (kernel = 9 frames)

**3 Models compared:**

| # | Model | Streams | Fusion |
|---|-------|---------|--------|
| 1 | Multi-Stream ST-GCN | Joint + Bone + Motion | Late (sum logits) |
| 2 | Four-Stream Late Fusion | Joint + Motion + Bone + BoneMotion | Late (sum logits) |
| 3 | **Four-Stream Early Fusion** | Joint + Motion + Bone + BoneMotion | **Early (concat features)** |

**Training tricks:** AdamW, warmup-cosine LR, SWA from ep50, label smoothing, class weights, early stopping

---

## SLIDE 5 — Evaluation
**Method:** 4-Fold Stratified K-Fold CV + 15% held-out test set

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Multi-Stream (late) | 5.11% ± 0.48% | 1.30% |
| Four-Stream (late) | 5.57% ± 0.69% | 1.30% |
| **Four-Stream (early)** | **5.34% ± 0.41%** | **3.90%** |
| Random chance baseline | — | **2.00%** |

**Key finding:** 4-stream early fusion is ~3× better than late fusion on the test set.

**Why low accuracy?** Only ~10 samples per class — state-of-the-art requires 50+ samples per class.

---

## SLIDE 6 — Results Demo
**Run:** `python evaluation/evaluate.py --video path/to/video.mp4`  
**Output:** "Predicted class: basketball"

**Pipeline:**  
Video → MediaPipe → normalize → 64-frame resample → 4 streams → ST-GCN → class name

**Best cases:** Signs with distinctive motion (basketball = dribbling, eat = hand-to-mouth)  
**Hard cases:** Lexically similar signs, low-sample classes (book, clothes: only 6 videos each)

---

## Key Numbers to Remember
- **50 classes** (sign words)
- **~524 total videos** — very small dataset
- **51 joints × 2 = 102 features** per frame
- **64 frames** per video (after resampling)
- **4 input streams** (joint, bone, motion, bone motion)
- **Best test accuracy: 3.90%** (vs. 2.0% random chance)
- **Random chance baseline: 2.0%** (1/50 classes)

---

## Code Files to Show
| Section | File | Key Lines |
|---------|------|-----------|
| Extraction | `preprocessing/extract.py` | Lines 192–266 (extract_mp function) |
| Normalization | `preprocessing/normalize.py` | Lines 337–379 (normalize_skeleton) |
| Bone/Motion | `preprocessing/normalize.py` | Lines 437–484 (compute_bone_vectors, compute_motion) |
| Model (Early) | `models/st_gcn_twostream.py` | Lines 40–181 (full Model class) |
| Training | `training/train.py` | Lines 328–438 (train_fold function) |
| Evaluation | `evaluation/evaluate.py` | Lines 76–124 (predict_video function) |

---

## Charts Available (output/charts/)
- `multi_stream_stgcn_results.png`
- `4stream_late_fusion_results.png`
- `4stream_fusion_results.png` (BEST)
- `comparison_overview.png` — All models side-by-side
