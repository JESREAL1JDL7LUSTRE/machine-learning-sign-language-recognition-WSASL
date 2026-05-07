# 4-Stream Fusion ST-GCN Walkthrough (Beginner, Step-by-Step)

This explains the 4-stream model in very simple language. The main code is in `models/st_gcn_twostream.py`, and it uses the backbone in `models/st_gcn.py`.

## 1) Big idea in one sentence
We build four separate views of the same sign (joint, motion, bone, bone-motion), combine them early, and then make one final prediction.

## 2) The four streams (what they mean)
- **Joint:** the raw joint positions.
- **Motion:** how joints change from one frame to the next.
- **Bone:** the direction and length between connected joints.
- **Bone motion:** how the bones themselves change over time.

Each stream is a separate ST-GCN backbone. This lets the model learn different kinds of patterns.

## 3) Input and output
- Inputs: four arrays with shape `(N, 2, T, V)`.
- Output: logits `(N, num_classes)`.

## 4) File: `models/st_gcn_twostream.py`

### 4.1) `Model.__init__`
- When `early_fusion=True`, the model:
  1. Builds 4 `ST_GCN` streams with `extract_features=True`.
  2. Each stream outputs a **feature vector** (size 128).
  3. Concatenates all 4 feature vectors into one big vector (size 512).
  4. Passes it through a small MLP (two `Linear` layers).

**Why early fusion?**
It lets the model mix information across streams before the final decision.

### 4.2) `Model._compute_motion`
- Computes a smoothed temporal derivative (approximate motion).
- This is used only if `motion` is not provided.

### 4.3) `Model.forward`
- If motion or bone streams are missing, it fills them with zeros.
- If early fusion:
  1. Each stream outputs features.
  2. Features are concatenated.
  3. The fusion head outputs final logits.
- If late fusion:
  - Each stream outputs logits and they are summed.

## 5) Backbone details from `models/st_gcn.py`
Each stream uses the same improved ST-GCN backbone:
- A fixed skeleton graph plus adaptive learned edges.
- DropGraph regularization (randomly remove joints).
- 10 layers of spatial-temporal graph convolutions.
- Global average pooling and final head.

## 6) Where this model is trained
- `training/train.py` is the specialized trainer.
- `main.py` also supports it using `run_4stream_fusion`.
- Both use the same data streams from `output/`.

## 7) The training data (how it is prepared)
The four streams are created in `preprocessing/normalize.py`:
1. Clean missing keypoints.
2. Filter to 51 joints.
3. Normalize positions.
4. Compute bone vectors.
5. Compute motion for joints and bones.
6. Resample all streams to 64 frames.

## 8) Mental picture of one forward pass
1. The same sign is represented as four different tensors.
2. Each tensor goes through its own ST-GCN.
3. Four feature vectors are concatenated.
4. A small classifier combines them into final logits.

## 9) Why this model performs well
- Each stream highlights a different property of motion.
- Early fusion lets the model learn cross-stream relationships.
- DropGraph and adaptive adjacency improve robustness.
