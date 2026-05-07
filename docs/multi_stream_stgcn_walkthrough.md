# Multi-Stream ST-GCN (3-Stream) Walkthrough (Beginner)

This explains the simpler 3-stream ST-GCN in plain language. The main code is in `models/stgcn.py`.

## 1) Big idea in one sentence
We run three small ST-GCNs in parallel (joint, bone, motion) and add their predictions together.

## 2) The three streams
- **Joint stream:** uses raw joint positions.
- **Bone stream:** uses bone vectors.
- **Motion stream:** uses frame-to-frame joint changes.

Each stream has the same architecture, but they do not share weights.

## 3) Input and output
- Inputs: a dictionary with three tensors: `joint`, `bone`, `motion`.
- Each tensor shape: `(N, 2, T, V)`.
- Output: logits `(N, num_classes)`.

## 4) File: `models/stgcn.py`

### 4.1) Graph setup
- The graph is built inside this file using a fixed 51-joint layout.
- `build_adjacency` creates a normalized adjacency matrix.

### 4.2) Graph convolution
- `GraphConv` uses `einsum` to multiply node features by the adjacency matrix.
- A 1x1 convolution then mixes channels.

### 4.3) Temporal convolution
- `TemporalConv` uses a 1D convolution across time (kernel size 9).
- It keeps the same time length using padding.

### 4.4) STGCNBlock
- A graph conv + temporal conv + residual connection.
- Residual makes training stable.

### 4.5) STGCNStream
Each stream contains:
1. Input batch normalization.
2. 6 ST-GCN blocks with channel growth (64 -> 256).
3. Global average pooling.
4. A linear classifier.

### 4.6) STGCN (the multi-stream wrapper)
- Creates three streams: joint, bone, motion.
- In `forward`, it adds their logits together.

## 5) Where it is used
- `run_multi_stream_stgcn` in `main.py`.
- It uses `ThreeStreamDataset` and normalizes each stream separately.

## 6) Why this model is simpler
- It uses a fixed graph (no adaptive adjacency).
- It uses a smaller number of blocks than the improved backbone.
- Fusion is late (logit sum), not early.

## 7) Mental picture of one forward pass
1. The same sign is represented in three ways.
2. Each stream makes its own prediction.
3. The predictions are added to produce the final class scores.

## 8) When to choose this model
- If you want a simpler baseline.
- If you want faster training with fewer custom tricks.
- If you want a model that is easier to understand.
