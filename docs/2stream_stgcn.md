# 2-Stream ST-GCN (ported)

Implementation: [models/stgcn_2stream_ported.py](models/stgcn_2stream_ported.py)
Runner: [main.py](main.py)
Backbone: [models/st_gcn.py](models/st_gcn.py)
Graph utils: [models/graph.py](models/graph.py), [models/tgcn.py](models/tgcn.py)

## What it is
A two-stream, late-fusion ST-GCN. One stream processes raw joint coordinates and the other processes bone vectors. Each stream is an independent ST-GCN backbone (`ST_GCN`). The final logits are the sum of both streams.

## Inputs and outputs
- Input `x`: joint positions shaped `(N, 2, T, V)` where `N` is batch, `T` is frames, `V` is joints.
- Input `bone` (optional): bone vectors shaped `(N, 2, T, V)`; if omitted, zeros are used.
- Output: logits `(N, num_class)`.

## Data flow (high level)
1. Build a graph adjacency `A` from the mediapipe 51-joint layout.
2. Run the joint stream ST-GCN on `x`.
3. Run the bone stream ST-GCN on `bone`.
4. Sum the two logits and return.

## Core modules and functions
- `TwoStreamPortedSTGCN.__init__` sets up two `ST_GCN` backbones with shared configuration.
- `TwoStreamPortedSTGCN.forward` sums `joint_stream(x)` + `bone_stream(bone)`.
- `ST_GCN` (in [models/st_gcn.py](models/st_gcn.py)) is a 10-layer ST-GCN with:
  - `Graph` adjacency (`Graph(**graph_args)`) using layout `mediapipe_51`.
  - `AdaptiveAdjacency` (if enabled) that adds learnable edges to the fixed graph.
  - `drop_graph` to randomly drop joints during training.
  - `edge_importance_weighting` to learn per-layer importance weights for the adjacency.
  - `st_gcn` blocks composed of `ConvTemporalGraphical` + temporal convolutions and residuals.
- `ConvTemporalGraphical` (in [models/tgcn.py](models/tgcn.py)) applies temporal conv then multiplies by the adjacency with `einsum`.

## Graph details (mediapipe_51)
- The layout connects a 51-joint skeleton: upper body + both hands.
- The adjacency is normalized and split into spatial subsets using `strategy='spatial'`.
- The adaptive adjacency adds a learnable correction `A_fixed + alpha * softmax(A_learn)`.

## Runner integration
- The training runner is `run_2stream_stgcn` in [main.py](main.py).
- It constructs the model with:
  - `graph_args = {"layout": "mediapipe_51", "strategy": "spatial"}`
  - `adaptive_graph = True`
  - `drop_graph_prob = 0.1`
  - `dropout = DROPOUT`
- The runner uses `TwoStreamDataset` and feeds `(X_joint, X_bone)` into `forward_fn`.

## Key design choices
- Late fusion keeps each stream independent and sums logits, matching the original 2-stream ST-GCN paper.
- Reduced-width ST-GCN (`ST_GCN`) is used to better generalize on smaller datasets.
- `drop_graph` and adaptive adjacency are enabled to improve robustness to missing joints and topology noise.
