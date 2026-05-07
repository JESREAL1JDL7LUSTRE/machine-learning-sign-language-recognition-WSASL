# Multi-Stream ST-GCN (3-stream)

Implementation: [models/stgcn.py](models/stgcn.py)
Runner: [main.py](main.py)

## What it is
A custom three-stream ST-GCN that processes joint positions, bone vectors, and joint motion in parallel. Each stream is a lightweight ST-GCN branch, and the final logits are the sum of the three streams.

## Inputs and outputs
- Input can be a dict with keys `joint`, `bone`, and `motion`, each shaped `(N, 2, T, V)`.
- If a single tensor is passed, only the joint stream is used.
- Output is logits `(N, num_class)`.

## Skeleton graph and adjacency
- The model defines its own 51-joint skeleton in [models/stgcn.py](models/stgcn.py).
- Edges include upper body, left hand, right hand, and two cross links.
- The adjacency is normalized with $D^{-1/2} A D^{-1/2}$ and stored as a constant buffer.

## Stream architecture
Each stream is an `STGCNStream`:
1. Input normalization with `BatchNorm1d` over `(C * V)`.
2. A stack of 6 `STGCNBlock` layers with channel progression:
   - `2 -> 64 -> 64 -> 128 -> 128 -> 256 -> 256`
   - Temporal strides applied at two stages to downsample time.
3. Global average pooling over `(T, V)`.
4. Final `Linear(256 -> num_classes)` classifier per stream.

## ST-GCN block details
- `GraphConv` multiplies node features by the fixed adjacency via `einsum` and applies a `1x1` conv.
- `TemporalConv` applies a temporal kernel (`ks=9`) with padding, batch norm, and dropout.
- `STGCNBlock` adds a residual path when needed.

## Fusion
- The three stream logits are summed: `joint + bone + motion`.
- This matches the late-fusion design of the original ST-GCN family.

## Runner integration
- `run_multi_stream_stgcn` in [main.py](main.py) builds this model and feeds a dict of the three streams.
- It uses `ThreeStreamDataset` and normalizes each stream independently before training.

## Key design choices
- Uses a fixed graph (no adaptive adjacency) for simplicity and stability.
- Each stream is fully independent and only fuses at the logit level.
- A simpler architecture is kept separate from the improved `ST_GCN` backbone used in the 2- and 4-stream variants.
