# Graph / TGCN comparison and usage (suggested)

This note summarizes how the project implements spatial-temporal graph convolutions
and where each implementation is used.

## Purpose
- Provide a quick reference for contributors deciding which implementation to use
  or refactor: the modular backbone (`models/graph.py` + `models/tgcn.py`) vs the
  self-contained comparison implementation (`models/stgcn_multi.py`).

## Implementations

### Modular implementation (recommended for reuse)
- Files: `models/graph.py`, `models/tgcn.py`, `models/st_gcn_backbone.py`
- Responsibilities:
  - `models/graph.py`: builds adjacency `A` (K, V, V), supports multiple layouts,
    normalization, `AdaptiveAdjacency`, and `drop_graph`.
  - `models/tgcn.py`: `ConvTemporalGraphical` — low-level op: temporal conv →
    reshape into K subsets → einsum with `A`.
  - `models/st_gcn_backbone.py`: composes blocks using the above, supports edge
    importance, adaptive graph, DropGraph, `extract_features` flag.
- Used by: `models/st_gcn_four_stream.py` (each stream is an instance of the backbone).

### Self-contained implementation (comparison model)
- File: `models/stgcn_multi.py`
- Responsibilities: local `build_adjacency`, `GraphConv`, `TemporalConv`, `STGCNStream`.
- Differences vs modular approach:
  - `GraphConv` performs adjacency einsum then 1×1 conv (order differs from `ConvTemporalGraphical`).
  - No `AdaptiveAdjacency` or DropGraph; single hard-coded `mediapipe_51` layout.
  - Simpler 6-block stream used for experimental comparisons.

## Similarities
- Both operate on inputs shaped `(N, C, T, V)` and follow the GCN→TCN→residual pattern.
- Both perform global pooling → classifier and are suitable for multi-stream fusion.

## When to use which
- Use the modular backbone (`models/graph.py` + `models/tgcn.py`) when you want:
  - adaptive topology, edge-importance weighting, DropGraph, or reuse across models.
  - early-fusion workflows that need `extract_features=True`.
- Keep `models/stgcn_multi.py` as a lightweight comparison implementation, or
  refactor it to use the modular utilities if you want consistency.

## Suggested next steps
- Option A — keep both: add unit tests and a short comment in `stgcn_multi.py`
  referencing the shared implementation for clarity.
- Option B — refactor `stgcn_multi.py` to import `models/graph.py` and `models/tgcn.py`
  and preserve its original hyperparameters.

Generated: May 12, 2026
