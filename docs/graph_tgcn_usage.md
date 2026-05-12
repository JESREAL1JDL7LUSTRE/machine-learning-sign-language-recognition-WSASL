# Graph and TGCN usage in the project

This short note explains what `models/graph.py` and `models/tgcn.py` implement and where they are used in the codebase.

## `models/graph.py`
- Builds the skeleton adjacency tensor `A` with shape `(K, V, V)` for a chosen layout (e.g. `mediapipe_51`).
- Provides utilities:
  - `Graph` — constructs edges, hop distances and partitions (uniform/distance/spatial)
  - `AdaptiveAdjacency` — learnable correction added to the fixed adjacency
  - `drop_graph` — DropGraph regularizer (randomly zeroes joint channels during training)
  - helpers: `get_hop_distance`, `normalize_digraph`, `normalize_undigraph`
- Purpose: central source of the physics-based graph topology and related helpers.

## `models/tgcn.py`
- Implements `ConvTemporalGraphical`, the core graph-convolution unit used inside ST‑GCN blocks.
- Operation:
  1. 2D conv expands channels to `C_out * K` (temporal conv × channel projection)
  2. reshape to expose the `K` graph subsets
  3. einsum with `A` to aggregate neighbor contributions per subset
- Purpose: low-level spatial-temporal graph convolution operator (graph + time).

## Where they are used
- `models/st_gcn_backbone.py` (single-stream ST‑GCN backbone):
  - imports and uses `Graph` (from `models/graph.py`) to build and register the adjacency `A` buffer.
  - optionally uses `AdaptiveAdjacency` and `drop_graph`.
  - uses `ConvTemporalGraphical` (from `models/tgcn.py`) inside each ST‑GCN block.
  - This backbone is the building block for `models/st_gcn_four_stream.py` (each stream is an instance of the backbone).

- `models/st_gcn_four_stream.py` (four-stream wrapper):
  - does not import `graph.py`/`tgcn.py` directly, but depends on them indirectly via the backbone streams (`models/st_gcn_backbone.py`).

- `models/stgcn_multi.py` (three-stream implementation):
  - is self-contained: it defines its own `build_adjacency`, `GraphConv`, `TemporalConv`, and `STGCNStream`.
  - it does NOT import `models/graph.py` or `models/tgcn.py` — it implements similar functionality locally.

## Notes / actionable options
- The codebase currently has two implementations of similar functionality:
  - shared utilities: `models/graph.py` + `models/tgcn.py` + `models/st_gcn_backbone.py` (used by four-stream)
  - standalone implementation: `models/stgcn_multi.py` (used as a comparison model)
- If you want consistency, I can:
  - refactor `models/stgcn_multi.py` to reuse `models/graph.py` and `models/tgcn.py`, or
  - add cross-references in the docs and tests to make differences explicit.

---
Generated: May 12, 2026
