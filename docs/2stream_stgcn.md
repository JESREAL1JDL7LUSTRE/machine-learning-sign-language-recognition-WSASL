# 2-Stream ST-GCN (Ported)

Implementation: [models/stgcn_2stream_ported.py](../models/stgcn_2stream_ported.py)  
Runner: [main.py](../main.py) (`run_2stream_stgcn`)  
Backbone: [models/st_gcn.py](../models/st_gcn.py)  
Graph utils: [models/graph.py](../models/graph.py), [models/tgcn.py](../models/tgcn.py)

---

## What It Is

A **two-stream, late-fusion ST-GCN** — a faithful port of Yan et al. (AAAI 2018) to modern PyTorch. Two completely independent ST-GCN backbones process different skeleton representations. Their output logits are summed before computing the final class prediction.

This model serves as **Baseline 2** in the comparison study.

---

## Inputs and Outputs

| Input | Shape | Description |
|---|---|---|
| `x` | `(N, 2, T, V)` | Joint positions — raw normalized (x, y) per frame |
| `bone` | `(N, 2, T, V)` | Bone vectors — (child_joint − parent_joint) |

- `N` = batch size, `T` = 64 frames, `V` = 51 joints, `2` = x and y channels
- `bone` is optional; zeros are used if not provided

Output: logits `(N, num_class)`

---

## Architecture

```
x (N, 2, T, 51)                   bone (N, 2, T, 51)
      │                                    │
  Joint Stream                         Bone Stream
  [ST_GCN backbone]                 [ST_GCN backbone]
      │                                    │
  logits (N, K)                       logits (N, K)
      │                                    │
      └─────────────── sum ───────────────┘
                        │
                  output (N, K)
```

Each stream is a 10-layer, reduced-width ST-GCN:
```
Input (N, 2, T, 51)
  → BatchNorm1d
  → DropGraph (optional, p=0.1)
  → 4 × [32-channel st_gcn block]     (layers 1–4)
  → 3 × [64-channel st_gcn block]     (layers 5–7, stride-2 at layer 5)
  → 3 × [128-channel st_gcn block]    (layers 8–10, stride-2 at layer 8)
  → GlobalAvgPool2d
  → 1×1 Conv2d → (N, num_class)
```

---

## Data Flow (Detailed)

1. **Build adjacency matrix** `A` from the mediapipe_51 layout with spatial strategy → shape `(K, 51, 51)`.
2. **Joint stream** runs the full 10-layer ST-GCN on `x` → logits `(N, K)`.
3. **Bone stream** runs the full 10-layer ST-GCN on `bone` → logits `(N, K)`.
4. **Sum logits** → final class scores `(N, K)`.

---

## Core Modules

| Module | File | Role |
|---|---|---|
| `TwoStreamPortedSTGCN` | `stgcn_2stream_ported.py` | Top-level two-stream wrapper |
| `ST_GCN` (`Model`) | `st_gcn.py` | Single-stream backbone (used twice) |
| `st_gcn` | `st_gcn.py` | One ST-GCN block (GraphConv + TemporalConv + residual) |
| `ConvTemporalGraphical` | `tgcn.py` | Graph convolution via einsum |
| `Graph` | `graph.py` | Skeleton adjacency matrix builder |
| `AdaptiveAdjacency` | `graph.py` | Learnable adjacency correction |
| `drop_graph` | `graph.py` | DropGraph joint regularization |

---

## Graph Details (mediapipe_51)

- **51 joints**: 9 upper body + 21 left hand + 21 right hand
- **Strategy = 'spatial'**: Partitions adjacency into `K=3` subsets per hop:
  - `a_root`: joint at same depth as centre
  - `a_close`: joint closer to centre (centripetal)
  - `a_further`: joint further from centre (centrifugal)
- **Adaptive adjacency**: `A_combined = A_fixed + 0.1 × softmax(A_learn)`

---

## Runner Integration

The training runner in `main.py`:
```python
def run_2stream_stgcn(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    model = TwoStreamPortedSTGCN(
        in_channels     = 2,
        num_class       = num_classes,
        graph_args      = {"layout": "mediapipe_51", "strategy": "spatial"},
        adaptive_graph  = True,
        drop_graph_prob = 0.1,
        dropout         = DROPOUT,
    )
    forward_fn = lambda model, inputs: model(inputs[0], bone=inputs[1])
    return _run_kfold(make_model, forward_fn, TwoStreamDataset, (X_joint, X_bone), y, ...)
```

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| **Late fusion** (sum logits) | Matches original Yan et al. paper; streams remain fully independent |
| **Reduced width** (32/64/128) | Fewer parameters → less overfitting on small datasets (~300 samples) |
| **DropGraph** (p=0.1) | Robustness to missing/occluded keypoints |
| **Adaptive adjacency** | Learns joint dependencies not in the fixed skeleton topology |
| **Bone stream** | Encodes limb orientation; invariant to global body position |
