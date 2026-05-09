# Multi-Stream ST-GCN (3-Stream)

Implementation: [models/stgcn.py](../models/stgcn.py)  
Runner: [main.py](../main.py) (`run_multi_stream_stgcn`)

---

## What It Is

A **three-stream ST-GCN with late fusion**. Three independent ST-GCN backbone branches process joint positions, bone vectors, and joint motion separately. Their logits are summed before the final prediction.

This model is **Baseline 3** in the comparison study (the original multi-stream approach before adding the bone-motion fourth stream).

---

## Three Input Streams

| Stream | Input | What It Captures |
|---|---|---|
| **Joint** | Raw (x,y) positions | Body pose and hand shape |
| **Bone** | child − parent vectors | Limb orientation (angle) |
| **Motion** | Frame-to-frame joint delta | Velocity and movement direction |

Each stream: `(N, 2, T, 51)` — batch, channels, 64 frames, 51 joints.

---

## Architecture

```
x_joint  (N,2,T,51)    x_bone (N,2,T,51)    x_motion (N,2,T,51)
      │                       │                      │
  STGCNStream            STGCNStream            STGCNStream
  (independent)          (independent)          (independent)
      │                       │                      │
 logits (N,K)           logits (N,K)           logits (N,K)
      └──────────────── sum ──────────────────────┘
                         │
                  output (N, K)
```

Each `STGCNStream` is a **6-block backbone** (wider than the single-stream):
```
Input (N, 2, T, 51)
  → BatchNorm1d
  → STGCNBlock(2→64,  stride=1)
  → STGCNBlock(64→64, stride=1)
  → STGCNBlock(64→128,stride=2)   ← halves temporal resolution
  → STGCNBlock(128→128,stride=1)
  → STGCNBlock(128→256,stride=2)  ← halves again
  → STGCNBlock(256→256,stride=1)
  → AdaptiveAvgPool2d
  → Linear(256 → num_class)
  → logits (N, num_class)
```

**Adjacency**: Symmetric D^{-1/2}·A·D^{-1/2} normalization with self-loops. No adaptive adjacency (uses fixed graph only). No DropGraph.

---

## Differences vs. Four-Stream Model

| Feature | 3-Stream (this) | 4-Stream (best) |
|---|---|---|
| Streams | joint, bone, motion | + bone_motion |
| Fusion | Late (sum logits) | **Early** (concat features) |
| Backbone width | 64/128/256 channels | 32/64/128 (reduced) |
| Adaptive adjacency | ❌ | ✅ |
| DropGraph | ❌ | ✅ |
| Cross-stream learning | ❌ | ✅ (shared MLP) |

---

## Building Blocks

| Class | Role |
|---|---|
| `GraphConv(in_ch, out_ch, A)` | Spatial graph conv via `einsum('nctv,vw->nctw', x, A)` |
| `TemporalConv(ch, ks=9, stride)` | Conv2d along time axis, per-joint |
| `STGCNBlock(in_ch, out_ch, A, stride)` | `GraphConv → ReLU → TemporalConv + residual → ReLU` |
| `STGCNStream(num_classes, in_ch, V)` | Full 6-block stream with classifier |
| `STGCN(num_classes)` | Three-stream wrapper with late fusion |
| `build_adjacency(num_joints, edges)` | Symmetric normalized adjacency `(V, V)` |

---

## Runner Integration

```python
def run_multi_stream_stgcn(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    model = STGCN(num_classes=num_classes, in_channels=2,
                  num_joints=NUM_JOINTS, dropout=DROPOUT)
    forward_fn = lambda model, inputs: model({"joint": inputs[0],
                                               "bone": inputs[1],
                                               "motion": inputs[2]})
    return _run_kfold(make_model, forward_fn,
                      ThreeStreamDataset, (X_joint, X_bone, X_motion), y, ...)
```

---

## Notes

- If `x` is a plain tensor (not a dict), only the joint stream is used (single-stream fallback mode).
- The `build_adjacency` function in this file is self-contained (does not use the `Graph` class from `graph.py`), using simple symmetric D^{-1/2}AD^{-1/2} normalization.
