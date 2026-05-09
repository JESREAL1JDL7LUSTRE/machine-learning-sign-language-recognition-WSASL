# models/

This directory contains all neural network model definitions used for sign language recognition. Each file implements one or more PyTorch `nn.Module` classes representing a specific model architecture.

---

## File Overview

| File | Purpose |
|---|---|
| `graph.py` | Skeleton graph construction, adaptive adjacency, DropGraph, graph utilities |
| `tgcn.py` | Core graph convolution unit (`ConvTemporalGraphical`) |
| `st_gcn.py` | Single-stream ST-GCN backbone with adaptive topology |
| `lstm.py` | TCN + Temporal Attention model (named `SignLSTM` for compatibility) |
| `stgcn.py` | Three-stream ST-GCN (joint + bone + motion, late fusion) |
| `st_gcn_twostream.py` | Four-stream ST-GCN with early or late fusion (current best) |
| `stgcn_2stream_ported.py` | Two-stream ST-GCN port (joint + bone, late fusion) |

---

## graph.py — Skeleton Graph Construction

**What it does:**  
Defines the human skeleton as a graph of joints (nodes) connected by anatomical edges. The adjacency matrix `A` encodes which joints are connected and is used in every ST-GCN layer to pass information between neighbouring joints.

**Key classes and functions:**

- **`Graph(layout, strategy, max_hop, dilation)`**  
  Builds the adjacency matrix `A` for a given skeleton layout.
  - `layout='mediapipe_51'` — 51-joint layout (9 upper body + 21+21 hand joints)
  - `layout='openpose'` — 18-joint OpenPose layout
  - `layout='ntu-rgb+d'` — 25-joint NTU RGB+D layout
  - `strategy='spatial'` — ST-GCN spatial partitioning into root/close/further subsets
  - `strategy='uniform'` — one shared adjacency matrix
  - `strategy='distance'` — one matrix per hop distance
  - Returns `self.A` of shape `(K, V, V)` where K = number of subsets, V = joints

- **`AdaptiveAdjacency(num_joints, num_subsets, alpha)`**  
  Learnable correction on top of the fixed graph: `A_combined = A_fixed + alpha * softmax(A_learn)`.  
  Allows the model to discover joint dependencies not encoded in the skeleton topology.

- **`drop_graph(x, drop_prob, training)`**  
  DropGraph regularization: randomly zeros entire joint channels during training to prevent the model from over-relying on specific joints (analogous to Dropout but on the spatial dimension).

- **`get_hop_distance(num_node, edge, max_hop)`**  
  BFS shortest-path distance matrix between all joint pairs.

- **`normalize_digraph(A)` / `normalize_undigraph(A)`**  
  Degree normalization for directed and undirected graphs respectively.

---

## tgcn.py — Graph Convolution Unit

**What it does:**  
Implements `ConvTemporalGraphical`, the fundamental building block of every ST-GCN layer. It performs the core spatial graph convolution: for each joint, it aggregates features from its graph neighbours weighted by the adjacency matrix.

**How it works:**
1. Apply a 2D convolution to project `C_in → C_out * K` channels (K = number of adjacency subsets).
2. Reshape to expose the K axis: `(N, K, C_out, T, V)`.
3. Multiply by the adjacency matrix via `einsum('nkctv,kvw->nctw', x, A)` — this aggregates each target joint's features from its source neighbours.

**Input/Output shapes:**
- Input: `(N, C_in, T, V)` + adjacency `(K, V, V)`
- Output: `(N, C_out, T_out, V)` + adjacency passthrough

---

## st_gcn.py — Single-Stream ST-GCN Backbone

**What it does:**  
The full single-stream ST-GCN model with 10 layers. This is the shared backbone used by `st_gcn_twostream.py` and `stgcn_2stream_ported.py`.

**Architecture:**
```
Input (N, 2, T, 51)
    → BatchNorm1d (per-joint channel normalization)
    → DropGraph (optional joint dropout)
    → 10 × st_gcn block:
          ConvTemporalGraphical (graph conv, spatial)
        + temporal Conv2d (kernel_size=9, captures motion)
        + residual connection
        → ReLU
    → GlobalAvgPool2d (collapses T and V dimensions)
    → 1×1 Conv2d → (N, num_class)  — OR —
    → view → (N, 128) when extract_features=True
```

**Channel progression (reduced-width for small datasets):**
- Layers 1–4: 32 channels
- Layers 5–7: 64 channels (stride-2 downsampling at layer 5)
- Layers 8–10: 128 channels (stride-2 downsampling at layer 8)

**Key arguments:**
- `edge_importance_weighting=True` — each layer learns per-edge scalar importance weights
- `adaptive_graph=True` — adds `AdaptiveAdjacency` learned correction to the fixed graph
- `drop_graph_prob=0.1` — 10% of joints dropped randomly during training
- `extract_features=True` — returns 128-d feature vector instead of logits (used in four-stream early fusion)

---

## lstm.py — TCN + Temporal Attention (SignLSTM)

**What it does:**  
A Temporal Convolutional Network (TCN) with attention-based pooling. Despite the class name `SignLSTM` (kept for checkpoint compatibility), this is NOT a recurrent model. It was designed as an alternative to LSTM that trains faster and generalizes better on small datasets.

**Architecture:**
```
Input (batch, T=64, features=150)
    → Linear projection → (batch, T, hidden=128)
    → Permute → (batch, hidden, T) for Conv1d
    → 4 × TCNBlock (dilations: 1, 2, 4, 8)
        Conv1d + BN + ReLU + Dropout (×2) + residual
    → Permute back → (batch, T, hidden)
    → TemporalAttention: weighted sum over T → (batch, hidden)
    → MLP: hidden → hidden//2 → num_classes
```

**Key design choices:**
- Dilated convolutions give exponentially growing receptive field (3 → 17 frames) without extra parameters.
- Temporal attention learns which frames in the video are most discriminative (peak gesture moment).
- TCNs are fully parallel over time — faster training than LSTMs.

---

## stgcn.py — Three-Stream ST-GCN (STGCN)

**What it does:**  
A three-stream ST-GCN with late fusion. Three independent ST-GCN branches each process a different motion representation, and their logits are summed before classification.

**Three streams:**
| Stream | Input | What it captures |
|---|---|---|
| Joint | Raw (x, y) positions | Body pose and hand shape |
| Bone | Child - parent displacement | Limb orientations |
| Motion | Frame-to-frame delta | Speed and direction of movement |

**Architecture per stream:**
```
Input (N, 2, T, 51)
    → BatchNorm1d
    → 6 × STGCNBlock (channels: 64 → 64 → 128 → 128 → 256 → 256)
    → GlobalAvgPool → Linear → (N, num_classes)
```

**Fusion:** `output = joint_logits + bone_logits + motion_logits`

---

## st_gcn_twostream.py — Four-Stream ST-GCN (Current Best)

**What it does:**  
The current best-performing model in the project. Wraps four ST-GCN backbone streams (joint, motion, bone, bone-motion) with either **early fusion** or **late fusion**.

**Four streams:**
| Stream | What it captures |
|---|---|
| Joint | Pose shape |
| Motion | Joint velocity |
| Bone | Limb orientation |
| Bone-Motion | Angular velocity / limb motion |

**Early fusion (default, recommended):**
```
Each stream → ST_GCN(extract_features=True) → 128-d vector
Concatenate → (N, 512)
→ Linear(512, 256) → LayerNorm → ReLU → Dropout(0.4)
→ Linear(256, num_class)
```
- Cross-stream interactions are captured before classification.
- LayerNorm is used instead of BatchNorm (safe at batch size 1).

**Late fusion:**
```
Each stream → ST_GCN() → logits (N, num_class)
Sum of all four logits → final output
```

**Usage:**
```python
model = Model(2, num_classes, graph_args, early_fusion=True)
logits = model(x_joint, motion=x_motion, bone=x_bone, bone_motion=x_bm)
```

---

## stgcn_2stream_ported.py — Two-Stream ST-GCN (Ported)

**What it does:**  
A faithful port of the original Yan et al. (AAAI 2018) two-stream ST-GCN to modern PyTorch. Two streams (joint + bone) with late fusion. One of the baseline comparison models.

**Key differences from original:**
- Reduced-width backbone (32/64/128 instead of 64/128/256).
- Optional `AdaptiveAdjacency` learned graph correction.
- Optional `DropGraph` regularization.
- Removed deprecated `torch.autograd.Variable` wrappers.

**Usage:**
```python
model = TwoStreamPortedSTGCN(in_channels=2, num_class=20,
                              graph_args={'layout': 'mediapipe_51', 'strategy': 'spatial'})
logits = model(x_joint, bone=x_bone)
```

---

## Saved Weights

| File | Description |
|---|---|
| `sign_lstm.pth` | Trained weights for the `SignLSTM` (TCN + Attention) model |
| `sign_stgcn.pth` | Trained weights for the four-stream ST-GCN model |

Load with:
```python
model.load_state_dict(torch.load("models/sign_stgcn.pth", map_location="cpu"))
```