# 4-Stream Early Fusion ST-GCN

Implementation: [models/st_gcn_twostream.py](../models/st_gcn_twostream.py)  
Runner: [main.py](../main.py) (`run_4stream_fusion`) and [training/train.py](../training/train.py)  
Backbone: [models/st_gcn.py](../models/st_gcn.py)

---

## What It Is

The **current best-performing model** in this project. Four independent ST-GCN backbone streams each process a different motion representation of the skeleton. With **early fusion**, the four 128-d feature vectors are **concatenated** and passed through a shared classifier MLP — allowing the model to learn cross-stream correlations before the final decision.

---

## Four Input Streams

| Stream | Input | What It Captures |
|---|---|---|
| **Joint** | Raw normalized (x,y) coordinates | Body pose and hand shape |
| **Motion** | Frame-to-frame joint delta | Joint velocity and direction |
| **Bone** | child_joint − parent_joint vectors | Limb orientation (angle) |
| **Bone-Motion** | Frame-to-frame bone delta | Angular velocity of limbs |

Each stream: shape `(N, 2, T, 51)` — batch, (x,y) channels, 64 frames, 51 joints.

---

## Architecture (Early Fusion)

```
x_joint  (N,2,T,51)   x_motion (N,2,T,51)   x_bone (N,2,T,51)   x_bm (N,2,T,51)
      │                     │                      │                    │
  ST_GCN                ST_GCN                 ST_GCN              ST_GCN
  (extract_features=True)                     (all four use identical config)
      │                     │                      │                    │
   128-d                  128-d                  128-d               128-d
      └──────────────── concatenate ──────────────────────────────────┘
                               │
                          (N, 512)
                               │
                    Linear(512 → 256)
                    LayerNorm(256)
                    ReLU
                    Dropout(0.4)
                    Linear(256 → num_class)
                               │
                       logits (N, num_class)
```

---

## Why Early Fusion Over Late Fusion?

| Aspect | Early Fusion | Late Fusion |
|---|---|---|
| Cross-stream interaction | ✅ Before classifier | ❌ Only at final sum |
| Parameters | More (shared MLP) | Less |
| Batch-size safety | LayerNorm (safe at N=1) | BatchNorm risk |
| Typical performance | **Better** | Slightly lower |

---

## ST-GCN Backbone Details

Each of the four streams uses the same `ST_GCN` (models/st_gcn.py) configuration:

```python
ST_GCN(
    in_channels              = 2,
    num_class                = num_class,   # unused when extract_features=True
    graph_args               = {"layout": "mediapipe_51", "strategy": "spatial"},
    edge_importance_weighting= True,        # learnable per-layer edge scalars
    adaptive_graph           = True,        # AdaptiveAdjacency correction
    drop_graph_prob          = 0.1,         # DropGraph during training
    extract_features         = True,        # output 128-d vector, not logits
    dropout                  = 0.4
)
```

Channel progression per stream:
- Layers 1–4: 32 channels
- Layer 5 (stride 2): 64 channels
- Layers 6–7: 64 channels
- Layer 8 (stride 2): 128 channels
- Layers 9–10: 128 channels
- GlobalAvgPool → **128-d feature vector**

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.05 |
| LR schedule | Warmup (10 ep) + Cosine annealing |
| Batch size | 4 |
| Epochs | 300 max |
| Early stopping | Patience = 35 |
| SWA | From epoch 50 |
| Dropout | 0.4 |
| DropGraph | 0.1 |
| Label smoothing | ε = 0.1 |
| Class weights | Inverse frequency |
| Augmentation | Noise, scale, rotate, flip, frame-drop, time-warp, crop, random-move |
| K-Folds | 4 |
| Test split | 15% (stratified, held out before CV) |

---

## Runner Integration

```python
def run_4stream_fusion(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    model = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True,
        adaptive_graph=True,
        drop_graph_prob=0.1,
        early_fusion=True,
        dropout=DROPOUT,
    )
    forward_fn = lambda model, inputs: model(inputs[0], motion=inputs[1],
                                              bone=inputs[2], bone_motion=inputs[3])
    return _run_kfold(make_model, forward_fn,
                      FourStreamDataset, (X_joint, X_motion, X_bone, X_bm), y, ...)
```

---

## Saved Model

Weights saved to: `models/sign_stgcn.pth`

Load for inference:
```python
from models.st_gcn_twostream import Model as FourStreamSTGCN
model = FourStreamSTGCN(2, num_classes, graph_args, early_fusion=True)
model.load_state_dict(torch.load("models/sign_stgcn.pth"))
model.eval()
logits = model(x_joint, motion=x_motion, bone=x_bone, bone_motion=x_bm)
```
