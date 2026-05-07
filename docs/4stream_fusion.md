# 4-Stream Fusion ST-GCN

Implementation: [models/st_gcn_twostream.py](models/st_gcn_twostream.py)
Runner: [main.py](main.py)
Training script: [training/train.py](training/train.py)
Backbone: [models/st_gcn.py](models/st_gcn.py)
Graph utils: [models/graph.py](models/graph.py), [models/tgcn.py](models/tgcn.py)

## What it is
A four-stream ST-GCN that fuses joint, motion, bone, and bone-motion streams. The default path is **early fusion**, where each stream outputs features (not logits), those features are concatenated, and a shared MLP classifier produces the final logits.

## Streams
- Joint stream: raw joint positions.
- Motion stream: frame-to-frame motion for joints.
- Bone stream: bone vectors (child - parent style).
- Bone-motion stream: motion of bones across frames.

## Inputs and outputs
- Inputs are `(N, 2, T, V)` tensors for each stream.
- Output is logits `(N, num_class)`.

## Data flow (early fusion, default)
1. Each stream is a `ST_GCN` with `extract_features=True`, which returns a 128-d feature vector per sample.
2. Concatenate four features to a 512-d vector.
3. Apply the fusion MLP:
   - `Linear(512 -> 256)`
   - `LayerNorm` (safer for small batches)
   - `ReLU`
   - `Dropout(0.4)`
   - `Linear(256 -> num_class)`

## Data flow (late fusion, optional)
If `early_fusion=False`, each stream returns logits and the model sums the four logits. This is kept for ablation and compatibility but is not the default path.

## Core modules and functions
- `Model.__init__` builds four `ST_GCN` streams and the fusion head when `early_fusion=True`.
- `Model._compute_motion` builds a smoothed temporal derivative when motion is not provided.
- `Model.forward` accepts `x`, plus optional `motion`, `bone`, and `bone_motion`. Missing streams fall back to zeros so the model still runs.
- `ST_GCN` uses adaptive adjacency and DropGraph to improve robustness to missing joints.

## Graph and convolution details
- Graph layout: `mediapipe_51`, using spatial partitioning of the adjacency.
- Graph convolution: `ConvTemporalGraphical` applies temporal conv then adjacency multiplication.
- Each `st_gcn` block uses residual connections and temporal convolutions, similar to the original ST-GCN.

## Runner integration
- `run_4stream_fusion` in [main.py](main.py) builds this model with:
  - `edge_importance_weighting = True`
  - `adaptive_graph = True`
  - `drop_graph_prob = 0.1`
  - `early_fusion = True`
- The forward function explicitly passes all four streams: `model(xj, motion=xm, bone=xb, bone_motion=xbm)`.

## Training script notes
- [training/train.py](training/train.py) trains this model with:
  - Label smoothing loss.
  - Optional class-weighted loss.
  - SWA (Stochastic Weight Averaging) after early stopping.
  - Data augmentation on each stream (jitter, scaling, rotation, flips, time-warp, random move).

## Key design choices
- Early fusion allows cross-stream interaction inside the classifier.
- Reduced-width `ST_GCN` keeps the parameter count smaller for limited datasets.
- LayerNorm in the fusion head helps with very small batch sizes (default batch size is 4).
