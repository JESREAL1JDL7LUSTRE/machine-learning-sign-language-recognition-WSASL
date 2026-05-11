# ST-GCN Streams and Data Flow

This document explains where the streams are defined in the codebase and how input data flows through a single ST-GCN stream and multi-stream models in this project.

## Overview

There are two related implementations in this repo:

- A three-stream model in `models/stgcn.py` that sums logits from three independent streams (`joint`, `bone`, `motion`).
- A configurable four-stream model in `models/st_gcn_twostream.py` (importing `models.st_gcn.Model`) that supports early or late fusion (`joint`, `motion`, `bone`, `bone_motion`).

Both implementations build on a single-stream ST-GCN backbone (the stream implementation). This file documents where the streams are created and the step-by-step computation inside one stream.

## Where streams are created (files & symbols)

- Three-stream wrapper: `models/stgcn.py`
  - Class: `STGCN` ΓÇö creates the three streams in `__init__` as `self.stream_joint`, `self.stream_bone`, and `self.stream_motion` and calls them in `STGCN.forward()`.
  - Single-stream class: `STGCNStream` ΓÇö implements the backbone and `forward()` that produces logits for one stream.

- Four-stream model with fusion: `models/st_gcn_twostream.py`
  - Class: `Model` ΓÇö creates `self.joint_stream`, `self.motion_stream`, `self.bone_stream`, and `self.bone_motion_stream` (instances of the single-stream `ST_GCN`), then:
    - If `early_fusion=True`: calls each stream with `extract_features=True` to obtain compact feature vectors, concatenates them, and applies a fusion MLP to obtain logits.
    - If `early_fusion=False` (late fusion): calls each stream to get logits and sums them.
  - Helper: `Model._compute_motion()` ΓÇö computes frame-to-frame motion if not provided by the caller.

## Single-stream computation (how a stream computes outputs)

The single-stream backbone (`STGCNStream`) performs these steps in `forward(x)` where `x` has shape `(N, C, T, V)` (batch, channels e.g. 2 for x/y, frames, joints):

1. Input normalization:
   - Reshape `x` to `(N, C*V, T)` and apply `BatchNorm1d` (`self.bn_in`) to normalize each `(channel ├ù joint)` across time.
   - Reshape back to `(N, C, T, V)`.

2. Spatial-temporal blocks:
   - A sequence of 6 `STGCNBlock`s (`self.layers`) executes the pattern: `GraphConv ΓåÆ ReLU ΓåÆ TemporalConv ΓåÆ + residual ΓåÆ ReLU`.
   - `GraphConv` performs adjacency-weighted aggregation using a precomputed normalized adjacency matrix `A` and `torch.einsum('nctv,vw->nctw', x, self.A)` before a 1├ù1 conv and BatchNorm.
   - `TemporalConv` is a 2D conv across time with kernel `(ks,1)` (default `ks=9`), followed by BN and optional dropout.

3. Global pooling and classification:
   - `AdaptiveAvgPool2d((1,1))` collapses `(T, V)` to a single feature vector per sample.
   - A final `Linear` (`self.fc`) maps the 256-d vector to `num_classes` logits (or, when `extract_features=True` in other variants, the stream may provide a smaller feature vector ΓÇö see the four-stream early fusion variant).

## Graph adjacency

- The adjacency used by `GraphConv` is built by `build_adjacency()` (in `models/stgcn.py`) which:
  - Builds a binary adjacency from `ALL_EDGES` (includes torso and hand connections), adds self-loops, computes degree, and returns a symmetrically normalized matrix `D^{-1/2} A D^{-1/2}` as a `torch.Tensor`.

## Fusion strategies

- Late fusion (default in `models/stgcn.py`): the wrapper calls each stream independently, obtains logits `(N, num_classes)` from each stream, and sums them:
  - `final_logits = joint_logits + bone_logits + motion_logits`.

- Early fusion (in `models/st_gcn_twostream.py` when `early_fusion=True`): each stream runs with `extract_features=True` to produce a compact feature vector (128-d). The model concatenates the 4 feature vectors ΓåÆ `(N, 512)`, then applies a fusion MLP to map to class logits. This allows cross-stream interactions before classification.

## How motion and bone are provided

- `motion`: may be supplied by the caller as a separate tensor. If not provided, `Model._compute_motion(x)` in `models/st_gcn_twostream.py` computes a centered difference approximation across time.
- `bone` and `bone_motion`: may be supplied by the caller. If omitted, the twostream wrapper defaults them to zero tensors (no information).

## Key symbols and where to inspect them

- `build_adjacency` ΓÇö adjacency matrix construction: [models/stgcn.py](models/stgcn.py)
- `GraphConv` ΓÇö spatial aggregation and einsum usage: [models/stgcn.py](models/stgcn.py)
- `TemporalConv` ΓÇö temporal conv block: [models/stgcn.py](models/stgcn.py)
- `STGCNBlock` ΓÇö one ST-GCN block (spatial + temporal + residual): [models/stgcn.py](models/stgcn.py)
- `STGCNStream.forward` ΓÇö full single-stream forward: [models/stgcn.py](models/stgcn.py)
- `STGCN` ΓÇö three-stream wrapper: [models/stgcn.py](models/stgcn.py)
- `Model` (four-stream) ΓÇö fusion strategies and `_compute_motion`: [models/st_gcn_twostream.py](models/st_gcn_twostream.py)

## Quick trace (example)

Given input `x_joint` of shape `(N, 2, T, V)` for the joint stream:

- `STGCNStream.forward(x_joint)`:
  - Normalize over time ΓåÆ pass through 6 STGCN blocks ΓåÆ gap ΓåÆ fc ΓåÆ `logits_joint` `(N, num_classes)`.

If using late fusion (four-stream model):

- `logits = joint_stream(x_joint) + motion_stream(x_motion) + bone_stream(x_bone) + bone_motion_stream(x_bone_motion)`

If using early fusion (four-stream model):

- `f_joint = joint_stream(x_joint)`  # returns 128-d features when `extract_features=True`
- `f_motion = motion_stream(x_motion)`
- `f_bone = bone_stream(x_bone)`
- `f_bone_motion = bone_motion_stream(x_bone_motion)`
- `fused = torch.cat([f_joint, f_motion, f_bone, f_bone_motion], dim=1)`  # (N,512)
- `logits = fusion_fc(fused)`

## Suggested next steps

- To trace a real example, run a tiny forward pass with dummy inputs (see the `if __name__ == "__main__"` blocks in both files).
- If you want, I can add a small runnable snippet in `docs/` showing a minimal test harness that constructs the model and runs a forward pass with dummy tensors.

---

*Document generated to explain where streams are defined and how they are computed in the codebase.*
