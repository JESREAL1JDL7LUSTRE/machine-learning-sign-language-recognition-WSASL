"""
stgcn_2stream_ported.py — Two-Stream ST-GCN (Ported to Modern PyTorch)
=======================================================================

A faithful port of the original two-stream Spatial-Temporal Graph
Convolutional Network from Yan et al. "Spatial Temporal Graph Convolutional
Networks for Skeleton-Based Action Recognition" (AAAI 2018).

Two streams with late fusion:
    1. Joint stream  : raw joint (x, y) positions — captures body pose
    2. Bone stream   : bone vectors (child - parent) — captures limb orientation

Fusion strategy: late fusion — logits from both streams are SUMMED.
    output = joint_stream_logits + bone_stream_logits

This is one of three comparison models evaluated in main.py.

Usage:
    model = TwoStreamPortedSTGCN(in_channels=2, num_class=20,
                                  graph_args={'layout': 'mediapipe_51', ...})
    logits = model(x_joint, bone=x_bone)

Key differences from the original paper's implementation:
    - Uses a reduced-width ST-GCN backbone (32/64/128 instead of 64/128/256)
      for better generalization on small datasets.
    - Optionally uses AdaptiveAdjacency (Zhang et al. 2020) on top of the
      fixed skeleton graph.
    - Optionally uses DropGraph (Jiang et al. 2021) for regularization.
    - Removed deprecated torch.autograd.Variable wrappers.
    - Uses modern einsum notation.
"""

import os, sys
# Ensure the project root is on sys.path so local 'models.*' imports work
# when this file is run directly (e.g. python models/stgcn_2stream_ported.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
# Import the backbone single-stream ST-GCN
from models.st_gcn_backbone import Model as ST_GCN


class TwoStreamPortedSTGCN(nn.Module):
    """Two-stream ST-GCN: joint positions + bone vectors, late fusion.

    Each stream is a completely independent Yan et al. ST-GCN backbone
    (models/st_gcn.py) with the same configuration. There is no shared
    weight between the two streams.

    The final class prediction is made by summing both streams' logits,
    which is equivalent to multiplying their class posteriors (assuming
    they are conditionally independent given the input video).

    Args:
        in_channels               (int)  : Input channels (2 for x, y coordinates).
        num_class                 (int)  : Number of sign language classes.
        graph_args                (dict) : Arguments for Graph() in each stream.
                                          Must include 'layout' and 'strategy'.
        edge_importance_weighting (bool) : Learnable per-layer, per-edge importance.
        adaptive_graph            (bool) : Add learnable adjacency correction.
        drop_graph_prob           (float): DropGraph probability during training.
        **kwargs                  (dict) : Additional kwargs for ST_GCN (e.g. dropout).
    """

    def __init__(
        self,
        in_channels               = 2,
        num_class                 = 20,
        graph_args                = None,
        edge_importance_weighting = True,
        adaptive_graph            = True,
        drop_graph_prob           = 0.1,
        **kwargs,
    ):
        super().__init__()

        # Default graph: MediaPipe 51-joint with spatial partitioning
        if graph_args is None:
            graph_args = {"layout": "mediapipe_51", "strategy": "spatial"}

        # Shared configuration for both streams (extract_features=False
        # because both streams output class logits for late fusion)
        stream_kwargs = dict(
            edge_importance_weighting = edge_importance_weighting,
            adaptive_graph            = adaptive_graph,
            drop_graph_prob           = drop_graph_prob,
            extract_features          = False,   # output logits, not features
            **kwargs,
        )

        # ── Joint stream ──────────────────────────────────────────────────────
        # Processes raw normalized (x, y) joint coordinates.
        # Captures the overall body pose (hand shape, arm position).
        self.joint_stream = ST_GCN(in_channels, num_class, graph_args, **stream_kwargs)

        # ── Bone stream ───────────────────────────────────────────────────────
        # Processes bone vectors (child_joint - parent_joint).
        # Captures limb orientations independently of global position.
        self.bone_stream  = ST_GCN(in_channels, num_class, graph_args, **stream_kwargs)

    def forward(self, x, bone=None):
        """Late fusion forward pass.

        Args:
            x    (Tensor): Joint positions, shape (N, 2, T, V).
            bone (Tensor): Bone vectors, shape (N, 2, T, V).
                           If None, zeros are used (bone stream contributes nothing).

        Returns:
            Tensor: Class logits (N, num_class), summed across both streams.
        """
        # Fill in zeros if bone data is not provided (graceful degradation)
        if bone is None:
            bone = torch.zeros_like(x)

        # Late fusion: sum the independent logits from both streams
        # Equivalent to log-linear combination of class posteriors
        return self.joint_stream(x) + self.bone_stream(bone)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GRAPH_ARGS = {"layout": "mediapipe_51", "strategy": "spatial"}

    model = TwoStreamPortedSTGCN(
        in_channels = 2,
        num_class   = 20,
        graph_args  = GRAPH_ARGS,
    )

    # Dummy batch: 4 samples, 2 channels (x,y), 64 frames, 51 joints
    x    = torch.randn(4, 2, 64, 51)
    bone = torch.randn(4, 2, 64, 51)
    out  = model(x, bone)

    total = sum(p.numel() for p in model.parameters())
    print(f"Joint input : {x.shape}")
    print(f"Bone input  : {bone.shape}")
    print(f"Output      : {out.shape}")
    print(f"Params      : {total:,}")
    print("✅ TwoStreamPortedSTGCN OK")