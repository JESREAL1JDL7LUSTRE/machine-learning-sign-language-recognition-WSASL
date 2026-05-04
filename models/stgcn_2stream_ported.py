"""
stgcn_2stream_ported.py
=======================
Original 2-Stream ST-GCN (Yan et al. AAAI 2018), ported to modern PyTorch.

Two late-fusion streams: joint positions + bone vectors.
Logits from both streams are summed before softmax.

Input  : x (N, 2, T, V) joint positions
         bone (N, 2, T, V) bone vectors (optional, zeros if omitted)
Output : (N, num_classes)
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from models.st_gcn import Model as ST_GCN


class TwoStreamPortedSTGCN(nn.Module):
    """
    2-stream ST-GCN: joint positions + bone vectors, late fusion.

    Each stream is an independent Yan et al. ST-GCN.
    Final output = sum of both streams' logits.

    Args:
        in_channels                : input channels (2 for x, y)
        num_class                  : number of sign classes
        graph_args                 : Graph() constructor arguments (layout, strategy)
        edge_importance_weighting  : learnable per-layer edge weights
        adaptive_graph             : add learnable adjacency on top of fixed graph
        drop_graph_prob            : DropGraph probability (0 = disabled)
        **kwargs                   : passed to each ST_GCN stream (dropout, etc.)
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

        if graph_args is None:
            graph_args = {"layout": "mediapipe_51", "strategy": "spatial"}

        stream_kwargs = dict(
            edge_importance_weighting = edge_importance_weighting,
            adaptive_graph            = adaptive_graph,
            drop_graph_prob           = drop_graph_prob,
            extract_features          = False,
            **kwargs,
        )

        self.joint_stream = ST_GCN(in_channels, num_class, graph_args, **stream_kwargs)
        self.bone_stream  = ST_GCN(in_channels, num_class, graph_args, **stream_kwargs)

    def forward(self, x, bone=None):
        """
        Args:
            x    : (N, 2, T, V) joint positions
            bone : (N, 2, T, V) bone vectors (zeros if None)
        Returns:
            logits (N, num_class)
        """
        if bone is None:
            bone = torch.zeros_like(x)
        return self.joint_stream(x) + self.bone_stream(bone)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GRAPH_ARGS = {"layout": "mediapipe_51", "strategy": "spatial"}

    model = TwoStreamPortedSTGCN(
        in_channels = 2,
        num_class   = 20,
        graph_args  = GRAPH_ARGS,
    )

    x    = torch.randn(4, 2, 64, 51)
    bone = torch.randn(4, 2, 64, 51)
    out  = model(x, bone)

    total = sum(p.numel() for p in model.parameters())
    print(f"Joint input : {x.shape}")
    print(f"Bone input  : {bone.shape}")
    print(f"Output      : {out.shape}")
    print(f"Params      : {total:,}")
    print("✅ TwoStreamPortedSTGCN OK")