"""
stgcn_single.py
===============
Single-stream custom ST-GCN (Generation 2, first GCN attempt).

Wraps STGCNStream from stgcn.py to expose a consistent interface.

Input  : (N, 2, T, V)  — joint positions only
Output : (N, num_classes)
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from models.stgcn import STGCNStream, NUM_JOINTS


class SingleStreamSTGCN(nn.Module):
    """
    Single-stream ST-GCN using joint positions.

    This was the first GCN model tried — it uses only raw joint position
    features with no bone or motion stream, producing a baseline that shows
    how much multi-stream input improves results.

    Args:
        num_classes : number of sign classes
        num_joints  : number of skeleton joints (default 51 for MediaPipe)
        dropout     : spatial and temporal dropout rate
    """

    def __init__(self, num_classes=20, num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()
        self.stream = STGCNStream(
            num_classes = num_classes,
            in_channels = 2,
            num_joints  = num_joints,
            dropout     = dropout,
        )

    def forward(self, x):
        """
        Args:
            x : (N, 2, T, V) joint position tensor
        Returns:
            logits (N, num_classes)
        """
        return self.stream(x)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SingleStreamSTGCN(num_classes=20)
    x = torch.randn(4, 2, 64, NUM_JOINTS)
    print(f"Input : {x.shape}")
    print(f"Output: {model(x).shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    print("✅ SingleStreamSTGCN OK")