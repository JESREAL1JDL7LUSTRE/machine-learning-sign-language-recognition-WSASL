"""
st_gcn_twostream.py
===================
Two-stream ST-GCN from:
  Yan et al. "Spatial Temporal Graph Convolutional Networks for
  Skeleton-Based Action Recognition", AAAI 2018.
  https://github.com/yysijie/st-gcn

Ported to modern PyTorch:
  - Removed torch.autograd.Variable
  - Input shape (N, C, T, V) instead of (N, C, T, V, M)
  - Motion computed as second-order difference (original formula)
"""

import torch
import torch.nn as nn

from models.st_gcn import Model as ST_GCN


class Model(nn.Module):
    """Two-stream ST-GCN.

    Stream 1 (origin): raw joint positions
    Stream 2 (motion): second-order temporal difference
        m[t] = x[t] - 0.5*x[t+1] - 0.5*x[t-1]

    Final score = origin_stream(x) + motion_stream(m)

    Input shape : (N, C, T, V)
    Output shape: (N, num_class)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Second-order motion (original formula from paper)
        # Pad with zeros at start and end
        zeros = torch.zeros(N, C, 1, V, device=x.device, dtype=x.dtype)
        m = torch.cat([
            zeros,
            x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
            zeros
        ], dim=2)

        return self.origin_stream(x) + self.motion_stream(m)