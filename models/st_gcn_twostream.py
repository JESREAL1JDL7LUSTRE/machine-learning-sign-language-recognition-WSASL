"""
st_gcn_twostream.py
===================
Three-stream ST-GCN extending Yan et al. AAAI 2018.

Streams:
  1. Origin  : raw joint positions
  2. Motion  : second-order temporal difference (original paper formula)
  3. Bone    : bone vectors (child - parent joint)

Final score = origin_stream(x) + motion_stream(m) + bone_stream(b)

Input shape : (N, C, T, V)
Output shape: (N, num_class)
"""

import torch
import torch.nn as nn

from models.st_gcn import Model as ST_GCN


class Model(nn.Module):
    """Three-stream ST-GCN.

    Stream 1 (origin): raw joint positions
    Stream 2 (motion): second-order temporal difference
        m[t] = x[t] - 0.5*x[t+1] - 0.5*x[t-1]
    Stream 3 (bone)  : bone vectors passed in directly

    Final score = sum of all three stream logits
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)
        self.bone_stream   = ST_GCN(*args, **kwargs)

    def forward(self, x, bone=None):
        """
        Args:
            x    : joint positions (N, C, T, V)
            bone : bone vectors    (N, C, T, V) — if None, zeros used
        """
        N, C, T, V = x.size()

        # ── Stream 2: second-order motion (original paper formula) ────────────
        zeros = torch.zeros(N, C, 1, V, device=x.device, dtype=x.dtype)
        m = torch.cat([
            zeros,
            x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
            zeros
        ], dim=2)

        # ── Stream 3: bone vectors ────────────────────────────────────────────
        if bone is None:
            bone = torch.zeros_like(x)

        return (
            self.origin_stream(x) +
            self.motion_stream(m) +
            self.bone_stream(bone)
        )