"""
stgcn.py — Multi-Stream Spatial-Temporal Graph Convolutional Network
=====================================================================
Based on: Yan et al. "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition" (AAAI 2018)

Three input streams:
  1. Joint positions  (N, 2, T, 51)
  2. Bone vectors     (N, 2, T, 51)
  3. Motion (Δjoint)  (N, 2, T, 51)

Each stream is processed by an independent ST-GCN branch.
Scores are fused by addition before softmax.

Joint layout (51 joints, 102 features):
  0       = nose
  1–2     = shoulders (L, R)
  3–4     = elbows (L, R)
  5–6     = wrists (L, R)
  7–8     = hips (L, R)
  9–29    = left hand  (wrist root = 9)
  30–50   = right hand (wrist root = 30)
"""

import torch
import torch.nn as nn
import numpy as np

NUM_JOINTS = 51


# ══════════════════════════════════════════════════════════════════════════════
# SKELETON GRAPH
# ══════════════════════════════════════════════════════════════════════════════

UPPER_BODY_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
]

HAND_EDGES_RAW = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LEFT_HAND_EDGES  = [(a+9,  b+9)  for a, b in HAND_EDGES_RAW]
RIGHT_HAND_EDGES = [(a+30, b+30) for a, b in HAND_EDGES_RAW]
CROSS_EDGES      = [(5, 9), (6, 30)]
ALL_EDGES        = UPPER_BODY_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + CROSS_EDGES


def build_adjacency(num_joints=NUM_JOINTS, edges=ALL_EDGES):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    D       = A.sum(axis=1)
    D_isqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_mat   = np.diag(D_isqrt)
    return torch.tensor(D_mat @ A @ D_mat, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ST-GCN BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(torch.einsum('nctv,vw->nctw', x, self.A)))


class TemporalConv(nn.Module):
    def __init__(self, ch, ks=9, stride=1, dropout=0.0):
        super().__init__()
        self.conv    = nn.Conv2d(ch, ch, (ks,1), (stride,1), ((ks-1)//2, 0))
        self.bn      = nn.BatchNorm2d(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, ks=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn  = GraphConv(in_ch, out_ch, A)
        self.tcn  = TemporalConv(out_ch, ks, stride, dropout)
        self.relu = nn.ReLU(inplace=True)
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride,1)),
                nn.BatchNorm2d(out_ch)
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        return self.relu(self.tcn(self.relu(self.gcn(x))) + self.residual(x))


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STREAM
# ══════════════════════════════════════════════════════════════════════════════

class STGCNStream(nn.Module):
    """One ST-GCN stream processing (N, 2, T, V) input."""

    def __init__(self, num_classes, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()
        A = build_adjacency(num_joints)

        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        cfg = [
            (in_channels, 64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
        ]

        self.layers = nn.ModuleList([
            STGCNBlock(ic, oc, A, stride=s,
                       dropout=dropout if i > 0 else 0.0)
            for i, (ic, oc, s) in enumerate(cfg)
        ])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x):
        N, C, T, V = x.shape
        xbn = x.permute(0,1,3,2).contiguous().view(N, C*V, T)
        xbn = self.bn_in(xbn)
        x   = xbn.view(N, C, V, T).permute(0,1,3,2)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.gap(x).view(N, -1))   # logits


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-STREAM ST-GCN
# ══════════════════════════════════════════════════════════════════════════════

class STGCN(nn.Module):
    """
    Multi-stream ST-GCN.

    Streams:
      joint  : raw joint positions
      bone   : bone vectors (child - parent)
      motion : frame-to-frame joint delta

    Fusion: sum of logits from all streams.

    Input  : dict with keys 'joint', 'bone', 'motion'
             each tensor: (N, 2, T, V)
    Output : (N, num_classes)
    """

    def __init__(self, num_classes=20, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()

        self.stream_joint  = STGCNStream(num_classes, in_channels,
                                         num_joints, dropout)
        self.stream_bone   = STGCNStream(num_classes, in_channels,
                                         num_joints, dropout)
        self.stream_motion = STGCNStream(num_classes, in_channels,
                                         num_joints, dropout)

    def forward(self, x):
        """
        x : dict {'joint': tensor, 'bone': tensor, 'motion': tensor}
            OR single tensor (N, 2, T, V) — uses joint stream only
        """
        if isinstance(x, dict):
            out = (self.stream_joint(x['joint']) +
                   self.stream_bone(x['bone'])   +
                   self.stream_motion(x['motion']))
        else:
            out = self.stream_joint(x)
        return out


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N, C, T, V = 4, 2, 64, NUM_JOINTS
    model = STGCN(num_classes=20)

    x = {
        'joint' : torch.randn(N, C, T, V),
        'bone'  : torch.randn(N, C, T, V),
        'motion': torch.randn(N, C, T, V),
    }
    out   = model(x)
    total = sum(p.numel() for p in model.parameters())

    print(f"Input  : {x['joint'].shape} × 3 streams")
    print(f"Output : {out.shape}")
    print(f"Params : {total:,}")
    print("✅ Multi-stream ST-GCN OK")