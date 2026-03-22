"""
stgcn.py — Spatial-Temporal Graph Convolutional Network
=========================================================
Adapted for filtered MediaPipe skeleton after normalize.py:

  Upper body : 9 joints  (nose, shoulders, elbows, wrists, hips)
  Left Hand  : 21 joints (relative to left wrist)
  Right Hand : 21 joints (relative to right wrist)
  TOTAL      : 51 joints = 102 features (x, y per joint)

Input  : (batch, 2, T, 51)
Output : (batch, num_classes)

Graph edges follow MediaPipe topology but re-indexed after filtering.

Joint index map after filter_joints():
  0  = nose
  1  = left shoulder
  2  = right shoulder
  3  = left elbow
  4  = right elbow
  5  = left wrist
  6  = right wrist
  7  = left hip
  8  = right hip
  9–29  = left hand  (wrist=9, thumb=10-13, index=14-17, middle=18-21,
                      ring=22-25, pinky=26-29)
  30–50 = right hand (same layout, offset by 21)
"""

import torch
import torch.nn as nn
import numpy as np

NUM_JOINTS = 51   # 9 upper body + 21 left hand + 21 right hand


# ══════════════════════════════════════════════════════════════════════════════
# SKELETON GRAPH
# ══════════════════════════════════════════════════════════════════════════════

# Upper body edges (re-indexed: nose=0, lsh=1, rsh=2, lel=3, rel=4,
#                               lwr=5, rwr=6, lhi=7, rhi=8)
UPPER_BODY_EDGES = [
    (0, 1), (0, 2),       # nose → shoulders
    (1, 2),               # shoulder bar
    (1, 3), (3, 5),       # left arm
    (2, 4), (4, 6),       # right arm
    (1, 7), (2, 8),       # shoulders → hips
    (7, 8),               # hip bar
]

# Hand edges (generic, applied to both hands with offset)
HAND_EDGES_RAW = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm knuckles
]

LEFT_HAND_OFFSET  = 9
RIGHT_HAND_OFFSET = 30

LEFT_HAND_EDGES  = [(a + LEFT_HAND_OFFSET,  b + LEFT_HAND_OFFSET)  for a, b in HAND_EDGES_RAW]
RIGHT_HAND_EDGES = [(a + RIGHT_HAND_OFFSET, b + RIGHT_HAND_OFFSET) for a, b in HAND_EDGES_RAW]

# Cross edges: pose wrist → hand wrist root
CROSS_EDGES = [
    (5, LEFT_HAND_OFFSET),    # left wrist → left hand root
    (6, RIGHT_HAND_OFFSET),   # right wrist → right hand root
]

ALL_EDGES = UPPER_BODY_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + CROSS_EDGES


def build_adjacency(num_joints=NUM_JOINTS, edges=ALL_EDGES):
    """Build normalized adjacency matrix D^(-1/2) A D^(-1/2)."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)

    D        = A.sum(axis=1)
    D_isqrt  = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_mat    = np.diag(D_isqrt)
    A_norm   = D_mat @ A @ D_mat

    return torch.tensor(A_norm, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ST-GCN BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        return self.bn(self.conv(x))


class TemporalConv(nn.Module):
    def __init__(self, ch, ks=9, stride=1, dropout=0.0):
        super().__init__()
        self.conv    = nn.Conv2d(ch, ch, (ks, 1), (stride, 1), ((ks-1)//2, 0))
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
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        return self.relu(self.tcn(self.relu(self.gcn(x))) + self.residual(x))


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class STGCN(nn.Module):
    """
    ST-GCN for Sign Language Recognition.

    Input  : (N, 2, T, 51)   — 2 channels (x,y), T frames, 51 joints
    Output : (N, num_classes)
    """

    def __init__(self, num_classes=20, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()

        A = build_adjacency(num_joints)

        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        cfg = [
        (in_channels, 128, 1),
        (128,         128, 1),
        (128,         128, 1),
        (128,         256, 2),
        (256,         256, 1),
        (256,         256, 1),
        (256,         512, 2),
        (512,         512, 1),
        (512,         512, 1),
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
        xbn = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        xbn = self.bn_in(xbn)
        x   = xbn.view(N, C, V, T).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.gap(x).view(N, -1))


# ══════════════════════════════════════════════════════════════════════════════
# RESHAPE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_to_graph(X):
    """
    Convert flat keypoints to ST-GCN graph format.

    Args:
        X : numpy array (N, T, F)
              F = 102 (after normalize.py) or 150 (raw)

    Returns:
        torch.Tensor (N, 2, T, V)
          V = F // 2
    """
    N, T, F = X.shape
    V = F // 2
    # (N, T, F) → (N, T, V, 2) → (N, 2, T, V)
    X_graph = X.reshape(N, T, V, 2).transpose(0, 3, 1, 2)
    return torch.tensor(X_graph, dtype=torch.float32)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = STGCN(num_classes=20)
    dummy  = torch.randn(4, 2, 64, NUM_JOINTS)
    output = model(dummy)
    total  = sum(p.numel() for p in model.parameters())

    print(f"Input  : {dummy.shape}")
    print(f"Output : {output.shape}")
    print(f"Params : {total:,}")
    print("✅ ST-GCN forward pass OK")