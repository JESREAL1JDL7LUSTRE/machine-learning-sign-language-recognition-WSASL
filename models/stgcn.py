"""
stgcn.py — Spatial-Temporal Graph Convolutional Network
=========================================================
Based on: Yan et al. "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition" (AAAI 2018)

Adapted for MediaPipe 75-joint skeleton:
  - 33 pose joints       (indices 0–32)
  - 21 left hand joints  (indices 33–53)
  - 21 right hand joints (indices 54–74)

Input shape : (batch, 2, T, 75)
               2 = x, y channels
               T = number of frames (64)
               75 = number of joints

Output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# SKELETON GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe Pose connections (33 joints, 0-indexed)
POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Hand connections (21 joints each)
HAND_EDGES_RAW = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LEFT_HAND_EDGES  = [(a + 33, b + 33) for a, b in HAND_EDGES_RAW]
RIGHT_HAND_EDGES = [(a + 54, b + 54) for a, b in HAND_EDGES_RAW]

# Connect pose wrists to hand roots
CROSS_EDGES = [(15, 33), (16, 54)]

ALL_EDGES  = POSE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + CROSS_EDGES
NUM_JOINTS = 75


def build_adjacency(num_joints=NUM_JOINTS, edges=ALL_EDGES):
    """Build normalized adjacency matrix."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat

    return torch.tensor(A_norm, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ST-GCN BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    """Graph convolution: H' = A * H * W"""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        return x


class TemporalConv(nn.Module):
    """Temporal convolution over the time dimension."""

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0)
        )
        self.bn      = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
    GraphConv → ReLU → TemporalConv → ReLU + residual
    """

    def __init__(self, in_channels, out_channels, A,
                 kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn  = GraphConv(in_channels, out_channels, A)
        self.tcn  = TemporalConv(out_channels, kernel_size, stride, dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x   = self.relu(self.gcn(x))
        x   = self.tcn(x)
        return self.relu(x + res)


# ══════════════════════════════════════════════════════════════════════════════
# FULL ST-GCN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class STGCN(nn.Module):
    """
    ST-GCN for Isolated Sign Language Recognition.

    Input  : (batch, 2, T, 75)
    Output : (batch, num_classes)
    """

    def __init__(self, num_classes=20, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()

        A = build_adjacency(num_joints)

        # Input normalization
        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN layers: (in_ch, out_ch, stride)
        cfg = [
            (in_channels, 64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
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

        # Normalize input
        xbn = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        xbn = self.bn_in(xbn)
        x   = xbn.view(N, C, V, T).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        x = self.gap(x).view(N, -1)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: reshape flat keypoints → graph format
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_to_graph(X):
    """
    Reshape flat keypoints to ST-GCN input format.

    Args:
        X : numpy array (N, T, 150)

    Returns:
        torch.Tensor (N, 2, T, 75)
    """
    N, T, F = X.shape
    assert F == 150, f"Expected 150 features (MediaPipe), got {F}. Use MediaPipe backend."

    # (N, T, 150) → (N, T, 75, 2) → (N, 2, T, 75)
    X_graph = X.reshape(N, T, 75, 2).transpose(0, 3, 1, 2)
    return torch.tensor(X_graph, dtype=torch.float32)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = STGCN(num_classes=20)
    dummy  = torch.randn(4, 2, 64, 75)
    output = model(dummy)

    total  = sum(p.numel() for p in model.parameters())
    print(f"Input  : {dummy.shape}")
    print(f"Output : {output.shape}")
    print(f"Params : {total:,}")
    print("✅ ST-GCN forward pass OK")