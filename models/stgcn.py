"""
stgcn.py — Multi-Stream Spatial-Temporal Graph Convolutional Network
=====================================================================

This module implements the three-stream ST-GCN (STGCN) architecture used
as one of the comparison models in the sign language recognition study.

Based on: Yan et al. "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition" (AAAI 2018).

Three input streams (each an independent ST-GCN branch):
    1. Joint positions   (N, 2, T, 51)  — raw (x, y) coordinates per joint per frame
    2. Bone vectors      (N, 2, T, 51)  — child_joint - parent_joint displacement
    3. Joint motion      (N, 2, T, 51)  — frame-to-frame coordinate delta

Fusion strategy: late fusion by summing logits from all three streams.
    Final_logits = joint_logits + bone_logits + motion_logits

Joint layout (51 joints, 102 features = 51 × 2):
    0       = nose
    1–2     = shoulders (left, right)
    3–4     = elbows (left, right)
    5–6     = wrists (left, right)
    7–8     = hips (left, right)
    9–29    = left hand  (wrist root = 9, joints 9–29)
    30–50   = right hand (wrist root = 30, joints 30–50)

Components:
    build_adjacency  — Constructs and normalizes the 51-joint adjacency matrix.
    GraphConv        — Graph convolution block (spatial message passing).
    TemporalConv     — Temporal convolution block (motion over time).
    STGCNBlock       — One complete ST-GCN block with residual connection.
    STGCNStream      — A single ST-GCN stream (full 6-layer backbone + classifier).
    STGCN            — The three-stream wrapper (joint, bone, motion) with late fusion.
"""

import torch
import torch.nn as nn
import numpy as np

# Total number of joints in the MediaPipe 51-joint layout
NUM_JOINTS = 51


# ══════════════════════════════════════════════════════════════════════════════
# SKELETON GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

# Upper body connections between the 9 torso/arm joints (indices 0–8)
UPPER_BODY_EDGES = [
    (0, 1), (0, 2), (1, 2),    # nose ↔ both shoulders
    (1, 3), (3, 5),             # left shoulder → elbow → wrist
    (2, 4), (4, 6),             # right shoulder → elbow → wrist
    (1, 7), (2, 8), (7, 8),    # hips and shoulder-hip connections
]

# Hand topology (generic, using local indices 0–20 where 0=wrist root)
# Each chain: wrist → finger base → middle → tip
HAND_EDGES_RAW = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # knuckle row (metacarpal connections)
]

# Shift generic hand indices to the global joint space:
# Left hand occupies joints 9–29 (offset +9 from 0-based)
LEFT_HAND_EDGES  = [(a+9,  b+9)  for a, b in HAND_EDGES_RAW]
# Right hand occupies joints 30–50 (offset +30 from 0-based)
RIGHT_HAND_EDGES = [(a+30, b+30) for a, b in HAND_EDGES_RAW]
# Cross connections: left wrist (joint 5) → left hand root (9)
#                   right wrist (joint 6) → right hand root (30)
CROSS_EDGES      = [(5, 9), (6, 30)]
# Complete edge list for all 51 joints
ALL_EDGES        = UPPER_BODY_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + CROSS_EDGES


def build_adjacency(num_joints=NUM_JOINTS, edges=ALL_EDGES):
    """Build a symmetric, degree-normalized adjacency matrix for the skeleton.

    Steps:
        1. Build binary adjacency matrix from the edge list.
        2. Add self-loops (diagonal = 1) so each joint aggregates its own features.
        3. Compute degree matrix D (diagonal of row-sums).
        4. Apply symmetric normalization: A_norm = D^{-1/2} @ A @ D^{-1/2}
           (standard GCN normalization, Kipf & Welling 2017).

    Args:
        num_joints (int) : Number of nodes in the graph (default: 51).
        edges      (list): List of (i, j) undirected edge tuples.

    Returns:
        Tensor: Normalized adjacency matrix, shape (num_joints, num_joints).
    """
    # Initialize raw binary adjacency matrix
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            # Set both directions (undirected graph)
            A[i, j] = A[j, i] = 1.0
    # Add self-loops so each joint also receives its own features
    np.fill_diagonal(A, 1.0)

    # Compute degree and its square root inverse for symmetric normalization
    D       = A.sum(axis=1)                              # row-sum = degree
    D_isqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)   # D^{-1/2}, safe division
    D_mat   = np.diag(D_isqrt)                           # diagonal matrix

    # Symmetric normalization: D^{-1/2} @ A @ D^{-1/2}
    return torch.tensor(D_mat @ A @ D_mat, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ST-GCN BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    """Spatial graph convolution: aggregates neighbour features using adjacency.

    For each joint v, computes:
        h_v = BatchNorm( Conv1x1( Σ_w A[v,w] * x_w ) )

    The adjacency multiplication is performed via einsum, which efficiently
    computes the weighted sum of all source joints for each target joint.

    Args:
        in_ch  (int)  : Input feature channels.
        out_ch (int)  : Output feature channels.
        A      (Tensor): Normalized adjacency matrix, shape (V, V), registered as buffer.
    """

    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        # Register adjacency as a non-trainable buffer (moves with .to(device))
        self.register_buffer('A', A)
        # 1×1 convolution to project channels (applied after graph aggregation)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """Apply graph convolution.

        Args:
            x (Tensor): Shape (N, C, T, V) — batch, channels, frames, joints.

        Returns:
            Tensor: Shape (N, out_ch, T, V) — graph-convolved features.
        """
        # 'nctv,vw->nctw': for each target joint w, sum over source joints v
        # weighted by A[v, w]. Then project channels and normalize.
        return self.bn(self.conv(torch.einsum('nctv,vw->nctw', x, self.A)))


class TemporalConv(nn.Module):
    """Temporal convolution: models sequential motion over adjacent frames.

    Uses a kernel of size (ks, 1) — ks frames × 1 joint.
    Padding = (ks-1)//2 gives 'same' temporal padding (no length change at stride=1).
    Stride > 1 reduces the temporal resolution (downsampling).

    Args:
        ch      (int)  : Number of input and output channels (same-width block).
        ks      (int)  : Temporal kernel size (default: 9, standard for ST-GCN).
        stride  (int)  : Temporal stride. 1 = same length; 2 = halve length.
        dropout (float): Dropout probability applied after BN.
    """

    def __init__(self, ch, ks=9, stride=1, dropout=0.0):
        super().__init__()
        self.conv    = nn.Conv2d(ch, ch, (ks, 1), (stride, 1), ((ks-1)//2, 0))
        self.bn      = nn.BatchNorm2d(ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply temporal convolution.

        Args:
            x (Tensor): Shape (N, ch, T, V).

        Returns:
            Tensor: Shape (N, ch, T//stride, V).
        """
        return self.dropout(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """One full ST-GCN block: GraphConv → ReLU → TemporalConv → + residual → ReLU.

    Data flow:
        x → GraphConv → ReLU → TemporalConv → + residual(x) → ReLU → output

    The residual path handles mismatched channel counts or strides via a
    1×1 conv with matching stride.

    Args:
        in_ch   (int)  : Input channels.
        out_ch  (int)  : Output channels.
        A       (Tensor): Adjacency matrix for the GraphConv.
        ks      (int)  : Temporal kernel size for TemporalConv.
        stride  (int)  : Temporal stride (1 or 2).
        dropout (float): Dropout in TemporalConv (not applied to first block).
    """

    def __init__(self, in_ch, out_ch, A, ks=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn  = GraphConv(in_ch, out_ch, A)
        self.tcn  = TemporalConv(out_ch, ks, stride, dropout)
        self.relu = nn.ReLU(inplace=True)
        # Residual connection: project if shape changes (channels or stride)
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        """Run one ST-GCN block.

        Args:
            x (Tensor): Shape (N, in_ch, T, V).

        Returns:
            Tensor: Shape (N, out_ch, T//stride, V).
        """
        # Spatial graph conv → ReLU → temporal conv → add residual → ReLU
        return self.relu(self.tcn(self.relu(self.gcn(x))) + self.residual(x))


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STREAM
# ══════════════════════════════════════════════════════════════════════════════

class STGCNStream(nn.Module):
    """One full ST-GCN stream: 6 blocks with increasing channels and 2× downsampling.

    Architecture (channel progression):
        in_channels → 64 (stride 1) → 64 (stride 1) → 128 (stride 2)
        → 128 (stride 1) → 256 (stride 2) → 256 (stride 1)

    After the 6 blocks, global average pooling collapses (T, V) → (1, 1)
    and a linear layer maps 256-d features to class logits.

    Input normalization:
        Before the first block, features are batch-normalized across the
        time axis after reshaping to (N, C*V, T).

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Input channels (2 for x, y). Default: 2.
        num_joints  (int): Number of body joints V. Default: NUM_JOINTS (51).
        dropout     (float): Dropout applied to all but the first block.
    """

    def __init__(self, num_classes, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()
        # Build the normalized adjacency matrix once and share across all blocks
        A = build_adjacency(num_joints)

        # Per-frame batch normalization of raw joint features
        # Shape expected: (N, C*V, T) — each (joint, channel) normalized over time
        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        # Block configuration: (in_channels, out_channels, stride)
        cfg = [
            (in_channels, 64,  1),   # Block 1: project to 64-d, same temporal length
            (64,          64,  1),   # Block 2: refine features
            (64,          128, 2),   # Block 3: expand to 128, halve time
            (128,         128, 1),   # Block 4: refine
            (128,         256, 2),   # Block 5: expand to 256, halve time again
            (256,         256, 1),   # Block 6: refine
        ]

        # Build all ST-GCN blocks; no dropout on the first block (i==0)
        self.layers = nn.ModuleList([
            STGCNBlock(ic, oc, A, stride=s,
                       dropout=dropout if i > 0 else 0.0)
            for i, (ic, oc, s) in enumerate(cfg)
        ])

        # Global average pool: collapses both time and joints to a single vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Linear classifier: 256-d feature → num_classes logits
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x):
        """Run the full stream forward pass.

        Args:
            x (Tensor): Input, shape (N, C, T, V).

        Returns:
            Tensor: Class logits, shape (N, num_classes).
        """
        N, C, T, V = x.shape

        # Input batch normalization:
        # Reshape → (N, C*V, T) → BN → reshape back to (N, C, T, V)
        xbn = x.permute(0, 1, 3, 2).contiguous().view(N, C*V, T)
        xbn = self.bn_in(xbn)
        x   = xbn.view(N, C, V, T).permute(0, 1, 3, 2)

        # Forward through all 6 ST-GCN blocks
        for layer in self.layers:
            x = layer(x)

        # Global average pool + flatten + classify
        return self.fc(self.gap(x).view(N, -1))   # → (N, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-STREAM ST-GCN (STGCN)
# ══════════════════════════════════════════════════════════════════════════════

class STGCN(nn.Module):
    """Three-stream ST-GCN with late fusion.

    Each stream is a completely independent STGCNStream backbone.
    The three streams process different representations of the same skeleton:
        joint  : raw (x, y) coordinates — captures pose shape
        bone   : bone vectors (child - parent) — captures limb orientation
        motion : frame-to-frame joint delta — captures speed and direction

    Fusion: final logits = sum of all three stream logits before softmax.
    This is equivalent to a product of their individual class posteriors
    when the streams are conditionally independent.

    Args:
        num_classes (int)  : Number of sign categories.
        in_channels (int)  : Channels per input stream (2 for x, y).
        num_joints  (int)  : Number of joints per frame.
        dropout     (float): Dropout in each stream's ST-GCN blocks.

    Input  : dict with keys 'joint', 'bone', 'motion'
             each value is a Tensor of shape (N, 2, T, V)
             OR a single Tensor (N, 2, T, V) — uses joint stream only (fallback)
    Output : (N, num_classes) — summed logits
    """

    def __init__(self, num_classes=20, in_channels=2,
                 num_joints=NUM_JOINTS, dropout=0.3):
        super().__init__()

        # Three independent ST-GCN streams (do NOT share weights)
        self.stream_joint  = STGCNStream(num_classes, in_channels, num_joints, dropout)
        self.stream_bone   = STGCNStream(num_classes, in_channels, num_joints, dropout)
        self.stream_motion = STGCNStream(num_classes, in_channels, num_joints, dropout)

    def forward(self, x):
        """Late fusion forward pass.

        Args:
            x : dict {'joint': Tensor, 'bone': Tensor, 'motion': Tensor}
                where each Tensor is (N, 2, T, V).
                OR a single Tensor (N, 2, T, V) — uses only the joint stream.

        Returns:
            Tensor: Class logits (N, num_classes).
        """
        if isinstance(x, dict):
            # Multi-stream: sum the logits from all three independent streams
            out = (self.stream_joint(x['joint']) +
                   self.stream_bone(x['bone'])   +
                   self.stream_motion(x['motion']))
        else:
            # Fallback single-stream: only use joint coordinates
            out = self.stream_joint(x)
        return out


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N, C, T, V = 4, 2, 64, NUM_JOINTS
    model = STGCN(num_classes=20)

    # Build dummy data dict for all three streams
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