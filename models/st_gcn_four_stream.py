"""
st_gcn_twostream.py — Four-Stream ST-GCN with Early or Late Fusion
====================================================================

This module wraps four ST-GCN backbone streams (joint, motion, bone,
bone-motion) into a single model with two fusion strategies:

    Early Fusion (default, recommended):
        Each stream extracts a 128-d feature vector (with extract_features=True).
        All four feature vectors are concatenated → (N, 512).
        A shared classifier MLP maps them to class logits.
        → Allows cross-stream interactions before classification.

    Late Fusion:
        Each stream outputs full class logits independently.
        Final output = sum of all four streams' logits.
        → Streams remain fully independent throughout.

The four streams encode complementary motion cues:
    joint      : raw joint (x, y) positions — body pose shape
    motion     : frame-to-frame joint delta — velocity / speed
    bone       : child-minus-parent vectors — limb orientation
    bone_motion: frame-to-frame bone vector delta — angular velocity

Usage:
    model = Model(2, num_classes, graph_args, early_fusion=True)
    logits = model(x_joint, motion=x_motion, bone=x_bone, bone_motion=x_bm)

This model is the current best-performing architecture in the project,
used by both train.py (standalone training) and main.py (comparison runs).
"""

import torch
import torch.nn as nn

# Import the backbone ST-GCN stream (single-stream model)
from models.st_gcn_backbone import Model as ST_GCN


class Model(nn.Module):
    """Four-stream ST-GCN with configurable early or late fusion.

    Args:
        *args      : Positional args passed to each ST_GCN stream.
                     Typical: (in_channels=2, num_class=N, graph_args=GRAPH_ARGS).
        early_fusion (bool): If True, concatenate stream features before the
                             classifier (early fusion). If False, sum logits
                             after the classifier (late fusion).
        **kwargs   : Keyword args forwarded to each ST_GCN stream
                     (e.g. edge_importance_weighting, adaptive_graph,
                     drop_graph_prob, dropout).

    Architecture (early fusion):
        joint_stream   (ST_GCN, extract_features=True) → 128-d
        motion_stream  (ST_GCN, extract_features=True) → 128-d
        bone_stream    (ST_GCN, extract_features=True) → 128-d
        bone_motion    (ST_GCN, extract_features=True) → 128-d
        concat → (N, 512) → Linear(512, 256) → LayerNorm → ReLU → Dropout(0.4)
               → Linear(256, num_class) → logits (N, num_class)

    Architecture (late fusion):
        joint_stream   (ST_GCN) → logits (N, num_class)
        motion_stream  (ST_GCN) → logits (N, num_class)
        bone_stream    (ST_GCN) → logits (N, num_class)
        bone_motion    (ST_GCN) → logits (N, num_class)
        sum → logits (N, num_class)
    """

    def __init__(self, *args, early_fusion=True, **kwargs):
        super().__init__()

        self.early_fusion = early_fusion

        if early_fusion:
            # ── Early Fusion Setup ────────────────────────────────────────────
            # Each ST_GCN stream runs with extract_features=True so it outputs
            # a 128-d feature vector (skips the classification head).
            # The reduced-width ST-GCN has a final channel count of 128.
            kwargs_feat = {**kwargs, 'extract_features': True}
            self.joint_stream       = ST_GCN(*args, **kwargs_feat)
            self.motion_stream      = ST_GCN(*args, **kwargs_feat)
            self.bone_stream        = ST_GCN(*args, **kwargs_feat)
            self.bone_motion_stream = ST_GCN(*args, **kwargs_feat)

            # The number of classes is the second positional argument
            num_class = args[1]

            # Early fusion classifier:
            # Input = 4 streams × 128 features = 512 features
            # LayerNorm is used instead of BatchNorm because it is safe
            # at batch size 1 (BatchNorm fails with a single sample).
            self.fusion_fc = nn.Sequential(
                nn.Linear(128 * 4, 256),          # compress 512 → 256
                nn.LayerNorm(256),                 # stable normalization
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),                   # regularize fused representation
                nn.Linear(256, num_class)          # final class logits
            )
        else:
            # ── Late Fusion Setup ─────────────────────────────────────────────
            # Each stream outputs logits directly (extract_features=False).
            # No fusion layer needed — outputs are summed in forward().
            self.joint_stream       = ST_GCN(*args, **kwargs)
            self.motion_stream      = ST_GCN(*args, **kwargs)
            self.bone_stream        = ST_GCN(*args, **kwargs)
            self.bone_motion_stream = ST_GCN(*args, **kwargs)

    def _compute_motion(self, x):
        """Compute frame-to-frame joint velocity as a fallback if not provided.

        Computes a centered difference approximation:
            motion[t] ≈ x[t] - 0.5 * x[t+1] - 0.5 * x[t-1]

        First and last frames are padded with zeros (boundary condition).

        Args:
            x (Tensor): Joint positions, shape (N, C, T, V).

        Returns:
            Tensor: Approximate motion, shape (N, C, T, V).
        """
        N, C, T, V = x.size()
        # Zero-filled boundary frames
        zeros = torch.zeros(N, C, 1, V, device=x.device, dtype=x.dtype)
        return torch.cat([
            zeros,                                            # t=0 (no prior frame)
            x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],  # centered diff
            zeros                                             # t=T-1 (no next frame)
        ], dim=2)

    def forward(self, x, motion=None, bone=None, bone_motion=None):
        """Run the four-stream forward pass.

        Args:
            x           (Tensor): Joint positions, shape (N, 2, T, V). Required.
            motion      (Tensor): Pre-computed joint motion (optional).
                                  If None, computed internally from x via centered diff.
            bone        (Tensor): Pre-computed bone vectors (optional).
                                  If None, zeros are used (no bone stream information).
            bone_motion (Tensor): Pre-computed bone motion (optional).
                                  If None, zeros are used.

        Returns:
            Tensor: Class logits, shape (N, num_class).
        """
        # Compute motion on-the-fly if the caller didn't pre-compute it
        if motion is None:
            motion = self._compute_motion(x)

        # Default to zero tensors if optional streams not provided
        if bone is None:
            bone = torch.zeros_like(x)
        if bone_motion is None:
            bone_motion = torch.zeros_like(x)

        if self.early_fusion:
            # ── Early Fusion Forward ──────────────────────────────────────────
            # Each stream produces a compact 128-d representation
            f_joint       = self.joint_stream(x)             # → (N, 128)
            f_motion      = self.motion_stream(motion)        # → (N, 128)
            f_bone        = self.bone_stream(bone)            # → (N, 128)
            f_bone_motion = self.bone_motion_stream(bone_motion)  # → (N, 128)

            # Concatenate all four feature vectors into a single 512-d vector
            # This allows the fusion classifier to model cross-stream correlations
            fused = torch.cat([f_joint, f_motion, f_bone, f_bone_motion], dim=1)
            # → (N, 512)

            # Shared classifier head (with LayerNorm for batch-size safety)
            return self.fusion_fc(fused)  # → (N, num_class)

        else:
            # ── Late Fusion Forward ───────────────────────────────────────────
            # Each stream produces full class logits independently
            # Summing logits = taking the product of softmax distributions
            return (
                self.joint_stream(x) +
                self.motion_stream(motion) +
                self.bone_stream(bone) +
                self.bone_motion_stream(bone_motion)
            )  # → (N, num_class)