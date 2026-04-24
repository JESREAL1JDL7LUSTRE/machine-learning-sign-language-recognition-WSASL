import torch
import torch.nn as nn

from models.st_gcn import Model as ST_GCN


class Model(nn.Module):

    def __init__(self, *args, early_fusion=True, **kwargs):
        super().__init__()

        self.early_fusion = early_fusion

        if early_fusion:
            # Each stream extracts 256-d features (no classifier head)
            kwargs_feat = {**kwargs, 'extract_features': True}
            self.joint_stream      = ST_GCN(*args, **kwargs_feat)
            self.motion_stream     = ST_GCN(*args, **kwargs_feat)
            self.bone_stream       = ST_GCN(*args, **kwargs_feat)
            self.bone_motion_stream= ST_GCN(*args, **kwargs_feat)

            # Early fusion classifier: 4 streams × 128 → num_class
            # (128-d per stream because st_gcn channels halved: 256→128)
            num_class = args[1]   # second positional arg
            self.fusion_fc = nn.Sequential(
                nn.Linear(128 * 4, 256),
                nn.LayerNorm(256),   # LayerNorm not BatchNorm — safe on batch size 1
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_class)
            )
        else:
            # Late fusion: each stream outputs logits, sum at end
            self.joint_stream      = ST_GCN(*args, **kwargs)
            self.motion_stream     = ST_GCN(*args, **kwargs)
            self.bone_stream       = ST_GCN(*args, **kwargs)
            self.bone_motion_stream= ST_GCN(*args, **kwargs)

    def _compute_motion(self, x):
        N, C, T, V = x.size()
        zeros = torch.zeros(N, C, 1, V, device=x.device, dtype=x.dtype)
        return torch.cat([
            zeros,
            x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
            zeros
        ], dim=2)

    def forward(self, x, bone=None, bone_motion=None):
        # Compute second-order motion internally (original paper style)
        motion = self._compute_motion(x)

        # Fallback zeros if streams not provided
        if bone is None:
            bone = torch.zeros_like(x)
        if bone_motion is None:
            bone_motion = torch.zeros_like(x)

        if self.early_fusion:
            # Each stream → 256-d feature vector
            f_joint       = self.joint_stream(x)
            f_motion      = self.motion_stream(motion)
            f_bone        = self.bone_stream(bone)
            f_bone_motion = self.bone_motion_stream(bone_motion)

            # Concatenate all features (N, 1024)
            fused = torch.cat([f_joint, f_motion, f_bone, f_bone_motion], dim=1)

            # Shared classifier (allows cross-stream interaction)
            return self.fusion_fc(fused)

        else:
            # Late fusion: sum of logits
            return (
                self.joint_stream(x) +
                self.motion_stream(motion) +
                self.bone_stream(bone) +
                self.bone_motion_stream(bone_motion)
            )