"""
st_gcn.py
=========
Original ST-GCN architecture from:
  Yan et al. "Spatial Temporal Graph Convolutional Networks for
  Skeleton-Based Action Recognition", AAAI 2018.
  https://github.com/yysijie/st-gcn

Ported to modern PyTorch (2.x):
  - Removed torch.autograd.Variable (deprecated since PyTorch 0.4)
  - Input shape changed from (N,C,T,V,M) to (N,C,T,V)
    (M=1 person, squeezed for simplicity)
  - Added 'mediapipe_51' graph layout
  - edge_importance_weighting kept as original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tgcn  import ConvTemporalGraphical
from models.graph import Graph


class Model(nn.Module):
    """Spatial Temporal Graph Convolutional Network.

    Args:
        in_channels (int): Number of input channels (2 for x,y)
        num_class (int): Number of action/sign classes
        graph_args (dict): Arguments for Graph() — layout and strategy
        edge_importance_weighting (bool): Learn per-edge importance weights
        **kwargs: dropout, etc.

    Input shape : (N, in_channels, T, V)
    Output shape: (N, num_class)
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_args,
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        # ── Build graph ───────────────────────────────────────────────────────
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ── Network config ────────────────────────────────────────────────────
        spatial_kernel_size  = A.size(0)
        temporal_kernel_size = 9
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # Original 10-layer ST-GCN architecture (Yan et al. 2018)
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 64,  kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64,          64,  kernel_size, 1, **kwargs),
            st_gcn(64,          64,  kernel_size, 1, **kwargs),
            st_gcn(64,          64,  kernel_size, 1, **kwargs),
            st_gcn(64,          128, kernel_size, 2, **kwargs),
            st_gcn(128,         128, kernel_size, 1, **kwargs),
            st_gcn(128,         128, kernel_size, 1, **kwargs),
            st_gcn(128,         256, kernel_size, 2, **kwargs),
            st_gcn(256,         256, kernel_size, 1, **kwargs),
            st_gcn(256,         256, kernel_size, 1, **kwargs),
        ])

        # ── Edge importance weighting (learnable) ────────────────────────────
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # ── Classifier ───────────────────────────────────────────────────────
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous()   # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()   # (N, C, T, V)

        # ST-GCN forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[2:])         # (N, 256, 1, 1)
        x = x.view(N, -1, 1, 1)

        # Prediction
        x = self.fcn(x)
        x = x.view(N, -1)

        return x


class st_gcn(nn.Module):
    """One ST-GCN block: graph conv + temporal conv + residual.

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (tuple): (temporal_ks, spatial_ks)
        stride (int): temporal stride
        dropout (float): dropout rate
        residual (bool): use residual connection
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A