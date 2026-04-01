"""
st_gcn.py
=========
Original ST-GCN (Yan et al. AAAI 2018) + improvements:
  1. Adaptive graph topology (Zhang et al. STA-GCN 2020)
  2. DropGraph regularization (Jiang et al. 2021)
  3. Early fusion support via feature projection layer

Input shape : (N, C, T, V)
Output shape: (N, num_class) OR (N, feat_dim) when extract_features=True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tgcn  import ConvTemporalGraphical
from models.graph import Graph, AdaptiveAdjacency, drop_graph


class Model(nn.Module):
    """ST-GCN with adaptive topology and DropGraph.

    Args:
        in_channels (int): input channels (2 for x,y)
        num_class (int): number of classes
        graph_args (dict): layout and strategy for Graph()
        edge_importance_weighting (bool): learnable per-layer edge weights
        adaptive_graph (bool): add learnable adjacency on top of fixed graph
        drop_graph_prob (float): DropGraph probability (0 = disabled)
        extract_features (bool): if True return 256-d features not logits
        **kwargs: dropout etc.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True,
                 adaptive_graph=True,
                 drop_graph_prob=0.1,
                 extract_features=False,
                 **kwargs):
        super().__init__()

        self.drop_graph_prob  = drop_graph_prob
        self.extract_features = extract_features

        # ── Build graph ───────────────────────────────────────────────────────
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size  = A.size(0)
        temporal_kernel_size = 9
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ── Adaptive graph ────────────────────────────────────────────────────
        if adaptive_graph:
            self.adaptive_adj = AdaptiveAdjacency(
                num_joints  = A.size(1),
                num_subsets = A.size(0),
                alpha       = 0.1
            )
        else:
            self.adaptive_adj = None

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # ── Reduced-width 10-layer architecture ───────────────────────────────
        # Channels halved (64→32, 128→64, 256→128) — cuts params ~4×.
        # With only ~300 training samples, smaller width generalizes better.
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 32,  kernel_size, 1, residual=False, **kwargs0),
            st_gcn(32,          32,  kernel_size, 1, **kwargs),
            st_gcn(32,          32,  kernel_size, 1, **kwargs),
            st_gcn(32,          32,  kernel_size, 1, **kwargs),
            st_gcn(32,          64,  kernel_size, 2, **kwargs),
            st_gcn(64,          64,  kernel_size, 1, **kwargs),
            st_gcn(64,          64,  kernel_size, 1, **kwargs),
            st_gcn(64,          128, kernel_size, 2, **kwargs),
            st_gcn(128,         128, kernel_size, 1, **kwargs),
            st_gcn(128,         128, kernel_size, 1, **kwargs),
        ])

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()

        # ── Data normalization ────────────────────────────────────────────────
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ── Apply DropGraph ───────────────────────────────────────────────────
        x = drop_graph(x, self.drop_graph_prob, self.training)

        # ── Get adjacency (fixed + adaptive) ─────────────────────────────────
        if self.adaptive_adj is not None:
            A = self.adaptive_adj(self.A)
        else:
            A = self.A

        # ── ST-GCN forward ────────────────────────────────────────────────────
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, A * importance)

        # ── Global average pool ───────────────────────────────────────────────
        x = F.avg_pool2d(x, x.size()[2:])   # (N, 128, 1, 1)

        if self.extract_features:
            return x.view(N, -1)             # (N, 128) for early fusion

        x = self.fcn(x)
        return x.view(N, -1)                 # (N, num_class)


class st_gcn(nn.Module):
    """One ST-GCN block."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding),
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
        res  = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res
        return self.relu(x), A