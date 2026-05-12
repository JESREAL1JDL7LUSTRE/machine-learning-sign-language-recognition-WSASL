"""
st_gcn.py — ST-GCN Backbone with Adaptive Topology and DropGraph
=================================================================

This is the single-stream Spatial-Temporal Graph Convolutional Network
based on Yan et al. "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition" (AAAI 2018).

Enhancements over the original paper:
    1. Adaptive graph topology (Zhang et al. STA-GCN 2020) — a learnable
       correction is added to the fixed skeleton adjacency matrix.
    2. DropGraph regularization (Jiang et al. 2021) — randomly drops entire
       joint channels during training to improve generalization.
    3. Reduced-width architecture — channels are halved (64→32, 128→64,
       256→128) to improve generalization on small datasets (~300 samples).

Input  shape : (N, C, T, V)  — batch, channels (2 for x,y), frames, joints
Output shape : (N, num_class) — class logits
               OR (N, 128)   — feature vector when extract_features=True

Components:
    Model    — The full ST-GCN model (BatchNorm → ST-GCN layers → pool → classify)
    st_gcn   — One ST-GCN block (graph conv + temporal conv + residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tgcn  import ConvTemporalGraphical
from models.graph import Graph, AdaptiveAdjacency, drop_graph


class Model(nn.Module):
    """Full ST-GCN model with adaptive graph and DropGraph.

    Architecture overview:
        1. Input batch normalization (normalizes per-joint channels across time).
        2. Optional DropGraph (randomly zeros joints to regularize).
        3. Optional adaptive adjacency (learns graph topology corrections).
        4. 10 ST-GCN blocks with growing channel width and stride-2 downsampling.
        5. Global average pooling over time and joints → (N, 128) feature vector.
        6. 1×1 conv classification head → (N, num_class) logits.

    Args:
        in_channels              (int)  : Input channels. 2 for (x, y) coordinates.
        num_class                (int)  : Number of output classes.
        graph_args               (dict) : Keyword args for Graph() (layout, strategy, etc.).
        edge_importance_weighting(bool) : Learn a per-layer, per-edge importance scalar.
                                         Multiplies A element-wise at each ST-GCN block.
        adaptive_graph           (bool) : Add AdaptiveAdjacency correction to A.
        drop_graph_prob          (float): Probability for DropGraph during training.
                                         Set to 0.0 to disable.
        extract_features         (bool) : If True, return the 128-d pooled feature
                                         vector instead of class logits. Used by the
                                         four-stream early fusion model to combine
                                         features before the shared classifier.
        **kwargs                        : Extra kwargs forwarded to each st_gcn block
                                         (e.g. dropout=0.4).
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True,
                 adaptive_graph=True,
                 drop_graph_prob=0.1,
                 extract_features=False,
                 **kwargs):
        super().__init__()

        # Store flags needed at forward time
        self.drop_graph_prob  = drop_graph_prob
        self.extract_features = extract_features

        # ── Build graph and register adjacency as a non-trainable buffer ──────
        # Graph() computes the skeleton edge list and adjacency matrix A
        self.graph = Graph(**graph_args)
        # Register A as a buffer so it moves to the correct device with .to(device)
        # but is NOT included in model.parameters() (no gradient)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Derive kernel sizes from the adjacency:
        #   spatial_kernel_size  = K (number of subsets from the spatial strategy)
        #   temporal_kernel_size = 9 (standard value from the original ST-GCN paper)
        spatial_kernel_size  = A.size(0)
        temporal_kernel_size = 9
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)

        # Input batch normalization — normalizes (in_channels * V) features
        # across the time dimension for each sample
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ── Adaptive graph correction ─────────────────────────────────────────
        if adaptive_graph:
            # AdaptiveAdjacency learns a small correction ΔA on top of the
            # fixed skeleton topology; initialized to near-zero so training
            # starts from the physics-based graph
            self.adaptive_adj = AdaptiveAdjacency(
                num_joints  = A.size(1),
                num_subsets = A.size(0),
                alpha       = 0.1
            )
        else:
            self.adaptive_adj = None

        # Remove 'dropout' from kwargs0 to avoid passing it to the first block
        # (the first block has residual=False and shouldn't use dropout)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # ── 10-layer ST-GCN architecture (reduced width) ─────────────────────
        # Original ST-GCN uses 64/128/256 channels; this uses 32/64/128
        # to reduce parameters ~4× for better generalization on small datasets.
        # stride=2 at layers 5 and 8 halves the temporal resolution each time.
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 32,  kernel_size, 1, residual=False, **kwargs0),  # Layer 1 — no residual (channel mismatch)
            st_gcn(32,          32,  kernel_size, 1, **kwargs),                   # Layer 2
            st_gcn(32,          32,  kernel_size, 1, **kwargs),                   # Layer 3
            st_gcn(32,          32,  kernel_size, 1, **kwargs),                   # Layer 4
            st_gcn(32,          64,  kernel_size, 2, **kwargs),                   # Layer 5 — downsample ×2
            st_gcn(64,          64,  kernel_size, 1, **kwargs),                   # Layer 6
            st_gcn(64,          64,  kernel_size, 1, **kwargs),                   # Layer 7
            st_gcn(64,          128, kernel_size, 2, **kwargs),                   # Layer 8 — downsample ×2
            st_gcn(128,         128, kernel_size, 1, **kwargs),                   # Layer 9
            st_gcn(128,         128, kernel_size, 1, **kwargs),                   # Layer 10
        ])

        # ── Edge importance weighting ─────────────────────────────────────────
        # Each ST-GCN layer gets its own learnable K×V×V scalar mask applied to A.
        # This lets the model learn which joint connections matter most at each layer.
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))   # initialized to all-ones (identity)
                for _ in self.st_gcn_networks
            ])
        else:
            # Scalar 1 effectively multiplies A by 1 (no effect)
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # ── Classification head ───────────────────────────────────────────────
        # 1×1 convolution maps the 128-d feature map to num_class logits
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):
        """Forward pass through the full ST-GCN.

        Args:
            x (Tensor): Input skeleton sequence, shape (N, C, T, V).

        Returns:
            Tensor:
                - (N, num_class) class logits when extract_features=False.
                - (N, 128)       feature vector when extract_features=True.
        """
        N, C, T, V = x.size()

        # ── Step 1: Input batch normalization ─────────────────────────────────
        # Reshape to (N, V*C, T) so BatchNorm1d normalizes each (joint, channel)
        # feature independently across the time dimension
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        # Reshape back to (N, C, T, V) for the ST-GCN layers
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ── Step 2: DropGraph regularization ─────────────────────────────────
        # Randomly zeroes entire joint dimensions to prevent over-reliance
        # on specific keypoints (only active during training)
        x = drop_graph(x, self.drop_graph_prob, self.training)

        # ── Step 3: Get effective adjacency (fixed ± adaptive) ─────────────────
        if self.adaptive_adj is not None:
            # Combine fixed physics-based graph with learned correction
            A = self.adaptive_adj(self.A)
        else:
            # Use only the fixed skeleton graph
            A = self.A

        # ── Step 4: Pass through 10 ST-GCN layers ────────────────────────────
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            # Scale adjacency element-wise by the learned importance weights
            # This allows each layer to focus on different joint pairs
            x, _ = gcn(x, A * importance)

        # ── Step 5: Global average pooling over time and joints ───────────────
        # Collapses (T, V) → (1, 1), giving a single 128-d descriptor per sample
        x = F.avg_pool2d(x, x.size()[2:])   # → (N, 128, 1, 1)

        # Return features early if used as a stream in a multi-stream model
        if self.extract_features:
            return x.view(N, -1)             # → (N, 128)

        # ── Step 6: Classification ────────────────────────────────────────────
        x = self.fcn(x)                      # → (N, num_class, 1, 1)
        return x.view(N, -1)                 # → (N, num_class)


class st_gcn(nn.Module):
    """One ST-GCN block: spatial graph convolution + temporal convolution + residual.

    Data flow inside one block:
        Input x → GCN → BN → ReLU → Temporal Conv → BN → Dropout
        Residual(x) → + → ReLU → Output

    The residual connection handles the case where channel counts or time
    resolution changes (stride > 1 or in_channels ≠ out_channels).

    Args:
        in_channels  (int)  : Number of input feature channels.
        out_channels (int)  : Number of output feature channels.
        kernel_size  (tuple): (temporal_kernel, spatial_kernel) e.g. (9, K).
        stride       (int)  : Temporal stride. Use 2 to halve sequence length.
        dropout      (float): Dropout probability (applied after temporal conv).
        residual     (bool) : Whether to add a skip/residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()

        # Validate kernel dimensions
        assert len(kernel_size) == 2         # must be (temporal_k, spatial_k)
        assert kernel_size[0] % 2 == 1       # temporal kernel must be odd for symmetric padding

        # Compute 'same' temporal padding (zero-pads so output length = input length / stride)
        padding = ((kernel_size[0] - 1) // 2, 0)

        # Spatial GCN: ConvTemporalGraphical performs the graph multiplication
        # kernel_size[1] = number of adjacency subsets K
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # Temporal convolution: models motion patterns over adjacent frames
        # kernel height = kernel_size[0] (temporal), kernel width = 1 (per-joint)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),          # normalize after graph conv
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1),          # temporal kernel
                      (stride, 1),                  # stride along time axis
                      padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),     # regularize
        )

        # ── Residual / skip connection ────────────────────────────────────────
        if not residual:
            # First layer: no residual (channels change from C to 32)
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            # Channels and length unchanged → identity shortcut
            self.residual = lambda x: x
        else:
            # Channels or length changed → learnable projection shortcut
            # Projects input to match output shape before adding
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """Run one ST-GCN block.

        Args:
            x (Tensor): Input feature map, shape (N, C_in, T, V).
            A (Tensor): Effective adjacency matrix, shape (K, V, V).
                        (May include adaptive + importance scaling.)

        Returns:
            Tuple[Tensor, Tensor]:
                - Output feature map, shape (N, C_out, T_out, V).
                - Adjacency A (passthrough for compatibility with the loop).
        """
        # Save input for the residual connection before modifying x
        res  = self.residual(x)
        # Graph convolution: aggregate spatial information from neighbours
        x, A = self.gcn(x, A)
        # Temporal conv + residual: model motion over time
        x    = self.tcn(x) + res
        return self.relu(x), A