"""
tgcn.py — ConvTemporalGraphical: The Basic Graph Convolution Unit
=================================================================

This is the fundamental building block used inside every ST-GCN layer.
It performs a *spatial graph convolution* across the joint dimension,
using a 2D convolution that simultaneously handles:
  - The temporal dimension (with a configurable kernel size along time)
  - The spatial graph kernel size K (number of adjacency subsets)

The module is ported from the original ST-GCN repository by Yan et al.:
  https://github.com/yysijie/st-gcn

Changes from the original:
  - Removed deprecated `torch.autograd.Variable` wrapper.
  - Updated einsum notation for modern PyTorch compatibility.

Input  : (N, C_in, T_in, V) — batch, channels, frames, joints
         (K, V, V)           — adjacency matrix (K spatial subsets)
Output : (N, C_out, T_out, V) — same spatial dims, projected channels
         (K, V, V)             — adjacency passthrough (unchanged)
"""

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    r"""Apply a graph convolution that operates jointly over space and time.

    This is the core GCN operation used in every ST-GCN block. The convolution:
      1. Expands C_in channels to (C_out * K) channels via a standard 2-D conv
         along the time axis (spatial kernel size = 1 in the joint dimension).
      2. Reshapes the output to expose the K graph-subset axis.
      3. Multiplies by the K × V × V adjacency matrix via einsum to aggregate
         messages from neighbouring joints under each of the K subsets.

    Mathematical operation (one subset k):
        x_out[n, c, t, v] = Σ_w A[k, v, w] * x_expanded[n, k, c, t, w]

    Args:
        in_channels  (int): Number of input feature channels (C_in).
        out_channels (int): Number of output feature channels (C_out).
        kernel_size  (int): Spatial graph kernel size K (= number of adjacency
                            subsets, determined by the Graph strategy).
        t_kernel_size (int): Temporal convolution kernel size along the time axis.
                             Default: 1 (single-frame graph conv, no temporal mixing).
        t_stride     (int): Stride for the temporal convolution. Default: 1.
        t_padding    (int): Zero-padding along the time axis. Default: 0.
        t_dilation   (int): Dilation for the temporal convolution. Default: 1.
        bias         (bool): Whether to add a learnable bias. Default: True.

    Shape:
        Input[0] : (N, C_in,  T_in,  V) — graph sequence
        Input[1] : (K, V, V)            — adjacency matrix (K subsets)
        Output[0]: (N, C_out, T_out, V) — transformed graph sequence
        Output[1]: (K, V, V)            — adjacency passthrough

        where:
            N     = batch size
            K     = spatial kernel size (= number of adjacency subsets)
            C_in  = input channels
            C_out = output channels
            T_in / T_out = input / output sequence length (depends on stride)
            V     = number of joints / nodes
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,       # spatial K — number of adjacency subsets
                 t_kernel_size=1,   # temporal kernel size (usually 1 here; TCN handles temporal separately)
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        # Remember K so we can reshape during forward
        self.kernel_size = kernel_size

        # A single Conv2d that produces C_out * K output channels.
        # The joint axis (V) is treated as width; the time axis (T) is height.
        # Kernel height = t_kernel_size (temporal); Kernel width = 1 (no spatial mixing yet).
        # The spatial mixing happens later via the einsum with the adjacency matrix.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,   # K copies of C_out features
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        """Apply graph convolution.

        Args:
            x (Tensor): Input feature map, shape (N, C_in, T_in, V).
            A (Tensor): Adjacency matrix, shape (K, V, V).

        Returns:
            Tuple[Tensor, Tensor]:
                - Output feature map, shape (N, C_out, T_out, V).
                - The same adjacency A (passthrough for chaining).
        """
        # Sanity check: the adjacency must match the kernel size K
        assert A.size(0) == self.kernel_size

        # Step 1: Apply temporal + channel projection convolution
        # x shape: (N, C_in, T, V) → (N, C_out * K, T_out, V)
        x = self.conv(x)

        # Step 2: Reshape to expose the K axis
        # (N, C_out * K, T_out, V) → (N, K, C_out, T_out, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)

        # Step 3: Multiply by adjacency to aggregate neighbouring joint features
        # 'nkctv,kvw->nctw': for each output joint w, sum over source joints v
        # weighted by the adjacency A[k, v, w], summed over K subsets k.
        # Result shape: (N, C_out, T_out, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)

        # Return contiguous tensor and pass adjacency through (for compatibility)
        return x.contiguous(), A