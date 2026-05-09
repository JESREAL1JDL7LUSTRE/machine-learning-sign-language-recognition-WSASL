"""
lstm.py — TCN + Temporal Attention Model for Sign Language Recognition
======================================================================

Despite the module/class being named 'SignLSTM' (kept for checkpoint
compatibility), this file implements a Temporal Convolutional Network (TCN)
with an attention-based temporal pooling head — NOT a recurrent model.

Why TCN instead of LSTM?
    TCNs have several advantages for fixed-length skeleton sequences:
      - Fully parallel over time (no sequential dependency).
      - Dilated convolutions exponentially expand the receptive field
        without extra parameters.
      - More stable gradients than long LSTM chains.

Model pipeline:
    Input (batch, T, features)
        → Linear projection to hidden_size
        → 4 dilated TCN blocks (dilation 1, 2, 4, 8)
        → Temporal attention pooling → (batch, hidden_size)
        → Two-layer MLP classifier
        → (batch, num_classes) logits

Components:
    TCNBlock         — A single residual dilated TCN block.
    TemporalAttention— Soft attention over the time dimension.
    SignLSTM         — Full TCN + attention model (main entry point).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal Convolutional Block ──────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Single residual TCN block with dilated causal convolution.

    Each block uses two 1-D convolutions (both with the same dilation).
    Progressive doubling of dilation across blocks allows the model to
    capture patterns at multiple time scales without increasing kernel size.

    Causal trimming: after each dilated conv, the output is sliced to match
    the original input length (removes future-padding that dilated conv adds).

    Residual connection: if in_channels ≠ out_channels, a 1×1 conv projects
    the skip path to the correct channel count before adding.

    Args:
        in_channels  (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size  (int): Convolution kernel size (applied to time axis).
        dilation     (int): Dilation factor. Effective receptive field per block:
                            (kernel_size - 1) * dilation + 1 frames.

    Input / output shape: (batch, channels, time)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # With causal padding = (k-1)*dilation, the output is the same length
        # as the input BEFORE we trim. We trim manually in forward() for safety.
        padding = (kernel_size - 1) * dilation

        # First dilated 1-D convolution (operates on time axis)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        # Second dilated 1-D convolution (same dilation, same receptive field)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )

        # Batch normalization after each convolution for training stability
        self.norm1   = nn.BatchNorm1d(out_channels)
        self.norm2   = nn.BatchNorm1d(out_channels)

        # Dropout for regularization between the two convolutions
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()

        # Residual connection — use a 1×1 conv if channel count changes,
        # otherwise a simple identity (no parameters)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        """Run one TCN block.

        Args:
            x (Tensor): Input, shape (batch, channels, time).

        Returns:
            Tensor: Output with residual added, shape (batch, out_channels, time).
        """
        # First convolution + norm + activation
        out = self.relu(self.norm1(self.conv1(x)))
        # Causal trim: remove extra frames added by the 'future' padding
        # This ensures the output is the same temporal length as the input
        out = out[:, :, :x.size(2)]

        out = self.dropout(out)

        # Second convolution + norm + activation
        out = self.relu(self.norm2(self.conv2(out)))
        out = out[:, :, :x.size(2)]   # causal trim again

        out = self.dropout(out)

        # Residual connection: add skip path to regularize gradient flow
        return self.relu(out + self.residual(x))


# ── Temporal Attention ────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Soft attention over the time dimension.

    Learns which frames carry the most discriminative information for
    classification and produces a weighted sum (soft selection) of all frames.

    Unlike mean/max pooling, attention can dynamically focus on the key motion
    phases of a sign (e.g. the core gesture, ignoring rest frames).

    Architecture:
        Linear(channels → 64) → Tanh → Linear(64 → 1) → Softmax(over time)
        → weighted sum over time → (batch, channels)

    Args:
        channels (int): Number of input feature channels (= hidden_size of TCN).
    """

    def __init__(self, channels):
        super().__init__()
        # Two-layer MLP that maps each frame's feature vector to a scalar score
        self.attn = nn.Sequential(
            nn.Linear(channels, 64),   # compress to smaller space
            nn.Tanh(),                  # squash to (-1, 1) for stability
            nn.Linear(64, 1)           # single unnormalized score per frame
        )

    def forward(self, x):
        """Compute attention-weighted temporal summary.

        Args:
            x (Tensor): Shape (batch, time, channels) — TCN output transposed.

        Returns:
            Tensor: Shape (batch, channels) — attention-pooled feature vector.
        """
        # Compute a scalar attention score for every frame
        scores  = self.attn(x)              # → (batch, time, 1)
        # Normalize scores to a proper probability distribution over time
        weights = torch.softmax(scores, dim=1)  # → (batch, time, 1)
        # Weighted sum: each frame's features multiplied by its attention weight
        return (x * weights).sum(dim=1)    # → (batch, channels)


# ── Main Model: TCN + Attention ───────────────────────────────────────────────

class SignLSTM(nn.Module):
    """TCN + Temporal Attention for Isolated Sign Language Recognition.

    Despite the class name (kept for backward checkpoint compatibility),
    this model is a Temporal Convolutional Network (TCN) with attention
    pooling — NOT a recurrent model.

    Architecture:
        1. Input projection: (T, input_size) → (T, hidden_size) via Linear + ReLU.
        2. Permute for Conv1d: (batch, T, H) → (batch, H, T).
        3. Four dilated TCN blocks with doubling dilation (1, 2, 4, 8):
              - dilation=1: receptive field = 3 frames
              - dilation=2: receptive field = 5 frames
              - dilation=4: receptive field = 9 frames
              - dilation=8: receptive field = 17 frames
        4. Permute back: (batch, H, T) → (batch, T, H).
        5. Temporal attention pooling: (batch, T, H) → (batch, H).
        6. Classifier MLP: (batch, H) → (batch, H//2) → (batch, num_classes).

    Args:
        input_size   (int)  : Number of input features per frame.
                              150 for MediaPipe (33 pose + 21+21 hand joints × 2).
        hidden_size  (int)  : Channel width of all TCN blocks. Default: 128.
        num_layers   (int)  : Kept for API compatibility (not used; fixed to 4 blocks).
        num_classes  (int)  : Number of sign categories to classify.
        dropout      (float): Dropout rate in the classifier MLP. Default: 0.3.
        bidirectional(bool) : Kept for API compatibility (not used in TCN).

    Input  : (batch, T, input_size) — sequence of skeleton feature frames
    Output : (batch, num_classes)   — class logits (before softmax)
    """

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=20,
        dropout=0.3,
        bidirectional=True    # kept for API compatibility; TCN is not bidirectional
    ):
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────────
        # Projects raw skeleton features (e.g. 150-d) to the TCN working width.
        # Applied independently to each time frame (Linear works on last dim).
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── TCN blocks (4 blocks, exponentially increasing dilation) ──────────
        # The four dilations give a combined receptive field of ~17 frames
        # at the last block, letting the model capture sign timing patterns
        # across roughly a quarter of a 64-frame sequence.
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=1),  # 3-frame RF
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=2),  # 5-frame RF
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=4),  # 9-frame RF
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=8),  # 17-frame RF
        ])

        # ── Temporal Attention pooling ────────────────────────────────────────
        # Learns which frames to emphasize (e.g., peak motion of the sign)
        self.attention = TemporalAttention(hidden_size)

        # ── Two-layer MLP classifier head ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),   # compress features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)     # final class logits
        )

    def forward(self, x):
        """Run the full TCN + Attention forward pass.

        Args:
            x (Tensor): Skeleton sequence, shape (batch, T, input_size).

        Returns:
            Tensor: Class logits, shape (batch, num_classes).
        """
        batch, T, _ = x.shape

        # Step 1: Project each frame from input_size → hidden_size
        x = self.input_proj(x)              # → (batch, T, hidden)

        # Step 2: Transpose for Conv1d (expects channels before time)
        x = x.permute(0, 2, 1)             # → (batch, hidden, T)

        # Step 3: Pass through all 4 dilated TCN blocks
        # Each block keeps the same (batch, hidden, T) shape
        for block in self.tcn_blocks:
            x = block(x)

        # Step 4: Transpose back to (batch, time, channels) for attention
        x = x.permute(0, 2, 1)             # → (batch, T, hidden)

        # Step 5: Attention pooling collapses the time dimension
        x = self.attention(x)              # → (batch, hidden)

        # Step 6: Classify to num_classes logits
        return self.classifier(x)          # → (batch, num_classes)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify that the model runs without error on dummy data
    batch_size  = 4
    time_steps  = 64
    features    = 150
    num_classes = 20

    model = SignLSTM(
        input_size  = features,
        hidden_size = 128,
        num_classes = num_classes,
        dropout     = 0.3
    )

    dummy = torch.randn(batch_size, time_steps, features)
    out   = model(dummy)

    print("Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total_params:,}")
    print(f"Input  shape     : {dummy.shape}")
    print(f"Output shape     : {out.shape}")
    print("✅ Forward pass OK")