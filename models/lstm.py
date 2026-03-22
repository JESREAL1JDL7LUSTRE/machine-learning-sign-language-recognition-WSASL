import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal Convolutional Block ──────────────────────────────────────────────
class TCNBlock(nn.Module):
    """
    Single residual TCN block with dilated causal convolution.
    Dilation lets the model see progressively longer time windows.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.norm1   = nn.BatchNorm1d(out_channels)
        self.norm2   = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()

        # Residual connection — match channels if needed
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, time)
        out = self.relu(self.norm1(self.conv1(x)))
        out = out[:, :, :x.size(2)]   # causal trim
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = out[:, :, :x.size(2)]   # causal trim
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


# ── Temporal Attention ────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Learns which frames are most important for classification.
    Produces a weighted sum over the time dimension.
    """
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, time, channels)
        scores  = self.attn(x)              # (batch, time, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, time, 1)
        return (x * weights).sum(dim=1)    # (batch, channels)


# ── Main Model: TCN + Attention ───────────────────────────────────────────────
class SignLSTM(nn.Module):
    """
    TCN + Temporal Attention for Isolated Sign Language Recognition.

    Despite the class name (kept for compatibility), this is a
    Temporal Convolutional Network with an attention pooling head.

    Input  : (batch, T, input_size)
    Output : (batch, num_classes)
    """

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=20,
        dropout=0.3,
        bidirectional=True    # kept for API compatibility, not used in TCN
    ):
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── TCN layers (4 blocks, increasing dilation) ────────────────────
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=1),
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=2),
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=4),
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=8),
        ])

        # ── Temporal Attention ────────────────────────────────────────────
        self.attention = TemporalAttention(hidden_size)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, T, input_size)
        batch, T, _ = x.shape

        # Project input features
        x = self.input_proj(x)              # (batch, T, hidden)

        # TCN expects (batch, channels, time)
        x = x.permute(0, 2, 1)             # (batch, hidden, T)
        for block in self.tcn_blocks:
            x = block(x)

        # Back to (batch, time, channels) for attention
        x = x.permute(0, 2, 1)             # (batch, T, hidden)

        # Attention pooling → (batch, hidden)
        x = self.attention(x)

        # Classify
        return self.classifier(x)          # (batch, num_classes)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
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