"""
bilstm_mediapipe.py
===================
True Bidirectional LSTM for sign language recognition.
(Generation 1 baseline – uses flat 102-feature MediaPipe keypoints)

Input  : (batch, T, input_size)   e.g. (N, 64, 102)
Output : (batch, num_classes)
"""

import torch
import torch.nn as nn


class BiLSTMSign(nn.Module):
    """
    Bidirectional LSTM with attention pooling.

    Args:
        input_size   : feature dimension per frame (default 102 after filtering)
        hidden_size  : LSTM hidden units per direction
        num_layers   : stacked LSTM depth
        num_classes  : number of sign classes
        dropout      : applied between layers and in classifier head
        bidirectional: True → 2× hidden in output
    """

    def __init__(
        self,
        input_size   = 102,
        hidden_size  = 128,
        num_layers   = 2,
        num_classes  = 20,
        dropout      = 0.3,
        bidirectional= True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        d = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention over LSTM outputs
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * d, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * d, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, T, features)
        out, _ = self.lstm(x)          # (batch, T, hidden*d)

        # Attention pooling
        scores  = self.attn(out)       # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)
        ctx     = (out * weights).sum(dim=1)  # (batch, hidden*d)

        return self.classifier(ctx)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BiLSTMSign(input_size=102, hidden_size=128, num_classes=20)
    x = torch.randn(4, 64, 102)
    print(f"Input : {x.shape}")
    print(f"Output: {model(x).shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    print("✅ BiLSTM OK")