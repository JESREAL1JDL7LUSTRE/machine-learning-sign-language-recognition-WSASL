import torch
import torch.nn as nn


class SignLSTM(nn.Module):
    """
    2-layer bidirectional LSTM for Isolated Sign Language Recognition.

    Input  : (batch, T, 150)   — T frames, 150 features per frame
    Output : (batch, num_classes)

    Architecture:
        BiLSTM (2 layers)  →  Dropout  →  FC
    """

    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=10,
        dropout=0.3,
        bidirectional=True
    ):
        super(SignLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers    = num_layers
        self.hidden_size   = hidden_size

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= bidirectional
        )

        # Output size doubles when bidirectional
        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x):
        # x: (batch, T, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * directions, batch, hidden_size)

        if self.bidirectional:
            # Concat last forward and last backward hidden state
            forward  = h_n[-2]   # last layer, forward
            backward = h_n[-1]   # last layer, backward
            h = torch.cat([forward, backward], dim=1)
        else:
            h = h_n[-1]

        out = self.dropout(h)
        return self.fc(out)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size  = 4
    time_steps  = 64
    features    = 150
    num_classes = 10

    model = SignLSTM(
        input_size  = features,
        hidden_size = 128,
        num_layers  = 2,
        num_classes = num_classes,
        dropout     = 0.3,
        bidirectional = True
    )

    dummy_input = torch.randn(batch_size, time_steps, features)
    output      = model(dummy_input)

    print("Model Architecture:")
    print(model)
    print(f"\nInput  shape : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")  # should be (4, 10)
    print("✅ Forward pass OK")