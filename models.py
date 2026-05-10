"""
models.py
Defines two neural network architectures for time-series classification:
1. CNN1D  – 1D Convolutional Neural Network
2. LSTM   – Long Short-Term Memory network
Input shape expected: (batch, 1, 500)
Output: logits for 2 classes (binary classification)
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D-CNN for time-series classification.
    Uses multiple conv blocks with BatchNorm, ReLU, and MaxPool,
    followed by adaptive pooling and fully-connected layers.
    """
    def __init__(self, num_classes=2):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 500 -> 250

            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 250 -> 125

            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 125 -> 62

            # Block 4
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # (batch, 256)
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for time-series.
    Uses 2 stacked LSTM layers, takes the last hidden state,
    and passes it through fully-connected layers.
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 500) -> reshape to (batch, 500, 1) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM output: (batch, seq_len, hidden_size), hidden states
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state of the last layer
        # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        out = self.classifier(last_hidden)
        return out