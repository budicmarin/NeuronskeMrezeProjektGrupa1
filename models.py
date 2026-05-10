import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))
    def forward(self, x): return self.c(self.f(x))

class GRU(nn.Module):
    def __init__(self, nc=2, hid=128, layers=1, drop=0.3):
        super().__init__()
        self.g = nn.GRU(1, hid, layers, batch_first=True)
        self.c = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, nc))
    def forward(self, x):
        o, _ = self.g(x.permute(0, 2, 1))
        return self.c(o[:, -1])