"""
data.py
Handles loading, preprocessing, and DataLoader creation for FordA and FordB datasets.
Expects .txt files where each row: <label> <500 space-separated measurements>
Labels are originally -1 and 1; mapped to 0 and 1 internally for PyTorch training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class FordDataset(Dataset):
    """PyTorch Dataset for Ford time-series data."""
    def __init__(self, features, labels):
        # features: (N, 500), labels: (N,)
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (N, 1, 500)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_txt(path):
    """
    Load a Ford dataset from a .txt file.
    Format per line: label val1 val2 ... val500
    Returns: features (N, 500), labels (N,)
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Convert to float; first is label, rest are 500 measurements
            vals = list(map(float, parts))
            label = int(float(vals[0]))
            features = vals[1:]
            data.append((label, features))

    labels = np.array([d[0] for d in data])
    features = np.array([d[1] for d in data])

    # Map labels from {-1, 1} -> {0, 1}
    labels = np.where(labels == -1, 0, 1)
    return features, labels


def normalize_features(train_features, *other_features):
    """
    Z-score normalization based on training set statistics.
    Computes mean/std across all training samples and time steps.
    """
    mean = np.mean(train_features)
    std = np.std(train_features)
    # Avoid division by zero
    if std == 0:
        std = 1.0

    normalized = [(f - mean) / std for f in [train_features] + list(other_features)]
    return normalized


def get_dataloaders(train_path, test_a_path, test_b_path, val_ratio=0.2, batch_size=64, seed=42):
    """
    Loads FordA train/val/test and FordB test, applies normalization, returns DataLoaders.
    """
    # Load raw data
    X_train_full, y_train_full = load_txt(train_path)
    X_test_a, y_test_a = load_txt(test_a_path)
    X_test_b, y_test_b = load_txt(test_b_path)

    # Normalize using training statistics
    X_train_full, X_test_a, X_test_b = normalize_features(X_train_full, X_test_a, X_test_b)

    # Create train/val split from FordA_TRAIN
    total_train = len(y_train_full)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    full_dataset = FordDataset(X_train_full, y_train_full)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    test_a_dataset = FordDataset(X_test_a, y_test_a)
    test_b_dataset = FordDataset(X_test_b, y_test_b)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_a_loader = DataLoader(test_a_dataset, batch_size=batch_size, shuffle=False)
    test_b_loader = DataLoader(test_b_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_a_loader, test_b_loader