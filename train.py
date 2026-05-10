"""
train.py
Training and evaluation pipeline.
- Trains CNN1D and LSTM on FordA_TRAIN (with validation split)
- Evaluates on FordA_TEST and FordB_TEST
- Saves best models and prints metrics
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from data import get_dataloaders
from models import CNN1D, LSTMClassifier


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
VAL_RATIO = 0.2
PATIENCE = 10  # early stopping


# Get the directory where train.py lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(SCRIPT_DIR, ".", "FordA", "FordA_TRAIN.txt")
TEST_A_PATH = os.path.join(SCRIPT_DIR, ".", "FordA", "FordA_TEST.txt")
TEST_B_PATH = os.path.join(SCRIPT_DIR, ".", "FordB", "FordB_TEST.txt")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }


def train_model(model, train_loader, val_loader, model_name, device, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"\n=== Training {model_name} ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, f"checkpoints/{model_name}_best.pt")
    print(f"Saved best model to checkpoints/{model_name}_best.pt")

    return model, history


def test_model(model, test_loader, model_name, dataset_name, device):
    criterion = nn.CrossEntropyLoss()
    loss, acc, labels, preds = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(labels, preds)
    metrics["loss"] = loss

    print(f"\n--- {model_name} on {dataset_name} ---")
    print(f"Loss:      {loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Confusion Matrix:{metrics['confusion_matrix']}")

    return metrics
    

def main():
    # 1. Load data
    train_loader, val_loader, test_a_loader, test_b_loader = get_dataloaders(
        TRAIN_PATH, TEST_A_PATH, TEST_B_PATH,
        val_ratio=VAL_RATIO, batch_size=BATCH_SIZE
    )

    # 2. Initialize models
    cnn = CNN1D(num_classes=2).to(DEVICE)
    lstm = LSTMClassifier(input_size=1, hidden_size=128, num_layers=2, num_classes=2).to(DEVICE)

    # 3. Train models
    cnn, cnn_hist = train_model(cnn, train_loader, val_loader, "CNN1D", DEVICE)
    lstm, lstm_hist = train_model(lstm, train_loader, val_loader, "LSTM", DEVICE)

    # 4. Evaluate on FordA_TEST
    cnn_a = test_model(cnn, test_a_loader, "CNN1D", "FordA_TEST", DEVICE)
    lstm_a = test_model(lstm, test_a_loader, "LSTM", "FordA_TEST", DEVICE)

    # 5. Evaluate on FordB_TEST (robustness check)
    cnn_b = test_model(cnn, test_b_loader, "CNN1D", "FordB_TEST", DEVICE)
    lstm_b = test_model(lstm, test_b_loader, "LSTM", "FordB_TEST", DEVICE)

    # 6. Save histories and metrics
    os.makedirs("results", exist_ok=True)
    with open("results/histories.json", "w") as f:
        json.dump({"CNN1D": cnn_hist, "LSTM": lstm_hist}, f, indent=2)

    with open("results/metrics.json", "w") as f:
        json.dump({
            "CNN1D": {"FordA_TEST": cnn_a, "FordB_TEST": cnn_b},
            "LSTM": {"FordA_TEST": lstm_a, "FordB_TEST": lstm_b}
        }, f, indent=2)

    print("\nAll done. Results saved to results/")


if __name__ == "__main__":
    main()