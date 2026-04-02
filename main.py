import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FordADataset(Dataset):
    def __init__(self, file_path):
        # Učitavanje podataka
        data = np.loadtxt(file_path)
        # Prva kolona je labela, ostalo su podaci
        self.x = torch.tensor(data[:, 1:], dtype=torch.float32)
        y = data[:, 0]

        # Mapiranje labela: -1 postaje 0, 1 ostaje 1 (za CrossEntropyLoss)
        y = np.where(y == -1, 0, 1)
        self.y = torch.tensor(y, dtype=torch.long)

        # Dodajemo dimenziju kanala jer CNN to očekuje (Batch, Channel, Length)
        # Pošto je univariate, imamo samo 1 kanal
        self.x = self.x.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Inicijalizacija
train_dataset = FordADataset('FordA/FordA_TRAIN.txt')
test_dataset = FordADataset('FordA/FordA_TEST.txt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv1d(1, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool1d(2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv1d(64, 128, kernel_size=3),
      nn.ReLU(),
      nn.AdaptiveAvgPool1d(1)  # Sažima sve na jednu vrijednost po kanalu
    )
    self.fc = nn.Linear(128, 2)  # 2 izlaza za klasifikaciju (0 i 1)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0), -1)  # "Spljošti" podatke za FC sloj
    out = self.fc(out)
    return out


model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, loader, epochs=10):
  model.train()
  for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in loader:
      optimizer.zero_grad()  # Resetiraj gradijente
      outputs = model(inputs)  # Forward pass
      loss = criterion(outputs, labels)  # Izračunaj grešku
      loss.backward()  # Backward pass (učenje)
      optimizer.step()  # Ažuriraj težine
      running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader):.4f}")


train(model, train_loader)


def test(model, loader):
  model.eval()  # Prebacivanje u mod za evaluaciju
  correct = 0
  total = 0
  with torch.no_grad():  # Isključujemo računanje gradijenata zbog brzine
    for inputs, labels in loader:

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f"Preciznost na testnom skupu: {100 * correct / total:.2f}%")


test(model, test_loader)