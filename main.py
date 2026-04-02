import torch
import torch.nn as nn
import torch.optim as optim
from PIL.ImageFont import core
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.xpu import device

#Device se postavlja na cpu zbog zastarijele grafičke katice
device = torch.device("cpu")
# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01

#Klasa DatasetLoader služi za očitavanje podataka iz .txt dataoteka
class DatasetLoader(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.x = torch.tensor(data[:, 1:], dtype=torch.float32)
        y = data[:, 0]
        y = np.where(y == -1, 0, 1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.x = self.x.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Inicijalizacija
train_dataset = DatasetLoader('FordA/FordA_TRAIN.txt')
test_dataset = DatasetLoader('FordA/FordA_TEST.txt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Evaluacijska funkcija
def eval_function(model,test_loader):
    model.eval()
    correct=0
    with torch.no_grad():
        for X,y in test_loader:
            #X,y se postavljaju na zadani device može biti cpu ili cuda na grafičkim karticama
            X,y=X.to(device),y.to(device)
            output=model(X)
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(y.view_as(pred)).sum().item()
        accuracy=100*correct/len(test_loader.dataset)
        return accuracy

# Train function
def train(model, train_loader, test_loader, epochs=1):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

  model.train()

  for epoch in range(epochs):
    for X, y in train_loader:
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()
      output = model(X)
      loss = criterion(output, y)
      loss.backward()
      optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {eval_function(model, test_loader):.2f}%')



def print_model(model):
    print(f'Model device: {device}')
    print(f'Model architecture:')
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f'Total parameters: {num_params}')
    print(f'Model size: {size_mb:.2f} MB')


def final_test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\n--- Klasifikacijski izvještaj ---")
    print(classification_report(all_labels, all_preds, target_names=['Klasa -1', 'Klasa 1']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-1', '1'], yticklabels=['-1', '1'])
    plt.xlabel('Predviđeno')
    plt.ylabel('Stvarno')
    plt.title('Matrica zabune')
    plt.savefig('confusion_matrix.png')
    print("\nGrafikon matrice zabune je spremljen kao 'confusion_matrix.png'")
    plt.close()

#FCCN klasa koristi konvolucijsku mrežu za treniranje modela
class FCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(500, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 2)
    )


  def forward(self, x):
    return self.layers(x)


model = FCNN().to(device)

print_model(model)
train(model, train_loader, test_loader, epochs=EPOCHS)
accuracy = eval_function(model, test_loader)
final_test(model,test_loader)
print(f'Test accuracy: {accuracy:.2f}%')