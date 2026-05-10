import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, random_split

class FordDS(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def load_txt(p):
    d = np.loadtxt(p)
    y = np.where(d[:,0] == -1, 0, 1).astype(int)
    x = d[:,1:]
    print(f"[data] {os.path.basename(p)}: {x.shape}  classes={np.bincount(y).tolist()}")
    return x, y

def get_loaders(train_p, test_a_p, test_b_p, val_ratio=0.2, bs=64, seed=42):
    x_tr, y_tr = load_txt(train_p)
    x_a, y_a = load_txt(test_a_p)
    x_b, y_b = load_txt(test_b_p)
    m, s = x_tr.mean(), x_tr.std()
    x_tr, x_a, x_b = [(v - m) / s for v in (x_tr, x_a, x_b)]
    n, v = len(y_tr), int(len(y_tr) * val_ratio)
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(FordDS(x_tr, y_tr), [n - v, v], generator=g)
    return (DataLoader(tr, bs, shuffle=True),
            DataLoader(va, bs),
            DataLoader(FordDS(x_a, y_a), bs),
            DataLoader(FordDS(x_b, y_b), bs))