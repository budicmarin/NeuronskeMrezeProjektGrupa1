import os, json, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data import get_loaders
from models import CNN1D, GRU
from plot import plot_hist, plot_cm, plot_cmp

#postavljanje uređaja koji pokreće kod
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Postavljanje broja Epoha=50
    Learning rate =1e-3
    Pattience =10
'''
EPOCHS, LR, BS, VALR, PAT = 50, 1e-3, 64, 0.2, 10

#dohvaćanje datoteka za učenje i testiranje i postavljanje njihovih putanja
SD = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "train": os.path.join(SD, "FordA", "FordA_TRAIN.txt"),
    "test_a": os.path.join(SD, "FordA", "FordA_TEST.txt"),
    "test_b": os.path.join(SD, "FordB", "FordB_TEST.txt"),
}
#funkcija za pokretanje epoha
def run_epoch(m, dl, crit, opt=None, clip=None):
    train = opt is not None
    m.train(train)
    L, P, Y = 0.0, [], []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            opt.zero_grad()
            out = m(x)
            l = crit(out, y)
            l.backward()
            if clip:
                nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
        else:
            with torch.no_grad():
                out = m(x)
                l = crit(out, y)
        L += l.item() * len(y)
        P += out.argmax(1).cpu().tolist()
        Y += y.cpu().tolist()
    return L / len(dl.dataset), accuracy_score(Y, P), Y, P
#funkcija za dobivanje podataka
def get_metrics(y, p):
    return {
        "acc": accuracy_score(y, p),
        "prec": precision_score(y, p, zero_division=0),
        "rec": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "cm": confusion_matrix(y, p).tolist(),
    }
#funkcija za trening
def train_model(m, tr_dl, va_dl, name, clip=None):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=PAT // 2)
    hist, best, wait, best_sd = {"tl": [], "vl": [], "va": []}, float("inf"), 0, None
    print(f"\n>> Training {name}  device={DEVICE}")
    for e in range(1, EPOCHS + 1):
        tl, _, _, _ = run_epoch(m, tr_dl, crit, opt, clip)
        vl, va, _, _ = run_epoch(m, va_dl, crit)
        hist["tl"].append(tl); hist["vl"].append(vl); hist["va"].append(va)
        sched.step(vl)
        print(f"  E{e:02d}: train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")
        if vl < best:
            best, wait, best_sd = vl, 0, m.state_dict()
        else:
            wait += 1
            if wait >= PAT:
                print(f"  Early stop @ epoch {e}")
                break
    m.load_state_dict(best_sd)
    os.makedirs("ckpt", exist_ok=True)
    torch.save(best_sd, f"ckpt/{name}.pt")
    print(f"  Saved ckpt/{name}.pt")
    return m, hist
#funkcija za testiranje modela
def test_model(m, dl, name, ds):
    vl, va, y, p = run_epoch(m, dl, nn.CrossEntropyLoss())
    r = get_metrics(y, p)
    r["loss"] = vl
    print(f"  >> {name} on {ds}:  loss={vl:.4f}  acc={r['acc']:.4f}  f1={r['f1']:.4f}")
    return r, y, p
#main funkcija
def main():

    tr_dl, va_dl, ta_dl, tb_dl = get_loaders(PATHS["train"], PATHS["test_a"], PATHS["test_b"], VALR, BS)
    results = {}
    #
    for name, Model, cfg, clip in [
        ("CNN1D", CNN1D, {"nc": 2}, None),
        ("GRU", GRU, {"nc": 2, "hid": 128, "layers": 1}, 1.0),
    ]:
        m = Model(**cfg).to(DEVICE)
        m, hist = train_model(m, tr_dl, va_dl, name, clip)
        ra, ya, pa = test_model(m, ta_dl, name, "FordA_TEST")
        rb, yb, pb = test_model(m, tb_dl, name, "FordB_TEST")
        results[name] = {"FordA_TEST": ra, "FordB_TEST": rb, "hist": hist}
        plot_hist(hist, name)
        plot_cm(ya, pa, name, "FordA_TEST")
        plot_cm(yb, pb, name, "FordB_TEST")
        payload = {"model_cfg": cfg, "train_cfg": {"epochs": len(hist["tl"]), "lr": LR, "batch": BS, "val_ratio": VALR}, "FordA_TEST": ra, "FordB_TEST": rb}
        with open(f"ckpt/{name}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved ckpt/{name}.json")
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump({k: {"FordA_TEST": v["FordA_TEST"], "FordB_TEST": v["FordB_TEST"]} for k, v in results.items()}, f, indent=2)
    with open("results/histories.json", "w") as f:
        json.dump({k: v["hist"] for k, v in results.items()}, f, indent=2)
    plot_cmp({k: {"FordA_TEST": v["FordA_TEST"], "FordB_TEST": v["FordB_TEST"]} for k, v in results.items()})
    print("\nDone. Check ckpt/, results/, plots/")

if __name__ == "__main__":
    main()