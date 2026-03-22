import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.stgcn import STGCN

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(ROOT, "output")
MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn.pth")

BATCH_SIZE    = 16
EPOCHS        = 200
LEARNING_RATE = 0.001
DROPOUT       = 0.5
SEED          = 42

TRAIN_SPLIT   = 0.80
VAL_SPLIT     = 0.10
TEST_SPLIT    = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class STGCNDataset(Dataset):
    """
    Converts flat (T, F) keypoints → (2, T, V) graph format on the fly.
    F = features (102 or 150), V = F // 2 joints.
    """

    def __init__(self, X, y, augment=False):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()   # (T, F)

        if self.augment:
            # Gaussian noise
            if np.random.rand() < 0.5:
                x += np.random.randn(*x.shape).astype(np.float32) * 0.01
            # Horizontal flip (negate x = even indices)
            if np.random.rand() < 0.5:
                x[:, 0::2] = -x[:, 0::2]
            # Random frame drop
            if np.random.rand() < 0.3:
                T    = len(x)
                mask = np.random.rand(T) > 0.1
                kept = x[mask]
                if len(kept) >= 2:
                    idx2 = np.linspace(0, len(kept)-1, T).astype(int)
                    x    = kept[idx2]
            # Time warp
            if np.random.rand() < 0.3:
                T     = len(x)
                warp  = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
                warp  = (warp / warp[-1] * (T - 1)).astype(int)
                warp  = np.clip(warp, 0, T - 1)
                x     = x[warp]

        # (T, F) → (T, V, 2) → (2, T, V)
        T, F = x.shape
        V    = F // 2
        x_graph = x.reshape(T, V, 2).transpose(2, 0, 1)   # (2, T, V)

        return (
            torch.tensor(x_graph,     dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"Loading {fname} ...")
            X = np.load(path)
            break
    else:
        raise FileNotFoundError(
            "No data found. Run preprocessing scripts first:\n"
            "  python preprocessing/extract.py 20\n"
            "  python preprocessing/normalize.py\n"
            "  python preprocessing/resample.py"
        )

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))
    print(f"   X shape  : {X.shape}")
    print(f"   y shape  : {y.shape}")
    print(f"   Features : {X.shape[2]} → {X.shape[2]//2} joints per frame")
    return X, y


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    X, y        = load_data()
    num_classes = len(np.unique(y))
    num_joints  = X.shape[2] // 2    # 51 if normalized, 75 if raw
    print(f"   Classes  : {num_classes}")
    print(f"   Joints   : {num_joints}")

    # 70 / 15 / 15 split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=SEED, stratify=y_tv
    )

    print(f"\n── Data Split ──────────────────────────────────────")
    print(f"   Train      : {len(X_train)} samples  ({TRAIN_SPLIT*100:.0f}%)")
    print(f"   Validation : {len(X_val)}  samples  ({VAL_SPLIT*100:.0f}%)")
    print(f"   Test       : {len(X_test)}  samples  ({TEST_SPLIT*100:.0f}%)")
    print(f"───────────────────────────────────────────────────")

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    train_loader = DataLoader(STGCNDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(STGCNDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(STGCNDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = STGCN(
        num_classes = num_classes,
        in_channels = 2,
        num_joints  = num_joints,
        dropout     = DROPOUT
    ).to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Model    : ST-GCN")
    print(f"   Joints   : {num_joints}")
    print(f"   Params   : {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  ST-GCN Training  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (out.argmax(1) == y_b).sum().item()

        train_acc  = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b  = X_b.to(DEVICE), y_b.to(DEVICE)
                out        = model(X_b)
                val_loss  += criterion(out, y_b).item()
                val_correct += (out.argmax(1) == y_b).sum().item()

        val_acc  = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            marker = "  ← best"
        else:
            marker = ""

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}{marker}"
        )

    # ── Final test ────────────────────────────────────────────────────────
    print(f"\n── Final Test Set Evaluation ───────────────────────")
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b     = X_b.to(DEVICE), y_b.to(DEVICE)
            test_correct += (model(X_b).argmax(1) == y_b).sum().item()

    test_acc = test_correct / len(test_loader.dataset)
    print(f"   Test Accuracy : {test_acc:.3f}  "
          f"({test_correct}/{len(test_loader.dataset)} correct)")
    print(f"   Best Val Acc  : {best_val_acc:.3f}")
    print(f"───────────────────────────────────────────────────")
    print(f"   Model saved → {MODEL_SAVE}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()