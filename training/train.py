"""
train.py — Training with original ST-GCN (two-stream)
======================================================
Uses the ported original ST-GCN code from Yan et al. AAAI 2018.
Two streams: joint positions + motion (second-order difference).
K-Fold cross validation with full augmentation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.st_gcn_twostream import Model as TwoStreamSTGCN

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(ROOT, "output")
MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn.pth")

BATCH_SIZE    = 16
EPOCHS        = 100
LEARNING_RATE = 0.001
DROPOUT       = 0.4
SEED          = 42
K_FOLDS       = 5
TEST_SPLIT    = 0.15

# Graph args — uses our MediaPipe 51-joint layout with spatial strategy
# (spatial strategy = original paper's best performing partition)
GRAPH_ARGS = {
    'layout'  : 'mediapipe_51',
    'strategy': 'spatial',
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_skeleton(x):
    """Full augmentation for (T, F) skeleton."""
    T, F = x.shape

    # Gaussian noise
    if np.random.rand() < 0.5:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01

    # Random scale
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.85, 1.15)

    # Random rotation
    if np.random.rand() < 0.5:
        angle    = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot    = x.copy()
        for i in range(0, F, 2):
            xv = x[:, i]; yv = x[:, i+1]
            x_rot[:, i]   = cos_a * xv - sin_a * yv
            x_rot[:, i+1] = sin_a * xv + cos_a * yv
        x = x_rot

    # Horizontal flip
    if np.random.rand() < 0.5:
        x[:, 0::2] = -x[:, 0::2]

    # Frame drop
    if np.random.rand() < 0.3:
        mask = np.random.rand(T) > 0.1
        kept = x[mask]
        if len(kept) >= 2:
            idx = np.linspace(0, len(kept)-1, T).astype(int)
            x   = kept[idx]

    # Time warp
    if np.random.rand() < 0.3:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        warp = (warp / warp[-1] * (T-1)).astype(int)
        x    = x[np.clip(warp, 0, T-1)]

    # Temporal crop
    if np.random.rand() < 0.3:
        crop_len = int(T * np.random.uniform(0.8, 1.0))
        start    = np.random.randint(0, T - crop_len + 1)
        cropped  = x[start:start+crop_len]
        idx      = np.linspace(0, len(cropped)-1, T).astype(int)
        x        = cropped[idx]

    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class STGCNDataset(Dataset):
    """
    Converts flat (T, F) keypoints → (C, T, V) graph format.
    F = 102, C = 2, V = 51
    """

    def __init__(self, X, y, augment=False):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def to_graph(self, x):
        """(T, F) → (C, T, V)"""
        T, F = x.shape
        V    = F // 2
        # (T, F) → (T, V, 2) → (2, T, V)
        return x.reshape(T, V, 2).transpose(2, 0, 1)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        if self.augment:
            x = augment_skeleton(x)

        return (
            torch.tensor(self.to_graph(x), dtype=torch.float32),
            torch.tensor(self.y[idx],      dtype=torch.long)
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"Loading {fname} ...")
            X = np.load(path)
            break
    else:
        raise FileNotFoundError(
            "No data found. Run preprocessing scripts first."
        )

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))
    print(f"   X shape  : {X.shape}")
    print(f"   y shape  : {y.shape}")
    print(f"   Joints   : {X.shape[2]//2}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(X_tr, y_tr, X_val, y_val, num_classes, fold):

    train_loader = DataLoader(
        STGCNDataset(X_tr,  y_tr,  augment=True),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        STGCNDataset(X_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = TwoStreamSTGCN(
        in_channels              = 2,
        num_class                = num_classes,
        graph_args               = GRAPH_ARGS,
        edge_importance_weighting= True,
        dropout                  = DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_acc = 0.0
    best_state   = None

    print(f"\n── Fold {fold} ─────────────────────────────────────────")
    print(f"   Train: {len(X_tr)}  Val: {len(X_val)}")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss = tr_correct = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (out.argmax(1) == y_b).sum().item()

        tr_acc  = tr_correct / len(train_loader.dataset)
        tr_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss = v_correct = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b  = X_b.to(DEVICE), y_b.to(DEVICE)
                out        = model(X_b)
                v_loss    += criterion(out, y_b).item()
                v_correct += (out.argmax(1) == y_b).sum().item()

        v_acc  = v_correct / len(val_loader.dataset)
        v_loss /= len(val_loader)
        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            marker = "  ← best"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                f"Val Loss: {v_loss:.4f} Acc: {v_acc:.3f}{marker}"
            )

    return best_val_acc, best_state


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train():
    X, y        = load_data()
    num_classes = len(np.unique(y))
    num_joints  = X.shape[2] // 2

    # Count params
    _m     = TwoStreamSTGCN(2, num_classes, GRAPH_ARGS, True, dropout=DROPOUT)
    total  = sum(p.numel() for p in _m.parameters())
    del _m

    print(f"   Classes  : {num_classes}")
    print(f"   Joints   : {num_joints}")
    print(f"   Model    : Two-Stream ST-GCN (Yan et al. 2018)")
    print(f"   Strategy : {GRAPH_ARGS['strategy']} partition")
    print(f"   Params   : {total:,}")

    # Hold out test set
    idx_all              = np.arange(len(y))
    idx_tv, idx_test     = train_test_split(
        idx_all, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    X_test, y_test       = X[idx_test], y[idx_test]
    X_tv,   y_tv         = X[idx_tv],   y[idx_tv]

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n── K-Fold Cross Validation (k={K_FOLDS}) ───────────────────")
    print(f"   Train+Val: {len(y_tv)}  Test (held out): {len(y_test)}")
    print(f"{'='*60}")
    print(f"  Two-Stream ST-GCN | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    skf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    best_acc  = 0.0
    best_state= None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), 1):
        val_acc, state = train_fold(
            X_tv[tr_idx], y_tv[tr_idx],
            X_tv[val_idx], y_tv[val_idx],
            num_classes, fold
        )
        fold_accs.append(val_acc)
        print(f"  Fold {fold} best val acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = state

    print(f"\n── K-Fold Results ──────────────────────────────────")
    for i, acc in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {acc:.3f}")
    print(f"   Mean  : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")

    # Final test with best model
    model = TwoStreamSTGCN(
        in_channels=2, num_class=num_classes,
        graph_args=GRAPH_ARGS, edge_importance_weighting=True,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_SAVE)

    test_loader = DataLoader(
        STGCNDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b     = X_b.to(DEVICE), y_b.to(DEVICE)
            test_correct += (model(X_b).argmax(1) == y_b).sum().item()

    test_acc = test_correct / len(y_test)

    print(f"\n── Final Test Set Evaluation ───────────────────────")
    print(f"   Test Accuracy : {test_acc:.3f}  ({test_correct}/{len(y_test)} correct)")
    print(f"   Best CV Acc   : {best_acc:.3f}")
    print(f"   CV Mean       : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")
    print(f"   Model saved → {MODEL_SAVE}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()