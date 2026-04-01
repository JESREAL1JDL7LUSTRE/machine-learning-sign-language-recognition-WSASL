"""
train.py — Optimized for 50-class WLASL
=========================================
Key fixes vs previous run:
  - K=4 folds (some classes have only 5 samples, K=5 would crash)
  - Early stopping with patience=35 (val acc is noisy on 50 classes)
  - Warmup (10 ep) + CosineAnnealing — gentler start
  - drop_last=True on train loader — prevents BatchNorm crash on last batch
  - weight_decay=5e-2 — strong regularization for 13M params / 508 samples
  - DROPOUT=0.4, DROP_GRAPH=0.1 — less aggressive than 20-class (need capacity)
  - LABEL_SMOOTHING=0.1 — less smoothing (50 classes is already hard)
  - EPOCHS=300 with early stopping — previous run stopped at ep 50-80 still climbing
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.st_gcn_twostream import Model as FourStreamSTGCN

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(ROOT, "output")
MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn.pth")

BATCH_SIZE    = 4       # small — noisier gradients = implicit regularization
EPOCHS        = 300     # more room — previous run stopped at ep 50-80 still climbing
LEARNING_RATE = 0.0005
DROPOUT       = 0.4     # reduced vs 20-class — 50 classes needs more capacity
SEED          = 42
K_FOLDS       = 4       # must be 4 — some classes have only 5 samples
TEST_SPLIT    = 0.15

# Early stopping
PATIENCE      = 35      # val acc oscillates on 50 classes — give it room to recover

GRAPH_ARGS = {
    'layout'  : 'mediapipe_51',
    'strategy': 'spatial',
}

ADAPTIVE_GRAPH    = True
DROP_GRAPH_PROB   = 0.1   # reduced — less aggressive than 20-class run
EARLY_FUSION      = True
LABEL_SMOOTHING   = 0.1   # less smoothing — 50 classes is already hard enough
USE_WEIGHTED_LOSS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.weight      = weight

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth = torch.zeros_like(log_prob)
            smooth.fill_(self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w    = self.weight[target].unsqueeze(1)
            loss = -(smooth * log_prob * w).sum(dim=1).mean()
        else:
            loss = -(smooth * log_prob).sum(dim=1).mean()
        return loss


def compute_class_weights(y, num_classes, device):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_skeleton(x):
    T, F = x.shape
    if np.random.rand() < 0.6:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.015
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.5:
        angle        = np.random.uniform(-20, 20) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot        = x.copy()
        for i in range(0, F, 2):
            xv = x[:, i]; yv = x[:, i+1]
            x_rot[:, i]   = cos_a * xv - sin_a * yv
            x_rot[:, i+1] = sin_a * xv + cos_a * yv
        x = x_rot
    if np.random.rand() < 0.5:
        x[:, 0::2] = -x[:, 0::2]
    if np.random.rand() < 0.4:
        mask = np.random.rand(T) > 0.15
        kept = x[mask]
        if len(kept) >= 2:
            x = kept[np.linspace(0, len(kept)-1, T).astype(int)]
    if np.random.rand() < 0.4:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        x    = x[np.clip((warp/warp[-1]*(T-1)).astype(int), 0, T-1)]
    if np.random.rand() < 0.4:
        crop_len = int(T * np.random.uniform(0.75, 1.0))
        start    = np.random.randint(0, T - crop_len + 1)
        cropped  = x[start:start+crop_len]
        x        = cropped[np.linspace(0, len(cropped)-1, T).astype(int)]
    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FourStreamDataset(Dataset):
    def __init__(self, X_joint, X_bone, X_bone_motion, y, augment=False):
        self.X_joint       = X_joint.astype(np.float32)
        self.X_bone        = X_bone.astype(np.float32)
        self.X_bone_motion = X_bone_motion.astype(np.float32)
        self.y             = y.astype(np.int64)
        self.augment       = augment

    def __len__(self):
        return len(self.y)

    def to_graph(self, x):
        T, F = x.shape
        V    = F // 2
        return x.reshape(T, V, 2).transpose(2, 0, 1)

    def __getitem__(self, idx):
        xj  = self.X_joint[idx].copy()
        xb  = self.X_bone[idx].copy()
        xbm = self.X_bone_motion[idx].copy()

        if self.augment:
            xj  = augment_skeleton(xj)
            xb  = augment_skeleton(xb)
            xbm = augment_skeleton(xbm)

        return (
            torch.tensor(self.to_graph(xj),  dtype=torch.float32),
            torch.tensor(self.to_graph(xb),  dtype=torch.float32),
            torch.tensor(self.to_graph(xbm), dtype=torch.float32),
            torch.tensor(self.y[idx],         dtype=torch.long)
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  Joint stream  : {fname}")
            X_joint = np.load(path)
            break
    else:
        raise FileNotFoundError("No joint data found.")

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))

    def load_stream(fname, name):
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  {name:<14}: {fname}")
            return np.load(path)
        print(f"  ⚠️  {fname} not found — using zeros")
        return np.zeros_like(X_joint)

    X_bone        = load_stream("X_bones.npy",       "Bone stream")
    X_bone_motion = load_stream("X_bone_motion.npy", "Bone motion")

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  X shape  : {X_joint.shape}")
    print(f"  Classes  : {len(unique)} | min: {counts.min()} | max: {counts.max()} | mean: {counts.mean():.1f}")

    return X_joint, X_bone, X_bone_motion, y


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD WITH EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(Xj_tr, Xb_tr, Xbm_tr, y_tr,
               Xj_val, Xb_val, Xbm_val, y_val,
               num_classes, fold):

    train_loader = DataLoader(
        FourStreamDataset(Xj_tr, Xb_tr, Xbm_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        drop_last=True   # prevents BatchNorm crash on odd-sized last batch
    )
    val_loader = DataLoader(
        FourStreamDataset(Xj_val, Xb_val, Xbm_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        drop_last=False
    )

    model = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting = True,
        adaptive_graph            = ADAPTIVE_GRAPH,
        drop_graph_prob           = DROP_GRAPH_PROB,
        early_fusion              = EARLY_FUSION,
        dropout                   = DROPOUT
    ).to(DEVICE)

    if USE_WEIGHTED_LOSS:
        class_weights = compute_class_weights(y_tr, num_classes, DEVICE)
    else:
        class_weights = None

    criterion = LabelSmoothingLoss(num_classes, LABEL_SMOOTHING, class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 5e-2   # strong weight decay for 13M params / ~320 samples
    )

    # Warmup 10 epochs then cosine decay — gentler start prevents early divergence
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc  = 0.0
    best_state    = None
    patience_ctr  = 0
    stopped_epoch = EPOCHS

    print(f"\n── Fold {fold} ─────────────────────────────────────────")
    print(f"   Train: {len(Xj_tr)}  Val: {len(Xj_val)}  Patience: {PATIENCE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = 0

        for xj, xb, xbm, yb in train_loader:
            xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                xbm.to(DEVICE), yb.to(DEVICE))
            optimizer.zero_grad()
            out  = model(xj, bone=xb, bone_motion=xbm)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (out.argmax(1) == yb).sum().item()

        tr_acc  = tr_correct / len(train_loader.dataset)
        tr_loss /= len(train_loader)

        model.eval()
        v_loss = v_correct = 0

        with torch.no_grad():
            for xj, xb, xbm, yb in val_loader:
                xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                    xbm.to(DEVICE), yb.to(DEVICE))
                out         = model(xj, bone=xb, bone_motion=xbm)
                v_loss     += criterion(out, yb).item()
                v_correct  += (out.argmax(1) == yb).sum().item()

        v_acc  = v_correct / len(val_loader.dataset)
        v_loss /= len(val_loader)
        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            patience_ctr = 0
            marker = "  ← best"
        else:
            patience_ctr += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:3d}/{EPOCHS} | "
                f"Train Acc: {tr_acc:.3f} | "
                f"Val Acc: {v_acc:.3f}{marker} | "
                f"Gap: {tr_acc - v_acc:.3f} | "
                f"Patience: {patience_ctr}/{PATIENCE}"
            )

        if patience_ctr >= PATIENCE:
            stopped_epoch = epoch
            print(f"  ⏹  Early stopping at epoch {epoch} — no val improvement for {PATIENCE} epochs")
            break

    print(f"  Best val acc: {best_val_acc:.3f} (stopped at epoch {stopped_epoch})")
    return best_val_acc, best_state


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("\nLoading data streams...")
    X_joint, X_bone, X_bone_motion, y = load_data()
    num_classes = len(np.unique(y))

    _m    = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True,
        adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB,
        early_fusion=EARLY_FUSION,
        dropout=DROPOUT
    )
    total = sum(p.numel() for p in _m.parameters())
    del _m

    print(f"\n  Classes       : {num_classes}")
    print(f"  Params        : {total:,}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Max epochs    : {EPOCHS} (early stop patience={PATIENCE})")
    print(f"  Dropout       : {DROPOUT}")
    print(f"  DropGraph     : {DROP_GRAPH_PROB}")
    print(f"  Label smooth  : {LABEL_SMOOTHING}")
    print(f"  Weight decay  : 5e-2 (AdamW)")

    idx_all          = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )

    Xj_test  = X_joint[idx_test];      Xb_test  = X_bone[idx_test]
    Xbm_test = X_bone_motion[idx_test]; y_test   = y[idx_test]
    Xj_tv    = X_joint[idx_tv];        Xb_tv    = X_bone[idx_tv]
    Xbm_tv   = X_bone_motion[idx_tv];  y_tv     = y[idx_tv]

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), Xj_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n── K-Fold Cross Validation (k={K_FOLDS}) ───────────────────")
    print(f"   Train+Val: {len(y_tv)}  Test (held out): {len(y_test)}")
    print(f"{'='*60}")

    skf        = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs  = []
    best_acc   = 0.0
    best_state = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xj_tv, y_tv), 1):
        val_acc, state = train_fold(
            Xj_tv[tr_idx],  Xb_tv[tr_idx],  Xbm_tv[tr_idx],  y_tv[tr_idx],
            Xj_tv[val_idx], Xb_tv[val_idx],  Xbm_tv[val_idx], y_tv[val_idx],
            num_classes, fold
        )
        fold_accs.append(val_acc)
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = state

    print(f"\n── K-Fold Results ──────────────────────────────────")
    for i, acc in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {acc:.3f}")
    print(f"   Mean  : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")

    # Final test
    model = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True,
        adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB,
        early_fusion=EARLY_FUSION,
        dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_SAVE)

    test_loader = DataLoader(
        FourStreamDataset(Xj_test, Xb_test, Xbm_test, y_test, augment=False),
        batch_size=16, shuffle=False, num_workers=0
    )

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for xj, xb, xbm, yb in test_loader:
            xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                xbm.to(DEVICE), yb.to(DEVICE))
            test_correct += (model(xj, bone=xb, bone_motion=xbm)
                            .argmax(1) == yb).sum().item()

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