"""
train.py
========
Four-stream ST-GCN training with:
  - Joint + Motion + Bone + Bone Motion streams
  - Early fusion (cross-stream interaction before classifier)
  - Adaptive graph topology (learnable edges)
  - DropGraph regularization
  - K-Fold cross validation (k=5)
  - Full augmentation pipeline
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

from models.st_gcn_twostream import Model as FourStreamSTGCN

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

GRAPH_ARGS = {
    'layout'  : 'mediapipe_51',
    'strategy': 'spatial',
}

# Model flags
ADAPTIVE_GRAPH  = True    # learnable adjacency on top of fixed graph
DROP_GRAPH_PROB = 0.1     # DropGraph probability
EARLY_FUSION    = True    # concatenate features before classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_skeleton(x):
    T, F = x.shape
    if np.random.rand() < 0.5:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.85, 1.15)
    if np.random.rand() < 0.5:
        angle        = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot        = x.copy()
        for i in range(0, F, 2):
            xv = x[:, i]; yv = x[:, i+1]
            x_rot[:, i]   = cos_a * xv - sin_a * yv
            x_rot[:, i+1] = sin_a * xv + cos_a * yv
        x = x_rot
    if np.random.rand() < 0.5:
        x[:, 0::2] = -x[:, 0::2]
    if np.random.rand() < 0.3:
        mask = np.random.rand(T) > 0.1
        kept = x[mask]
        if len(kept) >= 2:
            x = kept[np.linspace(0, len(kept)-1, T).astype(int)]
    if np.random.rand() < 0.3:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        x    = x[np.clip((warp/warp[-1]*(T-1)).astype(int), 0, T-1)]
    if np.random.rand() < 0.3:
        crop_len = int(T * np.random.uniform(0.8, 1.0))
        start    = np.random.randint(0, T - crop_len + 1)
        cropped  = x[start:start+crop_len]
        x        = cropped[np.linspace(0, len(cropped)-1, T).astype(int)]
    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FourStreamDataset(Dataset):
    """
    Loads all 4 streams: joint, bone, motion (not stored — computed in model),
    bone_motion.
    Converts (T, F) → (C, T, V) for each stream.
    """

    def __init__(self, X_joint, X_bone, X_bone_motion, y, augment=False):
        self.X_joint       = X_joint.astype(np.float32)
        self.X_bone        = X_bone.astype(np.float32)
        self.X_bone_motion = X_bone_motion.astype(np.float32)
        self.y             = y.astype(np.int64)
        self.augment       = augment

    def __len__(self):
        return len(self.y)

    def to_graph(self, x):
        """(T, F) → (C, T, V)"""
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
        raise FileNotFoundError("No joint data found. Run preprocessing first.")

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))

    def load_stream(fname, fallback_fn, name):
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  {name:<14}: {fname}")
            return np.load(path)
        print(f"  ⚠️  {fname} not found — run normalize.py")
        return fallback_fn(X_joint)

    X_bone        = load_stream("X_bones.npy",
                                lambda x: np.zeros_like(x), "Bone stream")
    X_bone_motion = load_stream("X_bone_motion.npy",
                                lambda x: np.zeros_like(x), "Bone motion")

    print(f"\n  X shape  : {X_joint.shape}")
    print(f"  y shape  : {y.shape}")
    print(f"  Joints   : {X_joint.shape[2]//2}")
    print(f"  Streams  : joint + motion(internal) + bone + bone_motion = 4")

    return X_joint, X_bone, X_bone_motion, y


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(Xj_tr, Xb_tr, Xbm_tr, y_tr,
               Xj_val, Xb_val, Xbm_val, y_val,
               num_classes, fold):

    train_loader = DataLoader(
        FourStreamDataset(Xj_tr, Xb_tr, Xbm_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        FourStreamDataset(Xj_val, Xb_val, Xbm_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting = True,
        adaptive_graph            = ADAPTIVE_GRAPH,
        drop_graph_prob           = DROP_GRAPH_PROB,
        early_fusion              = EARLY_FUSION,
        dropout                   = DROPOUT
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
    print(f"   Train: {len(Xj_tr)}  Val: {len(Xj_val)}")

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
    print(f"  Model         : Four-Stream ST-GCN")
    print(f"  Streams       : joint + motion + bone + bone_motion")
    print(f"  Fusion        : {'Early (concat+FC)' if EARLY_FUSION else 'Late (sum logits)'}")
    print(f"  Adaptive graph: {ADAPTIVE_GRAPH}")
    print(f"  DropGraph     : {DROP_GRAPH_PROB}")
    print(f"  Params        : {total:,}")

    idx_all          = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )

    Xj_test  = X_joint[idx_test]
    Xb_test  = X_bone[idx_test]
    Xbm_test = X_bone_motion[idx_test]
    y_test   = y[idx_test]

    Xj_tv  = X_joint[idx_tv]
    Xb_tv  = X_bone[idx_tv]
    Xbm_tv = X_bone_motion[idx_tv]
    y_tv   = y[idx_tv]

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), Xj_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n── K-Fold Cross Validation (k={K_FOLDS}) ───────────────────")
    print(f"   Train+Val: {len(y_tv)}  Test (held out): {len(y_test)}")
    print(f"{'='*60}")
    print(f"  Four-Stream ST-GCN | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
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
        print(f"  Fold {fold} best val acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = state

    print(f"\n── K-Fold Results ──────────────────────────────────")
    for i, acc in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {acc:.3f}")
    print(f"   Mean  : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")

    # ── Final test ────────────────────────────────────────────────────────────
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
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
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