"""
train.py — Four-stream ST-GCN, optimized for 20/50-class WLASL.

SWA fix: accumulate weights throughout training, do ONE update_bn at the end,
evaluate SWA model once after early stopping fires. No per-epoch BN resets.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.st_gcn_twostream import Model as FourStreamSTGCN

OUTPUT_DIR    = os.path.join(ROOT, "output")
MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn.pth")

BATCH_SIZE    = 4
EPOCHS        = 300
LEARNING_RATE = 0.0005
DROPOUT       = 0.4
SEED          = 42
K_FOLDS       = 4       # back to 4 — more val samples per fold
TEST_SPLIT    = 0.15
PATIENCE      = 35

SWA_START_EP  = 50      # start accumulating SWA weights at this epoch
SWA_LR        = 0.0001

GRAPH_ARGS        = {"layout": "mediapipe_51", "strategy": "spatial"}
ADAPTIVE_GRAPH    = True
DROP_GRAPH_PROB   = 0.1
EARLY_FUSION      = True
LABEL_SMOOTHING   = 0.1
USE_WEIGHTED_LOSS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth = torch.zeros_like(log_prob)
            smooth.fill_(self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w = self.weight[target].unsqueeze(1)
            return -(smooth * log_prob * w).sum(dim=1).mean()
        return -(smooth * log_prob).sum(dim=1).mean()


def compute_class_weights(y, num_classes, device):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def augment_skeleton(x):
    T, F = x.shape
    if np.random.rand() < 0.6:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.015
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.5:
        a = np.random.uniform(-20, 20) * np.pi / 180
        ca, sa = np.cos(a), np.sin(a)
        xr = x.copy()
        for i in range(0, F, 2):
            xv, yv = x[:, i], x[:, i+1]
            xr[:, i] = ca*xv - sa*yv
            xr[:, i+1] = sa*xv + ca*yv
        x = xr
    if np.random.rand() < 0.5:
        x[:, 0::2] = -x[:, 0::2]
    if np.random.rand() < 0.4:
        mask = np.random.rand(T) > 0.15
        kept = x[mask]
        if len(kept) >= 2:
            x = kept[np.linspace(0, len(kept)-1, T).astype(int)]
    if np.random.rand() < 0.4:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        x = x[np.clip((warp/warp[-1]*(T-1)).astype(int), 0, T-1)]
    if np.random.rand() < 0.4:
        cl = int(T * np.random.uniform(0.75, 1.0))
        st = np.random.randint(0, T - cl + 1)
        cr = x[st:st+cl]
        x = cr[np.linspace(0, len(cr)-1, T).astype(int)]
    return x.astype(np.float32)


class FourStreamDataset(Dataset):
    def __init__(self, X_joint, X_bone, X_bone_motion, y, augment=False):
        self.X_joint = X_joint.astype(np.float32)
        self.X_bone  = X_bone.astype(np.float32)
        self.X_bm    = X_bone_motion.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self): return len(self.y)

    def to_graph(self, x):
        T, F = x.shape
        V = F // 2
        return x.reshape(T, V, 2).transpose(2, 0, 1)

    def __getitem__(self, idx):
        xj = self.X_joint[idx].copy()
        xb = self.X_bone[idx].copy()
        xm = self.X_bm[idx].copy()
        if self.augment:
            xj = augment_skeleton(xj)
            xb = augment_skeleton(xb)
            xm = augment_skeleton(xm)
        return (torch.tensor(self.to_graph(xj), dtype=torch.float32),
                torch.tensor(self.to_graph(xb), dtype=torch.float32),
                torch.tensor(self.to_graph(xm), dtype=torch.float32),
                torch.tensor(self.y[idx],        dtype=torch.long))


def global_normalize(X_tr, *X_others):
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1),  keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    return ((X_tr - mean) / std,) + tuple((X - mean) / std for X in X_others)


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

    def ls(fname, name):
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  {name:<14}: {fname}")
            return np.load(path)
        print(f"  WARNING: {fname} not found — using zeros")
        return np.zeros_like(X_joint)

    X_bone = ls("X_bones.npy",       "Bone stream")
    X_bm   = ls("X_bone_motion.npy", "Bone motion")
    u, c   = np.unique(y, return_counts=True)
    print(f"\n  X shape  : {X_joint.shape}")
    print(f"  Classes  : {len(u)} | min: {c.min()} | max: {c.max()} | mean: {c.mean():.1f}")
    return X_joint, X_bone, X_bm, y


def eval_model(m, val_loader):
    m.eval()
    correct = 0
    with torch.no_grad():
        for xj, xb, xbm, yb in val_loader:
            xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                xbm.to(DEVICE), yb.to(DEVICE))
            correct += (m(xj, bone=xb, bone_motion=xbm)
                        .argmax(1) == yb).sum().item()
    return correct / len(val_loader.dataset)


def train_fold(Xj_tr, Xb_tr, Xbm_tr, y_tr,
               Xj_val, Xb_val, Xbm_val, y_val,
               num_classes, fold):

    train_loader = DataLoader(
        FourStreamDataset(Xj_tr, Xb_tr, Xbm_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(
        FourStreamDataset(Xj_val, Xb_val, Xbm_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    model = FourStreamSTGCN(
        2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True, adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB, early_fusion=EARLY_FUSION,
        dropout=DROPOUT).to(DEVICE)

    swa_model   = AveragedModel(model)
    swa_updates = 0   # count how many times we've updated SWA

    cw = compute_class_weights(y_tr, num_classes, DEVICE) if USE_WEIGHTED_LOSS else None
    criterion = LabelSmoothingLoss(num_classes, LABEL_SMOOTHING, cw)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=5e-2)

    def lr_lambda(ep):
        w = 10
        if ep < w: return ep / w
        return 0.5 * (1 + np.cos(np.pi * (ep - w) / (EPOCHS - w)))

    scheduler     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    best_val_acc  = 0.0
    best_state    = None
    patience_ctr  = 0
    stopped_epoch = EPOCHS

    print(f"\n── Fold {fold} ─────────────────────────────────────────")
    print(f"   Train: {len(Xj_tr)}  Val: {len(Xj_val)}  "
          f"Patience: {PATIENCE}  SWA from ep {SWA_START_EP}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_correct = 0
        for xj, xb, xbm, yb in train_loader:
            xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                xbm.to(DEVICE), yb.to(DEVICE))
            optimizer.zero_grad()
            out = model(xj, bone=xb, bone_motion=xbm)
            criterion(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_correct += (out.argmax(1) == yb).sum().item()
        tr_acc = tr_correct / len(train_loader.dataset)

        # ── LR schedule ───────────────────────────────────────────────────────
        if epoch >= SWA_START_EP:
            swa_model.update_parameters(model)
            swa_updates += 1
            swa_scheduler.step()
            if swa_updates == 1:
                print(f"  [SWA accumulation started at ep {epoch}]")
        else:
            scheduler.step()

        # ── Validate base model only (fast, every epoch) ──────────────────────
        v_acc = eval_model(model, val_loader)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
            marker = "  <- best"
        else:
            patience_ctr += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{EPOCHS} | Train Acc: {tr_acc:.3f} | "
                  f"Val Acc: {v_acc:.3f}{marker} | "
                  f"Gap: {tr_acc - v_acc:.3f} | "
                  f"Patience: {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            stopped_epoch = epoch
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Post-training: evaluate SWA model if we have enough updates ───────────
    if swa_updates >= 5:
        print(f"  Calibrating SWA BN stats ({swa_updates} weight snapshots)...")
        update_bn(train_loader, swa_model, device=DEVICE)
        v_swa = eval_model(swa_model, val_loader)
        print(f"  SWA val acc: {v_swa:.3f}  (base best: {best_val_acc:.3f})")
        if v_swa > best_val_acc:
            best_val_acc = v_swa
            best_state = {
                (k[len("module."):] if k.startswith("module.") else k): v.cpu().clone()
                for k, v in swa_model.state_dict().items()
            }
            print(f"  SWA improved val acc to {best_val_acc:.3f} — using SWA weights")
        else:
            print(f"  Base model better — keeping base weights")
    else:
        print(f"  SWA skipped (only {swa_updates} updates — early stopping too soon)")

    print(f"  Best val acc: {best_val_acc:.3f} (stopped at epoch {stopped_epoch})")
    return best_val_acc, best_state


def train():
    print("\nLoading data streams...")
    X_joint, X_bone, X_bm, y = load_data()
    num_classes = len(np.unique(y))

    _m = FourStreamSTGCN(2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True, adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB, early_fusion=EARLY_FUSION,
        dropout=DROPOUT)
    total = sum(p.numel() for p in _m.parameters())
    del _m

    print(f"\n  Classes: {num_classes}  Params: {total:,}  K-Folds: {K_FOLDS}")
    print(f"  Epochs: {EPOCHS} (patience={PATIENCE})  SWA from ep {SWA_START_EP}")
    print(f"  Dropout: {DROPOUT}  DropGraph: {DROP_GRAPH_PROB}  wd: 5e-2")

    idx_all = np.arange(len(y))
    idx_tv, idx_test = train_test_split(idx_all, test_size=TEST_SPLIT,
                                        random_state=SEED, stratify=y)
    Xj_test  = X_joint[idx_test]; Xb_test  = X_bone[idx_test]
    Xbm_test = X_bm[idx_test];    y_test   = y[idx_test]
    Xj_tv    = X_joint[idx_tv];   Xb_tv    = X_bone[idx_tv]
    Xbm_tv   = X_bm[idx_tv];      y_tv     = y[idx_tv]

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), Xj_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n  Train+Val: {len(y_tv)}  Test: {len(y_test)}")
    print(f"{'='*60}")

    skf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    best_acc  = 0.0
    best_state = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xj_tv, y_tv), 1):
        Xj_tr_n,  Xj_val_n,  Xj_test_n  = global_normalize(
            Xj_tv[tr_idx],  Xj_tv[val_idx],  Xj_test)
        Xb_tr_n,  Xb_val_n,  Xb_test_n  = global_normalize(
            Xb_tv[tr_idx],  Xb_tv[val_idx],  Xb_test)
        Xbm_tr_n, Xbm_val_n, Xbm_test_n = global_normalize(
            Xbm_tv[tr_idx], Xbm_tv[val_idx], Xbm_test)

        val_acc, state = train_fold(
            Xj_tr_n, Xb_tr_n, Xbm_tr_n, y_tv[tr_idx],
            Xj_val_n, Xb_val_n, Xbm_val_n, y_tv[val_idx],
            num_classes, fold)
        fold_accs.append(val_acc)
        print(f"  Fold {fold} best val acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc      = val_acc
            best_state    = state
            best_Xj_test  = Xj_test_n
            best_Xb_test  = Xb_test_n
            best_Xbm_test = Xbm_test_n

    print(f"\n── K-Fold Results ──────────────────────────────────")
    for i, a in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {a:.3f}")
    print(f"   Mean  : {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")

    model = FourStreamSTGCN(2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True, adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB, early_fusion=EARLY_FUSION,
        dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_SAVE)

    test_loader = DataLoader(
        FourStreamDataset(best_Xj_test, best_Xb_test, best_Xbm_test,
                          y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tc = 0
    model.eval()
    with torch.no_grad():
        for xj, xb, xbm, yb in test_loader:
            xj, xb, xbm, yb = (xj.to(DEVICE), xb.to(DEVICE),
                                xbm.to(DEVICE), yb.to(DEVICE))
            tc += (model(xj, bone=xb, bone_motion=xbm)
                   .argmax(1) == yb).sum().item()

    print(f"\n── Final Test Set Evaluation ───────────────────────")
    print(f"   Test Accuracy : {tc/len(y_test):.3f}  ({tc}/{len(y_test)} correct)")
    print(f"   Best CV Acc   : {best_acc:.3f}")
    print(f"   CV Mean       : {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")
    print(f"   Model saved   : {MODEL_SAVE}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()