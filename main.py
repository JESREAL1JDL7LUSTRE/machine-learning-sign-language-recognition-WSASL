"""
main.py — 5-Model ASL Sign Language Recognition Comparison Framework
=====================================================================

Usage examples:
  python main.py --bilstm
  python main.py --custom-stgcn
  python main.py --multi-stream-stgcn
  python main.py --2stream-stgcn
  python main.py --4stream-fusion
  python main.py --compare-5
  python main.py --compare-5 --results-only    # charts from cache, no training
  python main.py --compare-5 --epochs 100      # shorter run

Each model run produces:
  output/charts/<model_key>_results.png   — fold bars + confusion matrix
  output/charts/comparison_overview.png   — overall 5-model comparison
  output/model_results.json               — cached results for --results-only
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from models.stgcn               import STGCN, NUM_JOINTS
from models.stgcn_2stream_ported import TwoStreamPortedSTGCN
from models.st_gcn_twostream    import Model as FourStreamSTGCN

# ── Directories ────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(ROOT, "output")
CHARTS_DIR    = os.path.join(OUTPUT_DIR, "charts")
RESULTS_CACHE = os.path.join(OUTPUT_DIR, "model_results.json")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Training hyper-parameters (match train.py) ─────────────────────────────────
SEED          = 42
K_FOLDS       = 4
TEST_SPLIT    = 0.15
BATCH_SIZE    = 4
LEARNING_RATE = 0.0005
DROPOUT       = 0.4
PATIENCE      = 35
SWA_START_EP  = 50
SWA_LR        = 0.0001
LABEL_SMOOTH  = 0.1
WEIGHT_DECAY  = 5e-2
GRAPH_ARGS    = {"layout": "mediapipe_51", "strategy": "spatial"}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Published / known results (for reference lines in charts) ──────────────────
PUBLISHED = {
    "multi-stream-stgcn": {"test": 0.452, "cv": 0.624, "label": "Multi-Stream\nST-GCN (3s)"},
    "2stream-stgcn":      {"test": 0.387, "cv": 0.518, "label": "2-Stream\nST-GCN (ported)"},
    "4stream-fusion":     {"test": 0.484, "cv": 0.512, "label": "4-Stream\nEarly Fusion"},
    "4stream-late-fusion":{"test": None,  "cv": None,  "label": "4-Stream\nLate Fusion"},
}

MODEL_ORDER = ["multi-stream-stgcn", "2stream-stgcn", "4stream-fusion", "4stream-late-fusion"]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED TRAINING UTILITIES  (identical to train.py)
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
            w = self.weight[target].unsqueeze(1)
            return -(smooth * log_prob * w).sum(dim=1).mean()
        return -(smooth * log_prob).sum(dim=1).mean()


def compute_class_weights(y, num_classes, device):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    w       = 1.0 / counts
    w       = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def global_normalize(X_tr, *others):
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1),  keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    return ((X_tr - mean) / std,) + tuple((X - mean) / std for X in others)


def augment_skeleton(x):
    """Same augmentation pipeline as train.py."""
    T, F = x.shape
    V = F // 2

    def random_move(seq):
        angle_c = [-10., -5., 0., 5., 10.]
        scale_c = [0.9, 1.0, 1.1]
        trans_c = [-0.2, -0.1, 0., 0.1, 0.2]
        move_t  = 1
        node = np.arange(0, T, max(1, int(T / move_t))).round().astype(int)
        node = np.append(node, T)
        n    = len(node)
        A  = np.random.choice(angle_c, n)
        S  = np.random.choice(scale_c, n)
        Tx = np.random.choice(trans_c, n)
        Ty = np.random.choice(trans_c, n)
        a  = np.zeros(T); s  = np.zeros(T)
        tx = np.zeros(T); ty = np.zeros(T)
        for i in range(n - 1):
            st, ed = node[i], node[i + 1]
            a[st:ed]  = np.linspace(A[i],  A[i+1],  ed - st) * np.pi / 180
            s[st:ed]  = np.linspace(S[i],  S[i+1],  ed - st)
            tx[st:ed] = np.linspace(Tx[i], Tx[i+1], ed - st)
            ty[st:ed] = np.linspace(Ty[i], Ty[i+1], ed - st)
        seq2 = seq.copy().reshape(T, V, 2)
        for i in range(T):
            ca, sa = np.cos(a[i]) * s[i], np.sin(a[i]) * s[i]
            rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            moved = rot @ seq2[i].T
            moved[0] += tx[i]; moved[1] += ty[i]
            seq2[i] = moved.T
        return seq2.reshape(T, F)

    if np.random.rand() < 0.6:
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.015
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.5:
        a = np.random.uniform(-20, 20) * np.pi / 180
        ca, sa = np.cos(a), np.sin(a)
        xr = x.copy()
        for i in range(0, F, 2):
            xr[:, i]   = ca * x[:, i] - sa * x[:, i+1]
            xr[:, i+1] = sa * x[:, i] + ca * x[:, i+1]
        x = xr
    if np.random.rand() < 0.5:
        x[:, 0::2] = -x[:, 0::2]
    if np.random.rand() < 0.4:
        mask = np.random.rand(T) > 0.15
        kept = x[mask]
        if len(kept) >= 2:
            x = kept[np.linspace(0, len(kept) - 1, T).astype(int)]
    if np.random.rand() < 0.4:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        x = x[np.clip((warp / warp[-1] * (T - 1)).astype(int), 0, T - 1)]
    if np.random.rand() < 0.4:
        cl = int(T * np.random.uniform(0.75, 1.0))
        st = np.random.randint(0, T - cl + 1)
        cr = x[st:st + cl]
        x  = cr[np.linspace(0, len(cr) - 1, T).astype(int)]
    if np.random.rand() < 0.35:
        x = random_move(x)
    return x.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CLASSES
# ══════════════════════════════════════════════════════════════════════════════

def _to_graph(x):
    """Reshape (T, F) → (2, T, V) for ST-GCN."""
    T, F = x.shape
    V = F // 2
    return x.reshape(T, V, 2).transpose(2, 0, 1)          # (2, T, V)


# Removed FlatDataset and SingleGraphDataset (BiLSTM and single-stream ST-GCN removed)


class ThreeStreamDataset(Dataset):
    """For 3-stream ST-GCN (joint, bone, motion)."""
    def __init__(self, X_joint, X_bone, X_motion, y, augment=False):
        self.Xj = X_joint.astype(np.float32)
        self.Xb = X_bone.astype(np.float32)
        self.Xm = X_motion.astype(np.float32)
        self.y  = y.astype(np.int64)
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        xj, xb, xm = self.Xj[i].copy(), self.Xb[i].copy(), self.Xm[i].copy()
        if self.augment:
            xj = augment_skeleton(xj)
            xb = augment_skeleton(xb)
            xm = augment_skeleton(xm)
        return (torch.tensor(_to_graph(xj), dtype=torch.float32),
                torch.tensor(_to_graph(xb), dtype=torch.float32),
                torch.tensor(_to_graph(xm), dtype=torch.float32),
                torch.tensor(self.y[i]))


class TwoStreamDataset(Dataset):
    """For 2-stream ST-GCN (joint + bone, late fusion)."""
    def __init__(self, X_joint, X_bone, y, augment=False):
        self.Xj = X_joint.astype(np.float32)
        self.Xb = X_bone.astype(np.float32)
        self.y  = y.astype(np.int64)
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        xj, xb = self.Xj[i].copy(), self.Xb[i].copy()
        if self.augment:
            xj = augment_skeleton(xj)
            xb = augment_skeleton(xb)
        return (torch.tensor(_to_graph(xj), dtype=torch.float32),
                torch.tensor(_to_graph(xb), dtype=torch.float32),
                torch.tensor(self.y[i]))


class FourStreamDataset(Dataset):
    """For 4-stream ST-GCN (joint, motion, bone, bone_motion)."""
    def __init__(self, X_joint, X_motion, X_bone, X_bm, y, augment=False):
        self.Xj  = X_joint.astype(np.float32)
        self.Xm  = X_motion.astype(np.float32)
        self.Xb  = X_bone.astype(np.float32)
        self.Xbm = X_bm.astype(np.float32)
        self.y   = y.astype(np.int64)
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        xj, xm = self.Xj[i].copy(), self.Xm[i].copy()
        xb, xbm = self.Xb[i].copy(), self.Xbm[i].copy()
        if self.augment:
            xj  = augment_skeleton(xj)
            xm  = augment_skeleton(xm)
            xb  = augment_skeleton(xb)
            xbm = augment_skeleton(xbm)
        return (torch.tensor(_to_graph(xj),  dtype=torch.float32),
                torch.tensor(_to_graph(xm),  dtype=torch.float32),
                torch.tensor(_to_graph(xb),  dtype=torch.float32),
                torch.tensor(_to_graph(xbm), dtype=torch.float32),
                torch.tensor(self.y[i]))


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_streams():
    """Load all available data streams from output directory."""
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            print(f"  Joint stream   : {fname}")
            X_joint = np.load(path)
            break
    else:
        raise FileNotFoundError(
            "No joint data found in output/. Run extract.py + normalize.py + resample.py first.")

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))

    def _load(fname, fallback_zeros=True):
        p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(p):
            print(f"  {fname:<22}: loaded {np.load(p).shape}")
            return np.load(p)
        if fallback_zeros:
            print(f"  {fname:<22}: NOT FOUND — using zeros")
            return np.zeros_like(X_joint)
        return None

    X_bone    = _load("X_bones.npy")
    X_motion  = _load("X_motion.npy")
    X_bm      = _load("X_bone_motion.npy")

    u, c = np.unique(y, return_counts=True)
    print(f"\n  X shape   : {X_joint.shape}")
    print(f"  Classes   : {len(u)} | min: {c.min()} | max: {c.max()} | mean: {c.mean():.1f}")
    return X_joint, X_bone, X_motion, X_bm, y


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _eval_loop(model, loader, forward_fn, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, labels = [b.to(device) for b in batch]
            logits = forward_fn(model, inputs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def train_one_fold(
    model, train_loader, val_loader,
    forward_fn, num_classes, epochs, fold, device,
):
    """
    Generic fold training with warmup-cosine LR, SWA, early stopping.
    Returns (best_val_acc, best_state_dict, history_dict).
    """
    cw = compute_class_weights(
        np.array([b[-1].item() for b in train_loader.dataset]),
        num_classes, device)
    criterion  = LabelSmoothingLoss(num_classes, LABEL_SMOOTH, cw)
    optimizer  = torch.optim.AdamW(model.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(ep):
        w = 10
        if ep < w: return ep / w
        return 0.5 * (1 + np.cos(np.pi * (ep - w) / (epochs - w)))

    scheduler     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    swa_updates   = 0

    best_val_acc  = 0.0
    best_state    = None
    patience_ctr  = 0
    stopped_epoch = epochs
    history       = {"train_acc": [], "val_acc": []}

    print(f"\n── Fold {fold} ──────────────────────────────────────────────")
    n_tr = len(train_loader.dataset)
    n_vl = len(val_loader.dataset)
    print(f"   Train: {n_tr}  Val: {n_vl}  Patience: {PATIENCE}  SWA from ep {SWA_START_EP}")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_correct = 0
        for batch in train_loader:
            *inputs, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = forward_fn(model, inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_correct += (logits.argmax(1) == labels).sum().item()
        tr_acc = tr_correct / n_tr

        # ── LR schedule ───────────────────────────────────────────────────────
        if epoch >= SWA_START_EP:
            swa_model.update_parameters(model)
            swa_updates += 1
            swa_scheduler.step()
            if swa_updates == 1:
                print(f"  [SWA accumulation started at ep {epoch}]")
        else:
            scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        val_preds, val_labels = _eval_loop(model, val_loader, forward_fn, device)
        v_acc = (val_preds == val_labels).mean()

        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(v_acc))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            marker = "  <- best"
        else:
            patience_ctr += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | Train: {tr_acc:.3f} | "
                  f"Val: {v_acc:.3f}{marker} | Gap: {tr_acc - v_acc:.3f} | "
                  f"Patience: {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            stopped_epoch = epoch
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── SWA post-training ──────────────────────────────────────────────────────
    if swa_updates >= 5:
        print(f"  Calibrating SWA BN ({swa_updates} snapshots)...")
        update_bn(train_loader, swa_model, device=device)
        swa_preds, swa_labels = _eval_loop(swa_model, val_loader, forward_fn, device)
        v_swa = (swa_preds == swa_labels).mean()
        print(f"  SWA val: {v_swa:.3f}  base best: {best_val_acc:.3f}")
        if v_swa > best_val_acc:
            best_val_acc = v_swa
            best_state = {
                (k[len("module."):] if k.startswith("module.") else k): v.cpu().clone()
                for k, v in swa_model.state_dict().items()
            }
            print(f"  Using SWA weights")

    print(f"  Best val: {best_val_acc:.3f} (stopped ep {stopped_epoch})")
    history["best_val_acc"]  = float(best_val_acc)
    history["stopped_epoch"] = stopped_epoch
    return best_val_acc, best_state, history


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RUNNERS  (one per model type)
# Note: BiLSTM and single-stream ST-GCN runners have been removed.

def run_multi_stream_stgcn(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    """3-stream ST-GCN (joint + bone + motion)."""
    num_classes = len(np.unique(y))

    def make_model():
        return STGCN(
            num_classes = num_classes,
            in_channels = 2,
            num_joints  = NUM_JOINTS,
            dropout     = DROPOUT,
        ).to(device)

    def forward_fn(model, inputs):
        xj, xb, xm = inputs
        return model({"joint": xj, "bone": xb, "motion": xm})

    return _run_kfold(
        make_model, forward_fn,
        ThreeStreamDataset, (X_joint, X_bone, X_motion), y,
        num_classes, epochs, device, label_map,
    )


def run_2stream_stgcn(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    """Original 2-stream ported ST-GCN (joint + bone, late fusion)."""
    num_classes = len(np.unique(y))

    def make_model():
        return TwoStreamPortedSTGCN(
            in_channels = 2,
            num_class   = num_classes,
            graph_args  = GRAPH_ARGS,
            adaptive_graph   = True,
            drop_graph_prob  = 0.1,
            dropout          = DROPOUT,
        ).to(device)

    def forward_fn(model, inputs):
        xj, xb = inputs
        return model(xj, bone=xb)

    return _run_kfold(
        make_model, forward_fn,
        TwoStreamDataset, (X_joint, X_bone), y,
        num_classes, epochs, device, label_map,
    )


def run_4stream_fusion(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    """4-stream early fusion ST-GCN (current best)."""
    num_classes = len(np.unique(y))

    def make_model():
        return FourStreamSTGCN(
            2, num_classes, GRAPH_ARGS,
            edge_importance_weighting = True,
            adaptive_graph            = True,
            drop_graph_prob           = 0.1,
            early_fusion              = True,
            dropout                   = DROPOUT,
        ).to(device)

    def forward_fn(model, inputs):
        xj, xm, xb, xbm = inputs
        return model(xj, motion=xm, bone=xb, bone_motion=xbm)

    return _run_kfold(
        make_model, forward_fn,
        FourStreamDataset, (X_joint, X_motion, X_bone, X_bm), y,
        num_classes, epochs, device, label_map,
    )


def run_4stream_late_fusion(X_joint, X_bone, X_motion, X_bm, y, epochs, device, label_map):
    """4-stream late fusion ST-GCN."""
    num_classes = len(np.unique(y))

    def make_model():
        return FourStreamSTGCN(
            2, num_classes, GRAPH_ARGS,
            edge_importance_weighting = True,
            adaptive_graph            = True,
            drop_graph_prob           = 0.1,
            early_fusion              = False,
            dropout                   = DROPOUT,
        ).to(device)

    def forward_fn(model, inputs):
        xj, xm, xb, xbm = inputs
        return model(xj, motion=xm, bone=xb, bone_motion=xbm)

    return _run_kfold(
        make_model, forward_fn,
        FourStreamDataset, (X_joint, X_motion, X_bone, X_bm), y,
        num_classes, epochs, device, label_map,
    )


# ══════════════════════════════════════════════════════════════════════════════
# K-FOLD RUNNER  (shared by all models)
# ══════════════════════════════════════════════════════════════════════════════

def _run_kfold(make_model_fn, forward_fn, dataset_cls, X_arrays, y,
               num_classes, epochs, device, label_map):
    """
    Returns a results dict:
      fold_accs, cv_mean, cv_std, test_acc,
      all_preds (on held-out test), all_labels, fold_histories
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    idx_all = np.arange(len(y))
    idx_tv, idx_test = train_test_split(idx_all, test_size=TEST_SPLIT,
                                        random_state=SEED, stratify=y)

    # Normalize each stream independently (train stats)
    def norm_arrays(*idxs):
        """Returns normalized versions of each X for given index sets."""
        results = []
        norms   = []
        for X in X_arrays:
            mean = X[idx_tv].mean(axis=(0, 1), keepdims=True)
            std  = X[idx_tv].std(axis=(0, 1),  keepdims=True)
            std  = np.where(std < 1e-6, 1.0, std)
            norms.append((mean, std))
            results.append(tuple((X[idx] - mean) / std for idx in idxs))
        # Return per-stream tuples: results[i] = (tr_norm, val_norm, test_norm)
        return results

    test_arrays  = [X[idx_test] for X in X_arrays]
    tv_arrays    = [X[idx_tv]   for X in X_arrays]
    y_test       = y[idx_test]
    y_tv         = y[idx_tv]

    skf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    fold_histories = []
    cv_preds_all = []
    cv_labels_all = []
    best_overall_acc = 0.0
    best_state_global = None
    best_test_arrays_norm = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(tv_arrays[0], y_tv), 1):
        # Normalize with fold's train split stats
        norm_groups = norm_arrays(tr_idx, val_idx, np.arange(len(idx_test)))
        # norm_groups[i] = (tr_norm, val_norm, test_norm) for stream i

        tr_norm   = [g[0] for g in norm_groups]
        val_norm  = [g[1] for g in norm_groups]
        test_norm = [g[2] for g in norm_groups]

        tr_ds  = dataset_cls(*tr_norm,  y_tv[tr_idx],  augment=True)
        val_ds = dataset_cls(*val_norm, y_tv[val_idx], augment=False)
        tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = make_model_fn()
        val_acc, best_state, history = train_one_fold(
            model, tr_loader, val_loader,
            forward_fn, num_classes, epochs, fold, device)

        fold_accs.append(val_acc)
        fold_histories.append(history)
        print(f"  Fold {fold} best val acc: {val_acc:.3f}")

        # Collect CV predictions for per-class analysis
        val_preds, val_labels = _eval_loop(model, val_loader, forward_fn, device)
        cv_preds_all.extend(val_preds.tolist())
        cv_labels_all.extend(val_labels.tolist())

        if val_acc > best_overall_acc:
            best_overall_acc  = val_acc
            best_state_global = best_state
            best_test_arrays_norm = test_norm

    # ── Test evaluation with best fold's weights ───────────────────────────────
    test_model = make_model_fn()
    test_model.load_state_dict(best_state_global)
    test_ds  = dataset_cls(*best_test_arrays_norm, y_test, augment=False)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds, all_labels = _eval_loop(test_model, test_ld, forward_fn, device)
    test_acc = (all_preds == all_labels).mean()

    cv_mean = float(np.mean(fold_accs))
    cv_std  = float(np.std(fold_accs))

    print(f"\n── K-Fold Results ──────────────────────────────────────")
    for i, a in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {a:.3f}")
    print(f"   Mean  : {cv_mean:.3f} +/- {cv_std:.3f}")
    print(f"── Final Test Accuracy ─────────────────────────────────")
    print(f"   Test  : {test_acc:.3f}  ({int(test_acc*len(y_test))}/{len(y_test)} correct)")

    return {
        "fold_accs":       [float(a) for a in fold_accs],
        "cv_mean":         cv_mean,
        "cv_std":          cv_std,
        "test_acc":        float(test_acc),
        "all_preds":       all_preds.tolist(),
        "all_labels":      all_labels.tolist(),
        "cv_preds":        cv_preds_all,
        "cv_labels":       cv_labels_all,
        "num_classes":     num_classes,
        "fold_histories":  fold_histories,
        "label_map":       label_map,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "multi-stream-stgcn": "#55A868",
    "2stream-stgcn":      "#C44E52",
    "4stream-fusion":     "#8172B2",
    "4stream-late-fusion":"#D55E00",
}

FULL_NAMES = {
    "multi-stream-stgcn": "Multi-Stream ST-GCN\n(3-Stream)",
    "2stream-stgcn":      "Original 2-Stream\nST-GCN (Ported)",
    "4stream-fusion":     "4-Stream Early\nFusion (Current)",
    "4stream-late-fusion":"4-Stream Late\nFusion",
}


def plot_model_results(key, results, save_path):
    """Generate a 4-panel per-model results figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pub     = PUBLISHED[key]
    color   = COLORS[key]
    n_cls   = results["num_classes"]
    fold_accs = results["fold_accs"]
    histories = results["fold_histories"]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Fold accuracy bar chart ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    bars = ax1.bar([f"Fold {i+1}" for i in range(len(fold_accs))],
                   fold_accs, color=color, alpha=0.85, width=0.55, zorder=3)
    ax1.axhline(results["cv_mean"], color=color, ls="--", lw=1.5,
                label=f"CV Mean {results['cv_mean']:.3f}")
    ax1.axhline(results["test_acc"], color="#333333", ls=":", lw=1.5,
                label=f"Test Acc {results['test_acc']:.3f}")
    if pub["cv"] is not None:
        ax1.axhline(pub["cv"], color=color, ls="-.", lw=1.2, alpha=0.5,
                    label=f"Published CV {pub['cv']:.3f}")
    if pub["test"] is not None:
        ax1.axhline(pub["test"], color="#333333", ls="-.", lw=1.2, alpha=0.5,
                    label=f"Published Test {pub['test']:.3f}")
    for bar, acc in zip(bars, fold_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, min(1.0, max(fold_accs) * 1.25 + 0.05))
    ax1.set_ylabel("Accuracy")
    ax1.set_title("K-Fold Validation Accuracy", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel 2: Training curves (all folds overlaid) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    for i, h in enumerate(histories):
        ep = range(1, len(h["train_acc"]) + 1)
        ax2.plot(ep, h["train_acc"], color=color, alpha=0.3, lw=1)
        ax2.plot(ep, h["val_acc"],   color=color, alpha=0.7, lw=1.5,
                 label=f"Fold {i+1} val" if i == 0 else "_")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Curves (all folds)\n─ val   ╌ train", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 1.0)

    # ── Panel 3: Confusion matrix ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    all_preds  = np.array(results["all_preds"])
    all_labels = np.array(results["all_labels"])
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    # Only show ticks if not too many classes
    if n_cls <= 25:
        label_map = results.get("label_map", {})
        idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
        tick_labels = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
        ax3.set_xticks(range(n_cls))
        ax3.set_yticks(range(n_cls))
        ax3.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax3.set_yticklabels(tick_labels, fontsize=6)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_title(f"Confusion Matrix (Test Set, n={len(all_labels)})", fontweight="bold")

    # ── Panel 4: Per-class accuracy bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#FFFFFF")
    # Prefer CV data for per-class accuracy (more samples per class)
    per_src_preds = np.array(results.get("cv_preds", all_preds))
    per_src_labels = np.array(results.get("cv_labels", all_labels))

    label_map = results.get("label_map", {})
    idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
    counts = np.bincount(per_src_labels, minlength=n_cls)
    present = counts > 0
    per_class_acc = np.full(n_cls, np.nan, dtype=np.float32)
    for cls_id in range(n_cls):
        if present[cls_id]:
            mask = per_src_labels == cls_id
            per_class_acc[cls_id] = (per_src_preds[mask] == cls_id).mean()
    cls_labels = [idx_to_label.get(i, str(i)) for i in range(n_cls)]
    mean_acc = float(np.nanmean(per_class_acc)) if np.any(present) else 0.0
    plot_vals = np.where(np.isnan(per_class_acc), 0.0, per_class_acc)
    bar_colors = [color if (not np.isnan(a) and a >= mean_acc) else "#CCCCCC"
                  for a in per_class_acc]
    ax4.bar(range(n_cls), plot_vals, color=bar_colors, alpha=0.85)
    missing_count = int((~present).sum())
    ax4.axhline(mean_acc, color="#333333", ls="--", lw=1.5,
                label=f"Mean {mean_acc:.3f} (present only)")
    if missing_count:
        ax4.text(0.99, 0.97, f"Missing classes: {missing_count}",
                 transform=ax4.transAxes, ha="right", va="top", fontsize=8,
                 color="#444444")
    ax4.set_xticks(range(n_cls))
    ax4.set_xticklabels(cls_labels, rotation=90, fontsize=6 if n_cls > 30 else 8)
    ax4.set_ylabel("Accuracy")
    src_name = "CV" if "cv_preds" in results else "Test"
    ax4.set_title(f"Per-Class Accuracy ({src_name})", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.grid(axis="y", alpha=0.25)

    # Add a compact summary of non-zero classes to make small signals visible
    nonzero = [(i, float(per_class_acc[i]), int(counts[i]))
               for i in range(n_cls)
               if present[i] and not np.isnan(per_class_acc[i]) and per_class_acc[i] > 0]
    if nonzero:
        nonzero.sort(key=lambda x: (-x[1], -x[2]))
        top = nonzero[:10]
        lines = [
            "Nonzero classes (top 10):",
            *[
                f"{idx_to_label.get(i, i)}: {acc:.2f} ({cnt})"
                for i, acc, cnt in top
            ],
        ]
        ax4.text(0.99, 0.55, "\n".join(lines), transform=ax4.transAxes,
                 ha="right", va="top", fontsize=7, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", alpha=0.7))

    # ── Panel 5: Summary stats box ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#FFFFFF")
    ax5.axis("off")
    lines = [
        ("Model", FULL_NAMES[key].replace("\n", " ")),
        ("", ""),
        ("K-Fold (K=4)", ""),
        ("  CV Mean",   f"{results['cv_mean']:.4f}"),
        ("  CV Std",    f"± {results['cv_std']:.4f}"),
        ("  Best Fold", f"{max(fold_accs):.4f}"),
        ("  Worst Fold",f"{min(fold_accs):.4f}"),
        ("", ""),
        ("Final Results", ""),
        ("  Test Acc",  f"{results['test_acc']:.4f}"),
        ("  Test N",    f"{len(all_labels)}"),
        ("  Classes",   f"{n_cls}"),
        ("", ""),
        ("Published Results", ""),
        ("  Test Acc",  f"{pub['test']:.3f}" if pub['test'] else "—"),
        ("  CV Mean",   f"{pub['cv']:.3f}"   if pub['cv']   else "—"),
    ]
    y_pos = 0.97
    for label, val in lines:
        if label == "" and val == "":
            y_pos -= 0.025
            continue
        is_header = (val == "" and label != "")
        weight = "bold" if is_header else "normal"
        size   = 10    if is_header else 9
        ax5.text(0.02, y_pos, label, transform=ax5.transAxes,
                 fontsize=size, fontweight=weight, va="top",
                 color="#333333" if not is_header else "#000000")
        if val:
            ax5.text(0.62, y_pos, val, transform=ax5.transAxes,
                     fontsize=9, va="top", color="#000000")
        y_pos -= 0.055

    # ── Figure title ───────────────────────────────────────────────────────────
    fig.suptitle(f"Model Results: {FULL_NAMES[key].replace(chr(10), ' ')}",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved chart → {save_path}")


def plot_comparison_overview(all_results, save_path):
    """Generate the 5-model comparison overview figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys      = [k for k in MODEL_ORDER if k in all_results]
    names     = [FULL_NAMES[k].replace("\n", " ") for k in keys]
    colors    = [COLORS[k] for k in keys]

    run_test  = [all_results[k]["test_acc"]  for k in keys]
    run_cv    = [all_results[k]["cv_mean"]   for k in keys]
    run_cv_std= [all_results[k]["cv_std"]    for k in keys]
    pub_test  = [PUBLISHED[k]["test"] or 0   for k in keys]
    pub_cv    = [PUBLISHED[k]["cv"]   or 0   for k in keys]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    x    = np.arange(len(keys))
    w    = 0.32

    # ── Panel 1: Test Accuracy — Run vs Published ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    b1 = ax1.bar(x - w/2, run_test,  w, color=colors, alpha=0.85, label="This Run")
    b2 = ax1.bar(x + w/2, pub_test,  w, color=colors, alpha=0.40, label="Published")
    for bar, val in list(zip(b1, run_test)) + list(zip(b2, pub_test)):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Test Accuracy: This Run vs Published Results", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, min(1.0, max(run_test + pub_test) * 1.2 + 0.05))
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: CV Mean with error bars ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    valid_cv = [(i, k) for i, k in enumerate(keys) if all_results[k]["cv_mean"] > 0]
    if valid_cv:
        vi, vk = zip(*valid_cv)
        vcv    = [all_results[k]["cv_mean"] for k in vk]
        vstd   = [all_results[k]["cv_std"]  for k in vk]
        vcolors= [COLORS[k] for k in vk]
        ax2.bar(vi, vcv, color=vcolors, alpha=0.85, width=0.55)
        ax2.errorbar(vi, vcv, yerr=vstd, fmt="none", color="#333333", capsize=5, lw=1.5)
        pub_cvs = [PUBLISHED[k]["cv"] for k in vk if PUBLISHED[k]["cv"] is not None]
        pub_vi  = [i for i, k in zip(vi, vk) if PUBLISHED[k]["cv"] is not None]
        if pub_cvs:
            ax2.scatter(pub_vi, pub_cvs, marker="D", color="#333333", zorder=5, s=50,
                        label="Published CV")
        for i, (cv, std) in zip(vi, zip(vcv, vstd)):
            ax2.text(i, cv + std + 0.005, f"{cv:.3f}±{std:.3f}",
                     ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(names, fontsize=8, rotation=10)
    ax2.set_ylabel("CV Mean Accuracy")
    ax2.set_title("Cross-Validation Mean ± Std", fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Model evolution timeline ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(range(len(keys)), run_test,  "o-", color="#2196F3", lw=2, ms=8, label="Test (this run)")
    ax3.plot(range(len(keys)), pub_test,  "s--",color="#2196F3", lw=1.5, ms=8, alpha=0.5, label="Test (published)")
    ax3.plot(range(len(keys)), run_cv,    "o-", color="#4CAF50", lw=2, ms=8, label="CV (this run)")
    if any(p for p in pub_cv):
        ax3.plot(range(len(keys)), pub_cv, "s--",color="#4CAF50", lw=1.5, ms=8, alpha=0.5, label="CV (published)")
    ax3.fill_between(range(len(keys)),
                     [c - s for c, s in zip(run_cv, run_cv_std)],
                     [c + s for c, s in zip(run_cv, run_cv_std)],
                     alpha=0.15, color="#4CAF50")
    ax3.set_xticks(range(len(keys)))
    ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Model Evolution Timeline", fontweight="bold")
    ax3.set_ylim(0, 1.0)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    # ── Panel 4: Fold-level violin / box plot ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    fold_data  = [all_results[k]["fold_accs"] for k in keys]
    bp = ax4.boxplot(fold_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set(color="#555555", lw=1)
    for cap in bp["caps"]:
        cap.set(color="#555555", lw=1)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="#555555", markersize=4)
    ax4.set_xticks(range(1, len(keys) + 1))
    ax4.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax4.set_ylabel("Fold Accuracy")
    ax4.set_title("Fold Accuracy Distribution", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # ── Panel 5: Summary table ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF")
    ax5.axis("off")
    table_data = [
        ["Model", "Test\n(run)", "Test\n(pub)", "CV Mean\n(run)", "CV\n(pub)", "Δ Test"]
    ]
    for k in keys:
        r     = all_results[k]
        p     = PUBLISHED[k]
        delta = r["test_acc"] - (p["test"] or 0)
        table_data.append([
            FULL_NAMES[k].replace("\n", " "),
            f"{r['test_acc']:.3f}",
            f"{p['test']:.3f}" if p["test"] else "—",
            f"{r['cv_mean']:.3f}±{r['cv_std']:.3f}",
            f"{p['cv']:.3f}"   if p["cv"]   else "—",
            f"{delta:+.3f}",
        ])
    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF4FF")
        else:
            cell.set_facecolor("#FFFFFF")
    ax5.set_title("Results Summary Table", fontweight="bold", pad=12)

    fig.suptitle("ASL Sign Language Recognition — 5-Model Comparison Overview",
                 fontsize=15, fontweight="bold", y=0.99)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved comparison chart → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_results_cache():
    if os.path.exists(RESULTS_CACHE):
        with open(RESULTS_CACHE) as f:
            return json.load(f)
    return {}


def save_results_cache(cache):
    with open(RESULTS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Results cached → {RESULTS_CACHE}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

MODEL_RUNNERS = {
    "multi-stream-stgcn": run_multi_stream_stgcn,
    "2stream-stgcn":      run_2stream_stgcn,
    "4stream-fusion":     run_4stream_fusion,
    "4stream-late-fusion":run_4stream_late_fusion,
}


def run_model(key, data, epochs, device, label_map, cache, force=False):
    """Run (or load from cache) a single model and generate its chart."""
    if key in cache and not force:
        print(f"\n  [{key}] Loaded from cache (use --force to retrain)")
        results = cache[key]
    else:
        print(f"\n{'='*65}")
        print(f"  Running: {FULL_NAMES[key]}")
        print(f"{'='*65}")
        X_joint, X_bone, X_motion, X_bm, y = data
        t0 = time.time()
        results = MODEL_RUNNERS[key](X_joint, X_bone, X_motion, X_bm, y,
                                     epochs, device, label_map)
        elapsed = time.time() - t0
        results["elapsed_sec"] = elapsed
        print(f"\n  Elapsed: {elapsed/60:.1f} min")
        cache[key] = results
        save_results_cache(cache)

    chart_path = os.path.join(CHARTS_DIR, f"{key.replace('-','_')}_results.png")
    plot_model_results(key, results, chart_path)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ASL Sign Language Recognition — 5-Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --bilstm
  python main.py --custom-stgcn
  python main.py --multi-stream-stgcn
  python main.py --2stream-stgcn
  python main.py --4stream-fusion
  python main.py --compare-5
  python main.py --compare-5 --results-only
  python main.py --compare-5 --epochs 150 --force
        """,
    )

    # ── Model selection ────────────────────────────────────────────────────────
    parser.add_argument("--multi-stream-stgcn",  action="store_true",
                        help="Run Multi-Stream ST-GCN (3-stream: joint+bone+motion)")
    # NOTE: --2stream-stgcn / --4stream-fusion are rewritten to these before argparse
    parser.add_argument("--xstream-stgcn-2",    action="store_true",
                        help="Run Original 2-Stream ST-GCN (Ported, late fusion)  [alias: --2stream-stgcn]")
    parser.add_argument("--xstream-fusion-4",   action="store_true",
                        help="Run 4-Stream Early Fusion ST-GCN (Current best)     [alias: --4stream-fusion]")
    parser.add_argument("--xstream-late-fusion-4", action="store_true",
                        help="Run 4-Stream Late Fusion ST-GCN                     [alias: --4stream-late-fusion]")
    parser.add_argument("--compare-5",           action="store_true",
                        help="Run all models and produce comparison charts")

    # ── Options ───────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,  default=300,
                        help="Max training epochs per fold (default: 300)")
    parser.add_argument("--results-only", action="store_true",
                        help="Only produce charts from cached results, skip training")
    parser.add_argument("--force",        action="store_true",
                        help="Retrain even if cached results exist")
    parser.add_argument("--device",       type=str,  default=None,
                        help="Force device: cuda or cpu")
    parser.add_argument("--no-charts",    action="store_true",
                        help="Skip chart generation (useful for headless runs)")

    args = parser.parse_args()

    # ── Determine device ───────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE
    print(f"\n  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")

    # ── Determine which models to run ──────────────────────────────────────────
    selected = []
    if args.compare_5:
        selected = MODEL_ORDER[:]
    else:
        if args.multi_stream_stgcn: selected.append("multi-stream-stgcn")
        if args.xstream_stgcn_2:    selected.append("2stream-stgcn")
        if args.xstream_fusion_4:   selected.append("4stream-fusion")
        if args.xstream_late_fusion_4: selected.append("4stream-late-fusion")

    if not selected:
        parser.print_help()
        print("\nNo model selected. Use --multi-stream-stgcn, --compare-5, etc.")
        sys.exit(0)

    print(f"  Models  : {selected}")

    # ── Load cache ─────────────────────────────────────────────────────────────
    cache = load_results_cache()

    # ── Load data (only if training needed) ────────────────────────────────────
    needs_training = [k for k in selected if k not in cache or args.force]
    data      = None
    label_map = {}

    if needs_training and not args.results_only:
        print("\nLoading data streams...")
        X_joint, X_bone, X_motion, X_bm, y = load_all_streams()
        data = (X_joint, X_bone, X_motion, X_bm, y)
        label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                label_map = json.load(f)
    elif args.results_only and not cache:
        print("No cached results found and --results-only set. Run a model first.")
        sys.exit(1)

    # ── Run selected models ────────────────────────────────────────────────────
    all_results = {}
    for key in selected:
        if args.results_only:
            if key not in cache:
                print(f"  [{key}] No cached result — skipping (train without --results-only first)")
                continue
            results = cache[key]
            print(f"  [{key}] Loaded from cache")
        else:
            results = run_model(key, data, args.epochs, device, label_map, cache,
                                force=args.force)
        all_results[key] = results

    if not all_results:
        print("No results to chart.")
        sys.exit(0)

    # ── Individual charts ──────────────────────────────────────────────────────
    if not args.no_charts:
        print("\nGenerating individual model charts...")
        for key, results in all_results.items():
            chart_path = os.path.join(CHARTS_DIR, f"{key.replace('-','_')}_results.png")
            plot_model_results(key, results, chart_path)

        # ── Overview comparison chart (only when >1 model) ─────────────────────
        if len(all_results) > 1:
            print("\nGenerating comparison overview chart...")
            overview_path = os.path.join(CHARTS_DIR, "comparison_overview.png")
            plot_comparison_overview(all_results, overview_path)

    # ── Print final summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Test Acc':>9} {'CV Mean':>9} {'CV Std':>8}")
    print(f"  {'-'*65}")
    for key in MODEL_ORDER:
        if key not in all_results:
            continue
        r = all_results[key]
        name = FULL_NAMES[key].replace("\n", " ")
        print(f"  {name:<35} {r['test_acc']:>9.3f} {r['cv_mean']:>9.3f} {r['cv_std']:>8.3f}")

    if not args.no_charts:
        print(f"\n  Charts saved to: {CHARTS_DIR}/")


if __name__ == "__main__":
    # Rewrite numeric-prefixed flags before argparse sees them
    _raw_argv = sys.argv[1:]
    sys.argv = [sys.argv[0]] + [
        "--xstream-stgcn-2"    if a == "--2stream-stgcn"   else
        "--xstream-fusion-4"   if a == "--4stream-fusion"  else
        "--xstream-late-fusion-4" if a == "--4stream-late-fusion" else a
        for a in _raw_argv
    ]
    main()