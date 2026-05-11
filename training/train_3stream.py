"""
train_3stream.py — Standalone Training Script for 3-Stream ST-GCN
================================================================

This is the dedicated training script for the ThreeStreamSTGCN model.

Three input streams fed simultaneously:
    joint      — normalized (x,y) joint positions  (N, 2, T, 51)
    motion     — frame-to-frame joint delta         (N, 2, T, 51)
    bone       — child-minus-parent bone vectors    (N, 2, T, 51)

Outputs:
    - models/sign_stgcn_3stream.pth
    - output/results_3stream.json


SWA note: Accumulate weights throughout training, do ONE update_bn at the
end, evaluate SWA model once after early stopping fires. No per-epoch BN resets.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.st_gcn_twostream import Model as ThreeStreamSTGCN

OUTPUT_DIR    = os.path.join(ROOT, "output")
MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn_3stream.pth")

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
    """Cross-entropy with label smoothing and optional per-class weighting.

    Label smoothing prevents overconfidence by distributing probability mass:
        smoothed_target[y]   = 1.0 - smoothing
        smoothed_target[k≠y] = smoothing / (K - 1)

    Per-class weights compensate for class imbalance: rare classes receive
    higher loss weight so the model pays equal attention to all classes.

    Args:
        num_classes (int)  : Total number of output classes K.
        smoothing   (float): Label smoothing factor ε. Default: 0.1.
        weight      (Tensor|None): Per-class weights of shape (K,).
    """
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, pred, target):
        """Compute smoothed cross-entropy loss.

        Args:
            pred   (Tensor): Logits, shape (N, K).
            target (Tensor): Ground-truth labels, shape (N,), dtype long.

        Returns:
            Tensor: Scalar loss.
        """
        log_prob = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            # Build smoothed target distribution
            smooth = torch.zeros_like(log_prob)
            smooth.fill_(self.smoothing / (self.num_classes - 1))  # background mass
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)  # true class mass
        if self.weight is not None:
            # Scale each sample's loss by the per-class weight of its true label
            w = self.weight[target].unsqueeze(1)
            return -(smooth * log_prob * w).sum(dim=1).mean()
        return -(smooth * log_prob).sum(dim=1).mean()


def compute_class_weights(y, num_classes, device):
    """Compute inverse-frequency class weights for imbalanced datasets.

    Weight for class k = 1/count_k, then normalized so weights sum to num_classes.
    Classes with zero samples get weight 1 (neutral) to avoid division by zero.

    Args:
        y           (np.ndarray): Integer label array.
        num_classes (int)       : Total number of classes.
        device      (torch.device): Target device for the weight tensor.

    Returns:
        Tensor: Per-class weights, shape (num_classes,), float32.
    """
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)   # avoid div-by-zero for empty classes
    weights = 1.0 / counts                        # inverse frequency
    weights = weights / weights.sum() * num_classes  # normalize so mean weight = 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


def augment_streams(xj, xm, xb):
    T, F = xj.shape
    V = F // 2

    if np.random.rand() < 0.6:
        xj = xj + np.random.randn(*xj.shape).astype(np.float32) * 0.015
        xm = xm + np.random.randn(*xm.shape).astype(np.float32) * 0.015
        xb = xb + np.random.randn(*xb.shape).astype(np.float32) * 0.015
        

    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        xj = xj * scale
        xm = xm * scale
        xb = xb * scale
        

    if np.random.rand() < 0.5:
        a = np.random.uniform(-20, 20) * np.pi / 180
        ca, sa = np.cos(a), np.sin(a)
        def apply_rot(x):
            xr = x.copy()
            for i in range(0, F, 2):
                xv, yv = x[:, i], x[:, i+1]
                xr[:, i] = ca*xv - sa*yv
                xr[:, i+1] = sa*xv + ca*yv
            return xr
        xj = apply_rot(xj)
        xm = apply_rot(xm)
        xb = apply_rot(xb)
        

    if np.random.rand() < 0.5:
        xj[:, 0::2] = -xj[:, 0::2]
        xm[:, 0::2] = -xm[:, 0::2]
        xb[:, 0::2] = -xb[:, 0::2]
        

    if np.random.rand() < 0.4:
        mask = np.random.rand(T) > 0.15
        if mask.sum() >= 2:
            idx = np.linspace(0, mask.sum()-1, T).astype(int)
            xj = xj[mask][idx]
            xm = xm[mask][idx]
            xb = xb[mask][idx]
            

    if np.random.rand() < 0.4:
        warp = np.cumsum(np.abs(np.random.randn(T)) + 0.5)
        idx = np.clip((warp/warp[-1]*(T-1)).astype(int), 0, T-1)
        xj = xj[idx]; xm = xm[idx]
        xb = xb[idx]; 

    if np.random.rand() < 0.4:
        cl = int(T * np.random.uniform(0.75, 1.0))
        st = np.random.randint(0, T - cl + 1)
        idx = np.linspace(0, cl-1, T).astype(int)
        xj = xj[st:st+cl][idx]
        xm = xm[st:st+cl][idx]
        xb = xb[st:st+cl][idx]
        

    if np.random.rand() < 0.35:
        angle_candidate = [-10.0, -5.0, 0.0, 5.0, 10.0]
        scale_candidate = [0.9, 1.0, 1.1]
        trans_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]
        move_time = 1
        node = np.arange(0, T, max(1, int(T / move_time))).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        Tx = np.random.choice(trans_candidate, num_node)
        Ty = np.random.choice(trans_candidate, num_node)

        a_seq = np.zeros(T); s_seq = np.zeros(T)
        tx_seq = np.zeros(T); ty_seq = np.zeros(T)

        for i in range(num_node - 1):
            st_n, ed_n = node[i], node[i + 1]
            a_seq[st_n:ed_n] = np.linspace(A[i], A[i + 1], ed_n - st_n) * np.pi / 180
            s_seq[st_n:ed_n] = np.linspace(S[i], S[i + 1], ed_n - st_n)
            tx_seq[st_n:ed_n] = np.linspace(Tx[i], Tx[i + 1], ed_n - st_n)
            ty_seq[st_n:ed_n] = np.linspace(Ty[i], Ty[i + 1], ed_n - st_n)

        xj2 = xj.reshape(T, V, 2)
        xm2 = xm.reshape(T, V, 2)
        xb2 = xb.reshape(T, V, 2)
        

        for i in range(T):
            ca, sa = np.cos(a_seq[i]) * s_seq[i], np.sin(a_seq[i]) * s_seq[i]
            rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

            xj2[i] = (rot @ xj2[i].T).T
            xj2[i, :, 0] += tx_seq[i]
            xj2[i, :, 1] += ty_seq[i]

            xm2[i] = (rot @ xm2[i].T).T
            xb2[i] = (rot @ xb2[i].T).T
            

        xj = xj2.reshape(T, F)
        xm = xm2.reshape(T, F)
        xb = xb2.reshape(T, F)
        

    return xj.astype(np.float32), xm.astype(np.float32), xb.astype(np.float32)


class ThreeStreamDataset(Dataset):
    def __init__(self, X_joint, X_motion, X_bone, y, augment=False):
        self.X_joint = X_joint.astype(np.float32)
        self.X_motion = X_motion.astype(np.float32)
        self.X_bone  = X_bone.astype(np.float32)
        
        self.y       = y.astype(np.int64)
        self.augment = augment

    def __len__(self): return len(self.y)

    def to_graph(self, x):
        """Convert flat skeleton (T, F) to ST-GCN graph format (2, T, V).

        F = V * 2 (each joint has x and y coordinate).
        The model expects channels (x, y) first, then time, then joints.

        Args:
            x (np.ndarray): Shape (T, F=102).

        Returns:
            np.ndarray: Shape (2, T, 51) — (channels, frames, joints).
        """
        T, F = x.shape
        V = F // 2   # 51 joints
        # Reshape: (T, F) → (T, V, 2) → transpose → (2, T, V)
        return x.reshape(T, V, 2).transpose(2, 0, 1)

    def __getitem__(self, idx):
        """Return one sample: four graph tensors + class label."""
        xj = self.X_joint[idx].copy()
        xmj = self.X_motion[idx].copy()
        xb = self.X_bone[idx].copy()
        
        if self.augment:
            xj, xmj, xb = augment_streams(xj, xmj, xb)
        # Convert all streams to (2, T, V) graph tensors
        return (torch.tensor(self.to_graph(xj),  dtype=torch.float32),
                torch.tensor(self.to_graph(xmj), dtype=torch.float32),
                torch.tensor(self.to_graph(xb),  dtype=torch.float32),
                torch.tensor(self.y[idx],         dtype=torch.long))


def global_normalize(X_tr, *X_others):
    """Z-score normalize using training-split mean and std.

    Computes mean and std from X_tr only (no leakage), then applies the
    same transformation to all other arrays (e.g. val, test splits).
    Features with std < 1e-6 are left unchanged (division by 1.0).

    Args:
        X_tr     (np.ndarray): Training split, shape (N_tr, T, F).
        *X_others             : Additional arrays to normalize with same stats.

    Returns:
        Tuple of normalized arrays in the same order as inputs.
    """
    mean = X_tr.mean(axis=(0, 1), keepdims=True)   # mean per feature (T-averaged)
    std  = X_tr.std(axis=(0, 1),  keepdims=True)    # std per feature
    std  = np.where(std < 1e-6, 1.0, std)           # avoid division by zero
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
    X_motion = ls("X_motion.npy",      "Joint motion")
    X_bm   = ls("X_bone_motion.npy", "Bone motion")
    u, c   = np.unique(y, return_counts=True)
    print(f"\n  X shape  : {X_joint.shape}")
    print(f"  Classes  : {len(u)} | min: {c.min()} | max: {c.max()} | mean: {c.mean():.1f}")
    return X_joint, X_motion, X_bone, X_bm, y


def eval_model(m, val_loader):
    m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xj, xmj, xb, yb in val_loader:
            xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))
            preds = m(xj, motion=xmj, bone=xb).argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    correct = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    return correct / len(val_loader.dataset), all_preds, all_labels


def train_fold(Xj_tr, Xm_tr, Xb_tr, y_tr,
               Xj_val, Xm_val, Xb_val, y_val,
               num_classes, fold):

    train_loader = DataLoader(
        ThreeStreamDataset(Xj_tr, Xm_tr, Xb_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(
        ThreeStreamDataset(Xj_val, Xm_val, Xb_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    model = ThreeStreamSTGCN(
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
    best_vp = []
    best_vl = []
    history = {'train_acc': [], 'val_acc': []}
    patience_ctr  = 0
    stopped_epoch = EPOCHS

    print(f"\n── Fold {fold} ─────────────────────────────────────────")
    print(f"   Train: {len(Xj_tr)}  Val: {len(Xj_val)}  "
          f"Patience: {PATIENCE}  SWA from ep {SWA_START_EP}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_correct = 0
        for xj, xmj, xb, yb in train_loader:
            xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))
            optimizer.zero_grad()
            out = model(xj, motion=xmj, bone=xb)
            criterion(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_correct += (out.argmax(1) == yb).sum().item()
        tr_acc = tr_correct / len(train_loader.dataset)
        history["train_acc"].append(tr_acc)

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
        v_acc, vp, vl = eval_model(model, val_loader)

        history["val_acc"].append(v_acc)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_vp = vp
            best_vl = vl
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
        v_swa, vp_swa, vl_swa = eval_model(swa_model, val_loader)
        print(f"  SWA val acc: {v_swa:.3f}  (base best: {best_val_acc:.3f})")
        if v_swa > best_val_acc:
            best_val_acc = v_swa
            best_vp = vp_swa
            best_vl = vl_swa
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
    return best_val_acc, best_state, best_vp, best_vl, history


def train():
    print("\nLoading data streams...")
    X_joint, X_motion, X_bone, X_bm, y = load_data()
    num_classes = len(np.unique(y))

    _m = ThreeStreamSTGCN(2, num_classes, GRAPH_ARGS,
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
    Xm_test  = X_motion[idx_test]
    y_test   = y[idx_test]
    Xj_tv    = X_joint[idx_tv];   Xb_tv    = X_bone[idx_tv]
    Xm_tv    = X_motion[idx_tv]
    y_tv     = y[idx_tv]

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), Xj_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\n  Train+Val: {len(y_tv)}  Test: {len(y_test)}")
    print(f"{'='*60}")

    skf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []
    best_acc  = 0.0
    best_state = None
    cv_preds = []
    cv_labels = []
    fold_histories = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xj_tv, y_tv), 1):
        Xj_tr_n,  Xj_val_n,  Xj_test_n  = global_normalize(
            Xj_tv[tr_idx],  Xj_tv[val_idx],  Xj_test)
        Xm_tr_n,  Xm_val_n,  Xm_test_n  = global_normalize(
            Xm_tv[tr_idx],  Xm_tv[val_idx],  Xm_test)
        Xb_tr_n,  Xb_val_n,  Xb_test_n  = global_normalize(
            Xb_tv[tr_idx],  Xb_tv[val_idx],  Xb_test)
        

        val_acc, state, vp, vl, hist = train_fold(
            Xj_tr_n, Xm_tr_n, Xb_tr_n, y_tv[tr_idx],
            Xj_val_n, Xm_val_n, Xb_val_n, y_tv[val_idx],
            num_classes, fold)
        fold_accs.append(val_acc)
        cv_preds.extend(vp)
        cv_labels.extend(vl)
        fold_histories.append(hist)
        print(f"  Fold {fold} best val acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc      = val_acc
            best_state    = state
            best_Xj_test  = Xj_test_n
            best_Xm_test  = Xm_test_n
            best_Xb_test  = Xb_test_n
            

    print(f"\n── K-Fold Results ──────────────────────────────────")
    for i, a in enumerate(fold_accs, 1):
        print(f"   Fold {i}: {a:.3f}")
    print(f"   Mean  : {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")

    model = ThreeStreamSTGCN(2, num_classes, GRAPH_ARGS,
        edge_importance_weighting=True, adaptive_graph=ADAPTIVE_GRAPH,
        drop_graph_prob=DROP_GRAPH_PROB, early_fusion=EARLY_FUSION,
        dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_SAVE)

    test_loader = DataLoader(
        ThreeStreamDataset(best_Xj_test, best_Xm_test, best_Xb_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for xj, xmj, xb, yb in test_loader:
            xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))
            preds = model(xj, motion=xmj, bone=xb).argmax(1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(yb.cpu().numpy().tolist())
    tc = sum(1 for p, y in zip(test_preds, test_labels) if p == y)

    print(f"\n── Final Test Set Evaluation ───────────────────────")
    print(f"   Test Accuracy : {tc/len(y_test):.3f}  ({tc}/{len(y_test)} correct)")
    print(f"   Best CV Acc   : {best_acc:.3f}")
    print(f"   CV Mean       : {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"───────────────────────────────────────────────────")
    print(f"   Model saved   : {MODEL_SAVE}")

    # Load label map
    lmap_path = os.path.join(OUTPUT_DIR, "label_map.json")
    if os.path.exists(lmap_path):
        with open(lmap_path, 'r') as f:
            lmap = json.load(f)
    else:
        lmap = {str(i): i for i in range(num_classes)}

    results = {
        "fold_accs": fold_accs,
        "cv_mean": float(np.mean(fold_accs)),
        "cv_std": float(np.std(fold_accs)),
        "test_acc": float(tc/len(y_test)),
        "all_preds": test_preds,
        "all_labels": test_labels,
        "cv_preds": cv_preds,
        "cv_labels": cv_labels,
        "num_classes": num_classes,
        "fold_histories": fold_histories,
        "label_map": lmap
    }
    
    out_json = os.path.join(OUTPUT_DIR, "results_3stream.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   JSON saved    : {out_json}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()