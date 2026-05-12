"""
evaluate.py — Model Evaluation and Single-Video Inference
==========================================================

Two modes of operation:

1. Dataset evaluation (default, no arguments):
   Loads the pre-processed dataset from output/, runs the selected model
   over all samples, and prints a full sklearn classification_report plus
   a confusion matrix.

2. Single-video inference (--video <path>):
   Takes one raw video file, runs the complete preprocessing pipeline
   (extract → clean → filter → normalize → smooth → resample → engineer streams),
   and prints the predicted class.

Usage:
    # Dataset eval (defaults to 4stream-early)
    python evaluation/evaluate.py
    
    # Dataset eval with specific model
    python evaluation/evaluate.py --model 3stream
    
    # Single video inference
    python evaluation/evaluate.py --video clip.mp4 --model 4stream-early
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ── Path setup: add project root so we can import sibling packages ─────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Model imports
from models.lstm              import SignLSTM
from models.stgcn_multi             import STGCN, NUM_JOINTS
from models.st_gcn_four_stream  import Model as FourStreamSTGCN
from models.st_gcn_four_stream  import Model as ThreeStreamSTGCN

# Preprocessing imports
from preprocessing.extract    import extract_mp, make_mp_detectors, download_mp_models
from preprocessing.normalize  import (normalize_skeleton, clean_missing_keypoints,
                                      filter_joints, normalize_dataset,
                                      smooth_dataset, compute_bone_vectors,
                                      compute_motion)
from preprocessing.resample   import temporal_resample

# ── Global settings ────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(ROOT, "output")
TARGET_LEN = 64   # frames per sequence (must match training config)


# ── Graph Helper ──────────────────────────────────────────────────────────────
def _to_graph(x):
    """Reshape (T, F) → (2, T, V) for ST-GCN."""
    T, F = x.shape
    V = F // 2
    return x.reshape(T, V, 2).transpose(2, 0, 1).astype(np.float32)


# ── Dataset Classes ───────────────────────────────────────────────────────────
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

class ThreeStreamDataset(Dataset):
    def __init__(self, Xj, Xb, Xm, y):
        self.Xj = Xj.astype(np.float32)
        self.Xb = Xb.astype(np.float32)
        self.Xm = Xm.astype(np.float32)
        self.y  = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(_to_graph(self.Xj[i])),
                torch.tensor(_to_graph(self.Xb[i])),
                torch.tensor(_to_graph(self.Xm[i])),
                torch.tensor(self.y[i]))

class FourStreamDataset(Dataset):
    def __init__(self, Xj, Xm, Xb, Xbm, y):
        self.Xj  = Xj.astype(np.float32)
        self.Xm  = Xm.astype(np.float32)
        self.Xb  = Xb.astype(np.float32)
        self.Xbm = Xbm.astype(np.float32)
        self.y   = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(_to_graph(self.Xj[i])),
                torch.tensor(_to_graph(self.Xm[i])),
                torch.tensor(_to_graph(self.Xb[i])),
                torch.tensor(_to_graph(self.Xbm[i])),
                torch.tensor(self.y[i]))


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(num_classes, model_type):
    """Instantiate the selected model and load pre-trained weights."""
    print(f"\nLoading model: {model_type}")
    if model_type == "lstm":
        model = SignLSTM(150, 128, 2, num_classes, 0.3, True).to(DEVICE)
        model_path = os.path.join(ROOT, "models", "sign_lstm.pth")
    elif model_type == "3stream":
        # Use the same three-stream wrapper used during training
        model = ThreeStreamSTGCN(2, num_classes, {"layout": "mediapipe_51", "strategy": "spatial"},
                                 edge_importance_weighting=True, adaptive_graph=True, early_fusion=True).to(DEVICE)
        model_path = os.path.join(ROOT, "models", "sign_stgcn_3stream.pth")
    elif model_type == "4stream-late":
        model = FourStreamSTGCN(2, num_classes, {"layout": "mediapipe_51", "strategy": "spatial"}, early_fusion=False).to(DEVICE)
        model_path = os.path.join(ROOT, "models", "sign_stgcn_4stream_late.pth")
    elif model_type == "4stream-early":
        model = FourStreamSTGCN(2, num_classes, {"layout": "mediapipe_51", "strategy": "spatial"}, early_fusion=True).to(DEVICE)
        model_path = os.path.join(ROOT, "models", "sign_stgcn_4stream_early.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE)
        # Support checkpoints saved as plain state_dict or wrapped dicts
        state_dict = None
        if isinstance(ckpt, dict):
            # common keys: 'state_dict', 'model_state', 'model'
            for k in ("state_dict", "model_state", "model"):
                if k in ckpt:
                    state_dict = ckpt[k]
                    break
        if state_dict is None and not isinstance(ckpt, dict):
            state_dict = ckpt
        if state_dict is None and isinstance(ckpt, dict):
            # maybe the checkpoint IS the state_dict
            state_dict = ckpt

        try:
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from {model_path}")
        except Exception as e:
            print(f"⚠️  Warning: strict loading failed: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded weights from {model_path} with strict=False (partial match)")
            except Exception as e2:
                print(f"⚠️  Warning: strict=False still failed: {e2}")
                # Fallback: drop mismatched classifier keys (class count mismatch)
                filtered = {}
                mismatched = []
                for k, v in state_dict.items():
                    if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        mismatched.append(k)
                if mismatched:
                    print("⚠️  Dropping mismatched keys:", mismatched)
                model.load_state_dict(filtered, strict=False)
                print("✅ Loaded compatible weights after dropping mismatched keys")
    else:
        print(f"⚠️  Warning: Weights not found at {model_path}. Using random weights.")
    
    model.eval()
    return model


# ── Single video prediction ───────────────────────────────────────────────────
def predict_video(video_path, model, model_type, label_map):
    """Run the full preprocessing + inference pipeline on a single video."""
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\nExtracting skeleton...")
    download_mp_models()
    pose_det, hand_det = make_mp_detectors()
    skeleton = extract_mp(video_path, pose_det, hand_det, max_frames=TARGET_LEN)
    pose_det.close()
    hand_det.close()

    print("Preprocessing and feature engineering...")
    skeleton = clean_missing_keypoints(skeleton)

    if model_type == "lstm":
        skeleton = normalize_skeleton(skeleton)
        skeleton = temporal_resample(skeleton, target_len=TARGET_LEN)
        tensor = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
    else:
        # ST-GCN pipeline
        skel_batch = np.expand_dims(skeleton, 0)
        skel_filt  = filter_joints(skel_batch)
        skel_norm  = normalize_dataset(skel_filt)
        skel_smooth = smooth_dataset(skel_norm)
        
        X_joint_resampled = temporal_resample(skel_smooth[0], target_len=TARGET_LEN)
        X_joint = np.expand_dims(X_joint_resampled, 0)
        
        X_bones  = compute_bone_vectors(X_joint)
        X_motion = compute_motion(X_joint)
        X_bm     = compute_motion(X_bones)
        
        t_joint  = torch.tensor(_to_graph(X_joint[0])).unsqueeze(0).to(DEVICE)
        t_bone   = torch.tensor(_to_graph(X_bones[0])).unsqueeze(0).to(DEVICE)
        t_motion = torch.tensor(_to_graph(X_motion[0])).unsqueeze(0).to(DEVICE)
        t_bm     = torch.tensor(_to_graph(X_bm[0])).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            if model_type == "3stream":
                inputs = {"joint": t_joint, "bone": t_bone, "motion": t_motion}
                output = model(inputs)
            else:
                output = model(t_joint, motion=t_motion, bone=t_bone, bone_motion=t_bm)
                
    pred_idx = output.argmax(dim=1).item()
    return idx_to_label.get(pred_idx, f"unknown({pred_idx})")


# ── Evaluate on saved dataset ─────────────────────────────────────────────────
def evaluate_dataset(model_type):
    """Load saved dataset and print classification report + confusion matrix."""
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        print("❌ label_map.json not found. Run extract.py first.")
        return

    with open(label_map_path) as f:
        label_map = json.load(f)

    num_classes  = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    model = load_model(num_classes, model_type)

    print("\nLoading dataset streams...")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    if not os.path.exists(y_path):
        print("❌ Dataset not found in output directory.")
        return
    y = np.load(y_path)

    if model_type == "lstm":
        # LSTM uses 150-d features
        for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                X = np.load(path)
                break
        else:
            print("❌ No processed data found.")
            return
        dataset = LSTMDataset(X, y)
        
        def forward_fn(m, batch):
            x, _ = batch
            return m(x.to(DEVICE))
            
    else:
        # ST-GCN uses 102-d features split into streams
        try:
            X_j  = np.load(os.path.join(OUTPUT_DIR, "X_normalized.npy"))
            X_b  = np.load(os.path.join(OUTPUT_DIR, "X_bones.npy"))
            X_m  = np.load(os.path.join(OUTPUT_DIR, "X_motion.npy"))
            X_bm = np.load(os.path.join(OUTPUT_DIR, "X_bone_motion.npy"))
        except FileNotFoundError as e:
            print(f"❌ Missing stream data: {e}. Run normalize.py first.")
            return
            
        # Apply global z-score normalization (required for ST-GCN models)
        def z_score(arr):
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std  = arr.std(axis=(0, 1), keepdims=True)
            std  = np.where(std < 1e-6, 1.0, std)
            return (arr - mean) / std

        X_j  = z_score(X_j)
        X_b  = z_score(X_b)
        X_m  = z_score(X_m)
        X_bm = z_score(X_bm)

        if model_type == "3stream":
            dataset = ThreeStreamDataset(X_j, X_b, X_m, y)
            def forward_fn(m, batch):
                xj, xb, xm, _ = batch
                return m(xj.to(DEVICE), motion=xm.to(DEVICE), bone=xb.to(DEVICE))
        else:
            dataset = FourStreamDataset(X_j, X_m, X_b, X_bm, y)
            def forward_fn(m, batch):
                xj, xm, xb, xbm, _ = batch
                return m(xj.to(DEVICE), motion=xm.to(DEVICE), bone=xb.to(DEVICE), bone_motion=xbm.to(DEVICE))

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("\nRunning inference...")
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            output = forward_fn(model, batch)
            preds  = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch[-1].numpy())

    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("── Confusion Matrix ───────────────────────────────────")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or run inference on sign language models")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video for inference")
    parser.add_argument("--model", type=str, default="all",
                        choices=["lstm", "3stream", "4stream-late", "4stream-early", "all"],
                        help="Model architecture to use (default: all)")
    args = parser.parse_args()

    if args.video:
        if args.model == "all":
            print("❌ Cannot use --model all for single video inference. Please specify one model.")
            sys.exit(1)
            
        label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
        with open(label_map_path) as f:
            label_map = json.load(f)

        model = load_model(len(label_map), args.model)
        result = predict_video(args.video, model, args.model, label_map)
        print(f"\n====================================")
        print(f" Predicted class: {result}")
        print(f"====================================")
    else:
        if args.model == "all":
            print("🚀 Running evaluation for all ST-GCN models...")
            evaluate_dataset("3stream")
            evaluate_dataset("4stream-late")
            evaluate_dataset("4stream-early")
        else:
            evaluate_dataset(args.model)