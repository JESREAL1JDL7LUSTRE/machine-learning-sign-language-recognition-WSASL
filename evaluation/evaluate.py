"""
evaluate.py — Model Evaluation and Single-Video Inference
==========================================================

Two modes of operation:

1. Dataset evaluation (default, no arguments):
   Loads the pre-processed dataset from output/, runs the trained SignLSTM
   (TCN+Attention) model over all samples, and prints a full sklearn
   classification_report plus a confusion matrix.

2. Single-video inference (--video <path>):
   Takes one raw video file, runs the complete preprocessing pipeline
   (extract → clean → normalize → resample), and prints the predicted class.

Usage:
    python evaluation/evaluate.py                        # dataset eval
    python evaluation/evaluate.py --video clip.mp4       # single video
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

# ── Path setup: add project root so we can import sibling packages ─────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Model and preprocessing imports
from models.lstm      import SignLSTM
from training.dataset import SignDataset
from preprocessing.extract    import extract_skeleton
from preprocessing.normalize  import normalize_skeleton, clean_missing_keypoints
from preprocessing.resample   import temporal_resample

# ── Global settings ────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT, "models", "sign_lstm.pth")   # trained TCN weights
OUTPUT_DIR = os.path.join(ROOT, "output")
TARGET_LEN = 64   # frames per sequence (must match training config)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(num_classes):
    """Instantiate SignLSTM and load pre-trained weights.

    Builds the same TCN+Attention architecture that was saved during training,
    loads the state dict from MODEL_PATH, and switches to eval mode (disables
    dropout, uses running BN stats).

    Args:
        num_classes (int): Number of sign classes (from label_map.json).

    Returns:
        SignLSTM: Model on DEVICE, ready for inference.
    """
    model = SignLSTM(
        input_size   = 150,          # raw MediaPipe feature size
        hidden_size  = 128,
        num_layers   = 2,
        num_classes  = num_classes,
        dropout      = 0.3,
        bidirectional= True          # API-compat flag (TCN is not bidirectional)
    ).to(DEVICE)

    # Load saved weights; map_location handles loading GPU weights on CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()   # disable dropout and switch BN to inference mode
    return model


# ── Single video prediction ───────────────────────────────────────────────────
def predict_video(video_path, model, label_map):
    """Run the full preprocessing + inference pipeline on a single video.

    Processes raw video → keypoints → normalization → fixed-length tensor
    → forward pass → predicted class name.

    Pipeline:
        extract_skeleton      — MediaPipe frame-by-frame keypoint extraction
        clean_missing_keypoints — per-joint temporal interpolation of zeros
        normalize_skeleton    — center (mid-shoulder) + scale (shoulder width)
                                + relative hand re-anchoring
        temporal_resample     — linear interpolation to TARGET_LEN frames
        unsqueeze(0)          — add batch dimension
        model.forward()       — TCN + Attention forward pass
        argmax                — pick highest-scoring class

    Args:
        video_path (str) : Path to raw video file (.mp4 / .avi / .mov).
        model            : Loaded SignLSTM model in eval mode.
        label_map  (dict): {class_name: class_index} mapping.

    Returns:
        str: Predicted class name, or "unknown(<idx>)" if idx not in map.
    """
    # Build reverse mapping: index → class name for human-readable output
    idx_to_label = {v: k for k, v in label_map.items()}

    # Step 1: Extract raw skeleton keypoints from the video
    skeleton = extract_skeleton(video_path, max_frames=TARGET_LEN)

    # Step 2: Fill missing (zero) keypoints via per-joint interpolation
    skeleton = clean_missing_keypoints(skeleton)

    # Step 3: Center, scale, and re-anchor hand features
    skeleton = normalize_skeleton(skeleton)

    # Step 4: Resample to exactly TARGET_LEN frames via linear interpolation
    skeleton = temporal_resample(skeleton, target_len=TARGET_LEN)

    # Step 5: Convert to float32 tensor and add batch dimension
    # Shape: (1, TARGET_LEN, 150) — batch of one sample
    tensor = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Step 6: Forward pass (no_grad = no gradient tracking, faster + less memory)
    with torch.no_grad():
        output = model(tensor)              # → (1, num_classes) logits
        pred_idx = output.argmax(dim=1).item()  # index of highest logit

    return idx_to_label.get(pred_idx, f"unknown({pred_idx})")


# ── Evaluate on saved dataset ─────────────────────────────────────────────────
def evaluate_dataset():
    """Load saved dataset and print classification report + confusion matrix.

    Loads whichever processed dataset file is available (priority: X_final >
    X_normalized > X_raw), then runs the model in batch mode over all samples.

    Output:
        - sklearn classification_report: per-class precision, recall, F1, support
        - confusion matrix (raw counts)
    """
    # ── Load label map ────────────────────────────────────────────────────────
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        print("❌ label_map.json not found. Run extract.py first.")
        return

    with open(label_map_path) as f:
        label_map = json.load(f)

    num_classes  = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(num_classes)

    # ── Load data (try best available file in priority order) ─────────────────
    # X_final.npy is the final post-resample file from the full pipeline;
    # fall back to earlier pipeline outputs if the later ones don't exist yet.
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            X = np.load(path)
            break
    else:
        print("❌ No processed data found. Run preprocessing scripts first.")
        return

    # Load integer class labels
    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))

    # ── Build DataLoader ──────────────────────────────────────────────────────
    # augment=False: deterministic inference (no augmentation during evaluation)
    dataset    = SignDataset(X, y, augment=False)
    from torch.utils.data import DataLoader
    loader     = DataLoader(dataset, batch_size=16, shuffle=False)

    # ── Run inference over all batches ────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            output  = model(X_batch)                       # → (batch, num_classes)
            preds   = output.argmax(dim=1).cpu().numpy()   # predicted class indices
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    # ── Print results ─────────────────────────────────────────────────────────
    # Build human-readable class names in label index order
    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("── Confusion Matrix ───────────────────────────────────")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate or run inference on sign language model")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video for inference")
    args = parser.parse_args()

    if args.video:
        # Single-video inference mode
        label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
        with open(label_map_path) as f:
            label_map = json.load(f)

        model = load_model(len(label_map))
        result = predict_video(args.video, model, label_map)
        print(f"\nPredicted class: {result}")
    else:
        # Full dataset evaluation mode
        evaluate_dataset()