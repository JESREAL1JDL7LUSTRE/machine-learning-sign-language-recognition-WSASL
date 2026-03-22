import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

# ── Path fix ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.lstm      import SignLSTM
from training.dataset import SignDataset
from preprocessing.extract    import extract_skeleton
from preprocessing.normalize  import normalize_skeleton, clean_missing_keypoints
from preprocessing.resample   import temporal_resample

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT, "models", "sign_lstm.pth")
OUTPUT_DIR = os.path.join(ROOT, "output")
TARGET_LEN = 64


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(num_classes):
    model = SignLSTM(
        input_size   = 150,
        hidden_size  = 128,
        num_layers   = 2,
        num_classes  = num_classes,
        dropout      = 0.3,
        bidirectional= True
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ── Single video prediction ───────────────────────────────────────────────────
def predict_video(video_path, model, label_map):
    """
    Full pipeline: raw video → predicted class label.

    Args:
        video_path : path to .mp4 / .avi file
        model      : loaded SignLSTM
        label_map  : dict {label_name: index}

    Returns:
        predicted label (str)
    """
    idx_to_label = {v: k for k, v in label_map.items()}

    # 1. Extract
    skeleton = extract_skeleton(video_path, max_frames=TARGET_LEN)

    # 2. Clean + Normalize + Resample
    skeleton = clean_missing_keypoints(skeleton)
    skeleton = normalize_skeleton(skeleton)
    skeleton = temporal_resample(skeleton, target_len=TARGET_LEN)

    # 3. To tensor
    tensor = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 4. Predict
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()

    return idx_to_label.get(pred_idx, f"unknown({pred_idx})")


# ── Evaluate on saved dataset ─────────────────────────────────────────────────
def evaluate_dataset():
    """
    Load X_final.npy / y.npy and print a full classification report.
    """
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        print("❌ label_map.json not found. Run extract.py first.")
        return

    with open(label_map_path) as f:
        label_map = json.load(f)

    num_classes  = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    model = load_model(num_classes)

    # Load data
    for fname in ["X_final.npy", "X_normalized.npy", "X_raw.npy"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            X = np.load(path)
            break
    else:
        print("❌ No processed data found. Run preprocessing scripts first.")
        return

    y = np.load(os.path.join(OUTPUT_DIR, "y.npy"))

    dataset    = SignDataset(X, y, augment=False)
    from torch.utils.data import DataLoader
    loader     = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            output  = model(X_batch)
            preds   = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("── Confusion Matrix ───────────────────────────────────")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate or run inference on sign language model")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video for inference")
    args = parser.parse_args()

    if args.video:
        label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
        with open(label_map_path) as f:
            label_map = json.load(f)

        model = load_model(len(label_map))
        result = predict_video(args.video, model, label_map)
        print(f"\nPredicted class: {result}")
    else:
        evaluate_dataset()