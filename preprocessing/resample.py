import numpy as np
import os


def temporal_resample(skeleton, target_len=64):

    T = len(skeleton)

    if T == target_len:
        return skeleton

    indices = np.linspace(0, T - 1, target_len).astype(int)
    return skeleton[indices]


def resample_dataset(X, target_len=64):

    resampled = []

    for i in range(len(X)):
        resampled.append(temporal_resample(X[i], target_len=target_len))

    return np.array(resampled, dtype=np.float32)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")
    TARGET_LEN = 64

    # Try to load normalized first, fall back to raw
    norm_path = os.path.join(OUTPUT_DIR, "X_normalized.npy")
    raw_path  = os.path.join(OUTPUT_DIR, "X_raw.npy")

    if os.path.exists(norm_path):
        print(f"Loading X_normalized.npy ...")
        X = np.load(norm_path)
    elif os.path.exists(raw_path):
        print(f"Loading X_raw.npy (normalized version not found) ...")
        X = np.load(raw_path)
    else:
        print("❌ No data found. Run extract.py (and optionally normalize.py) first.")
        exit(1)

    print(f"   Input shape : {X.shape}")

    print(f"\nResampling to {TARGET_LEN} frames ...")
    X_resampled = resample_dataset(X, target_len=TARGET_LEN)

    save_path = os.path.join(OUTPUT_DIR, "X_final.npy")
    np.save(save_path, X_resampled)

    print(f"\n✅ Saved resampled data → {save_path}")
    print(f"   Output shape : {X_resampled.shape}")