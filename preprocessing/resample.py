import numpy as np
import os


def temporal_resample(skeleton, target_len=64):
    """
    Resample a skeleton sequence to a fixed number of frames.

    Uses linear index interpolation — works for both
    upsampling (short videos) and downsampling (long videos).

    Args:
        skeleton   : numpy array of shape (T, F)
        target_len : desired number of frames (default: 64)

    Returns:
        Resampled array of shape (target_len, F)
    """
    T = len(skeleton)

    if T == target_len:
        return skeleton

    indices = np.linspace(0, T - 1, target_len).astype(int)
    return skeleton[indices]


def resample_dataset(X, target_len=64):
    """
    Apply temporal_resample to every sample in the dataset.

    Args:
        X          : numpy array of shape (N, T, F)  — T may vary if stored as object array
        target_len : target number of frames

    Returns:
        numpy array of shape (N, target_len, F)
    """
    resampled = []

    for i in range(len(X)):
        resampled.append(temporal_resample(X[i], target_len=target_len))

    return np.array(resampled, dtype=np.float32)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR = "../output"
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