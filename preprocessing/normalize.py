import numpy as np
import os


def normalize_skeleton(skeleton):
    """
    Normalize a single skeleton sequence.

    Steps:
      1. Center all coordinates on the nose landmark (landmark 0).
      2. Scale by the shoulder width (distance between left & right shoulder).

    Args:
        skeleton: numpy array of shape (T, 150)

    Returns:
        Normalized skeleton of shape (T, 150)
    """
    skeleton = skeleton.copy().astype(np.float32)

    for t in range(skeleton.shape[0]):
        frame = skeleton[t]

        # Reference point: nose = landmark 0 → indices (0, 1)
        cx, cy = frame[0], frame[1]

        # Skip frames where nose is missing (all zeros)
        if cx == 0.0 and cy == 0.0:
            continue

        # ── 1. CENTER ──────────────────────────────────────────────────────
        # Subtract reference point from every (x, y) pair
        for i in range(0, len(frame), 2):
            frame[i]   -= cx   # x
            frame[i+1] -= cy   # y

        # ── 2. SCALE ───────────────────────────────────────────────────────
        # Use shoulder width as the scale reference
        # Pose landmark 11 = left shoulder, 12 = right shoulder
        left_shoulder  = frame[11 * 2 : 11 * 2 + 2]   # (x, y)
        right_shoulder = frame[12 * 2 : 12 * 2 + 2]   # (x, y)

        dist = np.linalg.norm(
            np.array(left_shoulder) - np.array(right_shoulder)
        )

        if dist > 1e-6:   # avoid division by zero
            frame /= dist

        skeleton[t] = frame

    return skeleton


def normalize_dataset(X):
    """
    Apply normalize_skeleton to every sample in the dataset.

    Args:
        X: numpy array of shape (N, T, 150)

    Returns:
        Normalized array of shape (N, T, 150)
    """
    X_norm = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        X_norm[i] = normalize_skeleton(X[i])

    return X_norm


def clean_missing_keypoints(skeleton, strategy="interpolate"):
    """
    Handle frames where keypoints are fully zero (landmark not detected).

    Strategies:
      - 'interpolate': linearly interpolate between valid frames
      - 'repeat'     : copy the last valid frame forward

    Args:
        skeleton: numpy array of shape (T, F)
        strategy: 'interpolate' or 'repeat'

    Returns:
        Cleaned skeleton of shape (T, F)
    """
    skeleton = skeleton.copy()
    T = skeleton.shape[0]

    valid = [t for t in range(T) if not np.all(skeleton[t] == 0)]

    if len(valid) == 0:
        return skeleton  # nothing we can do

    if strategy == "repeat":
        last_valid = skeleton[valid[0]]
        for t in range(T):
            if np.all(skeleton[t] == 0):
                skeleton[t] = last_valid
            else:
                last_valid = skeleton[t]

    elif strategy == "interpolate":
        # Forward fill first, then interpolate
        for t in range(T):
            if np.all(skeleton[t] == 0):
                # Find nearest valid frame before and after
                before = [v for v in valid if v < t]
                after  = [v for v in valid if v > t]

                if before and after:
                    t0, t1 = before[-1], after[0]
                    alpha = (t - t0) / (t1 - t0)
                    skeleton[t] = (1 - alpha) * skeleton[t0] + alpha * skeleton[t1]
                elif before:
                    skeleton[t] = skeleton[before[-1]]
                elif after:
                    skeleton[t] = skeleton[after[0]]

    return skeleton


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR = "../output"

    X_raw_path = os.path.join(OUTPUT_DIR, "X_raw.npy")

    if not os.path.exists(X_raw_path):
        print("❌ X_raw.npy not found. Run extract.py first.")
        exit(1)

    print("Loading X_raw.npy ...")
    X_raw = np.load(X_raw_path)
    print(f"   Shape: {X_raw.shape}")

    print("\nCleaning missing keypoints ...")
    X_clean = np.array(
        [clean_missing_keypoints(X_raw[i]) for i in range(len(X_raw))],
        dtype=np.float32
    )

    print("Normalizing skeletons ...")
    X_norm = normalize_dataset(X_clean)

    save_path = os.path.join(OUTPUT_DIR, "X_normalized.npy")
    np.save(save_path, X_norm)

    print(f"\n✅ Saved normalized data → {save_path}")
    print(f"   Shape: {X_norm.shape}")