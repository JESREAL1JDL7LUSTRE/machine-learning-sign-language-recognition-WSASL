"""
normalize.py
============
Full preprocessing pipeline for MediaPipe 150-feature skeleton data.

Steps:
  1. Clean missing keypoints (interpolate zeros)
  2. Filter to upper body + hands only (150 → 102 features)
  3. Center on mid-hip
  4. Scale by torso height
  5. Make hand coords relative to their wrist root
  6. Smooth joints over time (Gaussian filter)
  7. Compute bone vectors and motion streams

Output files:
  X_normalized.npy  — joint positions  (N, T, 102)
  X_bones.npy       — bone vectors     (N, T, 102)
  X_motion.npy      — motion deltas    (N, T, 102)
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# ── MediaPipe Pose joint indices ──────────────────────────────────────────────
UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # 9 joints

UPPER_BODY_FEAT = []
for j in UPPER_BODY_JOINTS:
    UPPER_BODY_FEAT.extend([j * 2, j * 2 + 1])   # 18 features

HAND_FEAT    = list(range(66, 150))               # 84 features
KEEP_INDICES = UPPER_BODY_FEAT + HAND_FEAT        # 102 total

# Filtered array reference indices
_F_NOSE = 0
_F_LSH  = 1
_F_RSH  = 2
_F_LHI  = 7
_F_RHI  = 8

FILT_LEFT_HAND_OFFSET  = 18
FILT_RIGHT_HAND_OFFSET = 60
HAND_JOINTS = 21

# ── Bone edges in filtered 51-joint space ────────────────────────────────────
# (parent, child) pairs — bone vector = child - parent
BONE_EDGES = [
    # Upper body
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5),
    (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
    # Left hand (offset 9)
    (9,10),(10,11),(11,12),(12,13),
    (9,14),(14,15),(15,16),(16,17),
    (9,18),(18,19),(19,20),(20,21),
    (9,22),(22,23),(23,24),(24,25),
    (9,26),(26,27),(27,28),(28,29),
    # Right hand (offset 30)
    (30,31),(31,32),(32,33),(33,34),
    (30,35),(35,36),(36,37),(37,38),
    (30,39),(39,40),(40,41),(41,42),
    (30,43),(43,44),(44,45),(45,46),
    (30,47),(47,48),(48,49),(49,50),
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CLEAN MISSING KEYPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def clean_missing_keypoints(skeleton, strategy="interpolate"):
    skeleton = skeleton.copy()
    T        = skeleton.shape[0]
    valid    = [t for t in range(T) if not np.all(skeleton[t] == 0)]

    if len(valid) == 0:
        return skeleton

    if strategy == "repeat":
        last = skeleton[valid[0]]
        for t in range(T):
            if np.all(skeleton[t] == 0):
                skeleton[t] = last
            else:
                last = skeleton[t]

    elif strategy == "interpolate":
        for t in range(T):
            if np.all(skeleton[t] == 0):
                before = [v for v in valid if v < t]
                after  = [v for v in valid if v > t]
                if before and after:
                    t0, t1 = before[-1], after[0]
                    alpha  = (t - t0) / (t1 - t0)
                    skeleton[t] = (1-alpha)*skeleton[t0] + alpha*skeleton[t1]
                elif before:
                    skeleton[t] = skeleton[before[-1]]
                elif after:
                    skeleton[t] = skeleton[after[0]]

    return skeleton


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FILTER JOINTS (150 → 102)
# ══════════════════════════════════════════════════════════════════════════════

def filter_joints(X):
    return X[:, :, KEEP_INDICES].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3-5 — NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════

def make_relative_hands(frame):
    frame = frame.copy()

    lx = frame[FILT_LEFT_HAND_OFFSET]
    ly = frame[FILT_LEFT_HAND_OFFSET + 1]
    if not (lx == 0.0 and ly == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_LEFT_HAND_OFFSET + j*2]     -= lx
            frame[FILT_LEFT_HAND_OFFSET + j*2 + 1] -= ly

    rx = frame[FILT_RIGHT_HAND_OFFSET]
    ry = frame[FILT_RIGHT_HAND_OFFSET + 1]
    if not (rx == 0.0 and ry == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_RIGHT_HAND_OFFSET + j*2]     -= rx
            frame[FILT_RIGHT_HAND_OFFSET + j*2 + 1] -= ry

    return frame


def normalize_skeleton(skeleton):
    skeleton = skeleton.copy().astype(np.float32)

    for t in range(skeleton.shape[0]):
        frame = skeleton[t]

        lhx = frame[_F_LHI * 2];  lhy = frame[_F_LHI * 2 + 1]
        rhx = frame[_F_RHI * 2];  rhy = frame[_F_RHI * 2 + 1]

        if lhx == 0.0 and rhx == 0.0:
            cx = frame[_F_NOSE * 2]
            cy = frame[_F_NOSE * 2 + 1]
        else:
            cx = (lhx + rhx) / 2.0
            cy = (lhy + rhy) / 2.0

        if cx == 0.0 and cy == 0.0:
            continue

        frame[0::2] -= cx
        frame[1::2] -= cy

        lsx = frame[_F_LSH * 2];  lsy = frame[_F_LSH * 2 + 1]
        rsx = frame[_F_RSH * 2];  rsy = frame[_F_RSH * 2 + 1]
        mid_sh  = np.array([(lsx+rsx)/2, (lsy+rsy)/2])
        torso   = np.linalg.norm(mid_sh)

        if torso > 1e-6:
            frame /= torso

        frame       = make_relative_hands(frame)
        skeleton[t] = frame

    return skeleton


def normalize_dataset(X):
    return np.array(
        [normalize_skeleton(X[i]) for i in range(len(X))],
        dtype=np.float32
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TEMPORAL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def smooth_skeleton(skeleton, sigma=1.0):
    """
    Apply Gaussian smoothing along the time axis.
    Reduces MediaPipe detection jitter.

    Args:
        skeleton : (T, F)
        sigma    : smoothing strength (1.0 = mild)
    """
    return gaussian_filter1d(skeleton, sigma=sigma, axis=0).astype(np.float32)


def smooth_dataset(X, sigma=1.0):
    return np.array(
        [smooth_skeleton(X[i], sigma) for i in range(len(X))],
        dtype=np.float32
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MULTI-STREAM: BONES + MOTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_bone_vectors(X):
    """
    Compute bone vectors for each frame.
    Bone = child_joint - parent_joint.

    Args:
        X : (N, T, 102)  — joint positions

    Returns:
        (N, T, 102) — bone vectors (same shape, unused bones = 0)
    """
    N, T, F = X.shape
    V       = F // 2
    # Reshape to (N, T, V, 2)
    joints  = X.reshape(N, T, V, 2)
    bones   = np.zeros_like(joints)

    for parent, child in BONE_EDGES:
        if parent < V and child < V:
            bones[:, :, child, :] = joints[:, :, child, :] - joints[:, :, parent, :]

    return bones.reshape(N, T, F).astype(np.float32)


def compute_motion(X):
    """
    Compute frame-to-frame motion (temporal derivative).
    motion[t] = joints[t+1] - joints[t]
    Last frame is zero-padded.

    Args:
        X : (N, T, F)

    Returns:
        (N, T, F) — motion stream
    """
    motion          = np.zeros_like(X, dtype=np.float32)
    motion[:, :-1]  = X[:, 1:] - X[:, :-1]
    return motion


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")

    X_raw_path = os.path.join(OUTPUT_DIR, "X_raw.npy")
    if not os.path.exists(X_raw_path):
        print("❌ X_raw.npy not found. Run extract.py first.")
        exit(1)

    print("Loading X_raw.npy ...")
    X_raw = np.load(X_raw_path)
    print(f"   Shape: {X_raw.shape}")

    print("\nStep 1 — Cleaning missing keypoints ...")
    X_clean = np.array(
        [clean_missing_keypoints(X_raw[i]) for i in range(len(X_raw))],
        dtype=np.float32
    )

    print("\nStep 2 — Filtering joints (150 → 102) ...")
    print("   Keeping: upper body (9 joints) + both hands (42 joints)")
    print("   Dropping: legs, face landmarks")
    X_filt = filter_joints(X_clean)
    print(f"   Shape after filter: {X_filt.shape}")

    print("\nStep 3 — Normalizing (center + scale + relative hands) ...")
    X_norm = normalize_dataset(X_filt)

    print("\nStep 4 — Temporal smoothing (sigma=1.0) ...")
    X_smooth = smooth_dataset(X_norm, sigma=1.0)

    print("\nStep 5 — Computing bone vectors ...")
    X_bones = compute_bone_vectors(X_smooth)

    print("\nStep 6 — Computing motion stream ...")
    X_motion = compute_motion(X_smooth)

    # Save all three streams
    np.save(os.path.join(OUTPUT_DIR, "X_normalized.npy"), X_smooth)
    np.save(os.path.join(OUTPUT_DIR, "X_bones.npy"),      X_bones)
    np.save(os.path.join(OUTPUT_DIR, "X_motion.npy"),     X_motion)

    print(f"\n✅ Saved all streams → {OUTPUT_DIR}")
    print(f"   X_normalized : {X_smooth.shape}  (joint positions)")
    print(f"   X_bones      : {X_bones.shape}   (bone vectors)")
    print(f"   X_motion     : {X_motion.shape}  (motion deltas)")
    print(f"   Features     : 102 (was 150 — dropped {150-102} noisy features)")