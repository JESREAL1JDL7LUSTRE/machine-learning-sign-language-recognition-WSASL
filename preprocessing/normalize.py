"""
normalize.py
============
Full preprocessing pipeline for MediaPipe 150-feature skeleton data.

Steps:
  1. Clean missing keypoints (interpolate zeros)
  2. Filter to upper body + hands only (150 → 102 features)
     Drops: legs, face landmarks — noise for sign language
  3. Center on mid-hip (stable reference)
  4. Scale by torso height
  5. Make hand coords relative to their wrist root

Feature layout after filtering:
  Upper body: 9 joints * 2 = 18 features
  Left hand : 21 joints * 2 = 42 features
  Right hand : 21 joints * 2 = 42 features
  TOTAL = 102 features per frame
"""

import numpy as np
import os

# ── MediaPipe Pose joint indices ──────────────────────────────────────────────
NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

# Upper body joints to KEEP (drop legs 25-32, drop face 1-10, 17-22)
UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # 9 joints

# Feature indices in the original 150-feature array
UPPER_BODY_FEAT = []
for j in UPPER_BODY_JOINTS:
    UPPER_BODY_FEAT.extend([j * 2, j * 2 + 1])   # 18 features

# Hand features: left hand = 66-107, right hand = 108-149
HAND_FEAT   = list(range(66, 150))                # 84 features
KEEP_INDICES= UPPER_BODY_FEAT + HAND_FEAT         # 102 total

# After filtering, new indices for reference joints
# Upper body order: nose(0), lshoulder(1), rshoulder(2), lelbow(3),
#                   relbow(4), lwrist(5), rwrist(6), lhip(7), rhip(8)
_F_NOSE    = 0   # joint index in filtered array
_F_LSH     = 1
_F_RSH     = 2
_F_LHI     = 7
_F_RHI     = 8

# Hand offsets in filtered array
FILT_LEFT_HAND_OFFSET  = 18   # starts after 9 upper body joints * 2
FILT_RIGHT_HAND_OFFSET = 60   # 18 + 42
HAND_JOINTS = 21


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CLEAN MISSING KEYPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def clean_missing_keypoints(skeleton, strategy="interpolate"):
    """
    Handle frames where all keypoints are zero (not detected).

    Strategies:
      'interpolate' — linear interpolation between valid frames
      'repeat'      — copy last valid frame forward
    """
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
                    t0, t1  = before[-1], after[0]
                    alpha   = (t - t0) / (t1 - t0)
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
    """
    Keep only upper body + both hands. Drop legs and face.

    Args:
        X : (N, T, 150)

    Returns:
        (N, T, 102)
    """
    return X[:, :, KEEP_INDICES].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3-5 — NORMALIZE (center + scale + relative hands)
# ══════════════════════════════════════════════════════════════════════════════

def make_relative_hands(frame):
    """
    Make hand keypoints relative to their own wrist (joint 0).
    Works on filtered 102-feature frame.
    """
    frame = frame.copy()

    # Left hand root
    lx = frame[FILT_LEFT_HAND_OFFSET]
    ly = frame[FILT_LEFT_HAND_OFFSET + 1]
    if not (lx == 0.0 and ly == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_LEFT_HAND_OFFSET + j*2]     -= lx
            frame[FILT_LEFT_HAND_OFFSET + j*2 + 1] -= ly

    # Right hand root
    rx = frame[FILT_RIGHT_HAND_OFFSET]
    ry = frame[FILT_RIGHT_HAND_OFFSET + 1]
    if not (rx == 0.0 and ry == 0.0):
        for j in range(HAND_JOINTS):
            frame[FILT_RIGHT_HAND_OFFSET + j*2]     -= rx
            frame[FILT_RIGHT_HAND_OFFSET + j*2 + 1] -= ry

    return frame


def normalize_skeleton(skeleton):
    """
    Normalize a filtered (T, 102) skeleton sequence.

    1. Center on mid-hip
    2. Scale by torso height
    3. Relative hand coordinates
    """
    skeleton = skeleton.copy().astype(np.float32)

    for t in range(skeleton.shape[0]):
        frame = skeleton[t]

        # Mid-hip reference (_F_LHI=7, _F_RHI=8 in filtered joints)
        lhx = frame[_F_LHI * 2]
        lhy = frame[_F_LHI * 2 + 1]
        rhx = frame[_F_RHI * 2]
        rhy = frame[_F_RHI * 2 + 1]

        if lhx == 0.0 and rhx == 0.0:
            cx = frame[_F_NOSE * 2]
            cy = frame[_F_NOSE * 2 + 1]
        else:
            cx = (lhx + rhx) / 2.0
            cy = (lhy + rhy) / 2.0

        if cx == 0.0 and cy == 0.0:
            continue

        # Center
        frame[0::2] -= cx
        frame[1::2] -= cy

        # Scale by torso height (mid-shoulder to mid-hip)
        lsx = frame[_F_LSH * 2];  lsy = frame[_F_LSH * 2 + 1]
        rsx = frame[_F_RSH * 2];  rsy = frame[_F_RSH * 2 + 1]
        mid_sh  = np.array([(lsx+rsx)/2, (lsy+rsy)/2])
        mid_hip = np.array([0.0, 0.0])
        torso   = np.linalg.norm(mid_sh - mid_hip)

        if torso > 1e-6:
            frame /= torso

        # Relative hand coordinates
        frame = make_relative_hands(frame)
        skeleton[t] = frame

    return skeleton


def normalize_dataset(X):
    """Apply full normalization to (N, T, 102) dataset."""
    return np.array(
        [normalize_skeleton(X[i]) for i in range(len(X))],
        dtype=np.float32
    )


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

    save_path = os.path.join(OUTPUT_DIR, "X_normalized.npy")
    np.save(save_path, X_norm)

    print(f"\n✅ Saved → {save_path}")
    print(f"   Shape: {X_norm.shape}")
    print(f"   Features: 102 (was 150 — dropped {150-102} noisy features)")