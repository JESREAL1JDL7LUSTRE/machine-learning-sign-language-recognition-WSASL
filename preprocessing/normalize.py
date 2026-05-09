"""
normalize.py — Skeleton Normalization and Feature Engineering
=============================================================

This is the second step of the preprocessing pipeline, applied AFTER extract.py.
It transforms the raw keypoint array (N, T, 150) into cleaned, normalized, and
augmented representations suitable for graph-based models.

Pipeline steps (run sequentially when called as __main__):
    1. Clean missing keypoints  — per-joint temporal interpolation over zero frames
    2. Filter joints            — keep 51 joints (102 features) from the raw 75 (150 features)
    3. Normalize per frame      — center on mid-shoulder, scale by shoulder width
    4. Smooth temporally        — Gaussian filter along time to reduce detection jitter
    5. Compute bone vectors     — child - parent displacement for each joint edge
    6. Compute joint motion     — frame-to-frame coordinate delta
    7. Compute bone motion      — frame-to-frame bone vector delta

Saved outputs (to output/):
    X_normalized.npy   — joint positions after normalization and smoothing
    X_bones.npy        — bone vectors
    X_motion.npy       — joint motion (velocity)
    X_bone_motion.npy  — bone motion (angular velocity)

Each of these becomes an independent input stream to the ST-GCN models.

Usage:
    python preprocessing/normalize.py

Functions:
    filter_joints           — Select 51 joints from the 75 raw MediaPipe joints
    clean_missing_keypoints — Interpolate zeros along each joint's track
    make_relative_hands     — Re-anchor hand joints to their respective wrist
    normalize_skeleton      — Per-frame centering + scale normalization
    normalize_dataset       — Apply normalize_skeleton over an entire dataset
    smooth_skeleton         — Gaussian temporal smoothing of one video
    smooth_dataset          — Apply smooth_skeleton over an entire dataset
    compute_bone_vectors    — Compute bone (child - parent) features
    compute_motion          — Compute frame-to-frame delta features
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── Joint index definitions (in the FILTERED 51-joint space) ──────────────────
# After filtering, the 9 upper-body joints occupy positions 0–8:
#   0=nose, 1=L_shoulder, 2=R_shoulder, 3=L_elbow, 4=R_elbow
#   5=L_wrist, 6=R_wrist, 7=L_hip, 8=R_hip
# The raw MediaPipe pose outputs 33 joints; we only keep these 9:
UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # raw MediaPipe indices

# Convert joint indices to flat feature indices (each joint has x at 2*j, y at 2*j+1)
UPPER_BODY_FEAT = []
for j in UPPER_BODY_JOINTS:
    UPPER_BODY_FEAT.extend([j * 2, j * 2 + 1])

# Hand features are already at the end of the 150-d feature vector (joints 33–74)
# and occupy flat indices 66–149 (42 features per hand × 2 hands = 84)
HAND_FEAT = list(range(66, 150))

# Combined set of indices to KEEP from the raw 150-d feature vector → 102 features
# This selects: 9 upper-body joints × 2 + 21+21 hand joints × 2 = 102 features = 51 joints
KEEP_INDICES = UPPER_BODY_FEAT + HAND_FEAT

# ── Named joint indices in the FILTERED 51-joint space ────────────────────────
# Used for robust center and scale estimation in normalize_skeleton
_F_NOSE = 0   # nose (global position reference)
_F_LSH  = 1   # left shoulder
_F_RSH  = 2   # right shoulder
_F_LEL  = 3   # left elbow
_F_REL  = 4   # right elbow
_F_LWR  = 5   # left wrist
_F_RWR  = 6   # right wrist
_F_LHI  = 7   # left hip
_F_RHI  = 8   # right hip

# ── Offsets into the 102-d filtered feature vector ────────────────────────────
# Left hand starts at joint 9 → flat index 18 (9 upper-body joints × 2)
FILT_LEFT_HAND_OFFSET  = 18
# Right hand starts at joint 30 → flat index 60 (18 + 21 left hand joints × 2)
FILT_RIGHT_HAND_OFFSET = 60
HAND_JOINTS = 21   # joints per hand

# ── Bone edge list for the FILTERED 51-joint skeleton ─────────────────────────
# Each (parent, child) pair defines a bone vector = child_pos - parent_pos.
# Used to compute bone stream features for multi-stream models.
# Upper body edges:
BONE_EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
    # Left hand finger chains (parent=wrist root at joint 9, child=finger joints 10–29):
    (9, 10), (10, 11), (11, 12), (12, 13),     # thumb
    (9, 14), (14, 15), (15, 16), (16, 17),     # index
    (9, 18), (18, 19), (19, 20), (20, 21),     # middle
    (9, 22), (22, 23), (23, 24), (24, 25),     # ring
    (9, 26), (26, 27), (27, 28), (28, 29),     # pinky
    # Right hand finger chains (parent=wrist root at joint 30):
    (30, 31), (31, 32), (32, 33), (33, 34),    # thumb
    (30, 35), (35, 36), (36, 37), (37, 38),    # index
    (30, 39), (39, 40), (40, 41), (41, 42),    # middle
    (30, 43), (43, 44), (44, 45), (45, 46),    # ring
    (30, 47), (47, 48), (48, 49), (49, 50),    # pinky
]


# ══════════════════════════════════════════════════════════════════════════════
# JOINT FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_joints(X):
    """Select the 51 informative joints from the raw 75-joint MediaPipe output.

    The raw MediaPipe extractor outputs 75 joints (33 pose + 21 left hand + 21 right hand),
    encoded as 150 flat features (x, y per joint). Many body joints (knees, ankles, face
    details) are irrelevant for sign language recognition. This function keeps:
        - 9 upper-body joints (nose, shoulders, elbows, wrists, hips)
        - 21 left-hand joints
        - 21 right-hand joints
    Total: 51 joints → 102 flat features.

    Args:
        X (np.ndarray): Raw skeleton array, shape (N, T, 150).

    Returns:
        np.ndarray: Filtered array, shape (N, T, 102), float32.
    """
    return X[:, :, KEEP_INDICES].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MISSING KEYPOINT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_joint_track(track_xy):
    """Interpolate a single joint's (x, y) trajectory over time.

    Frames where both x and y are exactly 0.0 are treated as 'missing'
    (the detector did not find this joint). Their values are filled by
    linear interpolation from surrounding valid frames.

    If the joint is missing for the ENTIRE video, it is left as zeros.

    Args:
        track_xy (np.ndarray): Shape (T, 2) — x and y per frame for one joint.

    Returns:
        np.ndarray: Shape (T, 2) — interpolated trajectory.
    """
    track_xy = track_xy.astype(np.float32)
    T = track_xy.shape[0]

    # A frame is 'valid' if at least one of x or y is non-zero
    valid = ~np.all(np.isclose(track_xy, 0.0), axis=1)
    if not np.any(valid):
        return track_xy   # joint missing in all frames — leave as zeros

    valid_idx = np.where(valid)[0]    # indices of valid frames
    full_idx  = np.arange(T)          # all frame indices

    # Linear interpolation using NumPy interp (handles boundary extrapolation)
    out = track_xy.copy()
    out[:, 0] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 0])  # x channel
    out[:, 1] = np.interp(full_idx, valid_idx, track_xy[valid_idx, 1])  # y channel
    return out


def clean_missing_keypoints(skeleton):
    """Fill missing keypoints by per-joint linear temporal interpolation.

    This is more robust than frame-level filling (e.g. repeating the last frame)
    because only the specific joints that are missing are fixed, not the whole frame.

    Args:
        skeleton (np.ndarray): Shape (T, F) — one video's raw features.
                               F must be even (each joint has x at 2k, y at 2k+1).

    Returns:
        np.ndarray: Shape (T, F) — float32 skeleton with zeros interpolated.
    """
    skeleton = skeleton.astype(np.float32).copy()
    T, F = skeleton.shape
    V = F // 2   # number of joints

    # Reshape to (T, V, 2) so we can process each joint's track independently
    joints = skeleton.reshape(T, V, 2)
    for v in range(V):
        joints[:, v, :] = _interpolate_joint_track(joints[:, v, :])

    return joints.reshape(T, F)   # flatten back to (T, F)


# ══════════════════════════════════════════════════════════════════════════════
# HAND RE-ANCHORING
# ══════════════════════════════════════════════════════════════════════════════

def make_relative_hands(frame):
    """Re-anchor hand keypoints relative to their own wrist root.

    After global centering and scaling (performed in normalize_skeleton),
    the hand joints are still in global (body-relative) coordinates.
    This function subtracts the wrist root position from all other hand joints,
    so that hand shape is captured independently of where the hand is in space.

    This makes the hand features invariant to arm position and improves
    the model's ability to discriminate fine-grained hand shapes.

    Args:
        frame (np.ndarray): One frame's flat feature vector, shape (F,).
                            F = 102 after filtering (51 joints × 2).

    Returns:
        np.ndarray: Shape (F,) — frame with relative hand coordinates.
    """
    frame = frame.copy()

    # ── Left hand: joint 9 is the wrist root ──────────────────────────────────
    lx = frame[FILT_LEFT_HAND_OFFSET]       # x of left wrist root
    ly = frame[FILT_LEFT_HAND_OFFSET + 1]   # y of left wrist root
    if not (lx == 0.0 and ly == 0.0):
        # Subtract wrist root from all 21 left-hand joints
        for j in range(HAND_JOINTS):
            frame[FILT_LEFT_HAND_OFFSET + j * 2]     -= lx
            frame[FILT_LEFT_HAND_OFFSET + j * 2 + 1] -= ly

    # ── Right hand: joint 30 is the wrist root ────────────────────────────────
    rx = frame[FILT_RIGHT_HAND_OFFSET]
    ry = frame[FILT_RIGHT_HAND_OFFSET + 1]
    if not (rx == 0.0 and ry == 0.0):
        # Subtract wrist root from all 21 right-hand joints
        for j in range(HAND_JOINTS):
            frame[FILT_RIGHT_HAND_OFFSET + j * 2]     -= rx
            frame[FILT_RIGHT_HAND_OFFSET + j * 2 + 1] -= ry

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# CENTER AND SCALE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _joint_xy(frame, joint_idx):
    """Extract (x, y) for a specific joint from a flat feature frame.

    Returns None if both x and y are approximately zero (joint missing).

    Args:
        frame     (np.ndarray): Flat feature vector, shape (F,).
        joint_idx (int)       : Joint index (0-based in filtered 51-joint space).

    Returns:
        np.ndarray or None: (2,) array of [x, y], or None if missing.
    """
    x = frame[joint_idx * 2]
    y = frame[joint_idx * 2 + 1]
    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        return None   # joint not detected
    return np.array([x, y], dtype=np.float32)


def _pick_center(frame):
    """Choose the best available centering point for a frame (priority order).

    The center is used to zero-mean the skeleton so it is position-invariant.
    Priority:
        1. Mid-shoulder (average of left and right shoulder) — most stable
        2. Mid-hip      (average of left and right hip)
        3. Nose         (fallback for heavily occluded torsos)
        4. None         (all landmarks missing → skip this frame)

    Args:
        frame (np.ndarray): Flat feature vector.

    Returns:
        np.ndarray or None: (2,) center point, or None if no reference found.
    """
    lsh = _joint_xy(frame, _F_LSH)
    rsh = _joint_xy(frame, _F_RSH)
    if lsh is not None and rsh is not None:
        return 0.5 * (lsh + rsh)   # mid-shoulder (primary center)

    lhi = _joint_xy(frame, _F_LHI)
    rhi = _joint_xy(frame, _F_RHI)
    if lhi is not None and rhi is not None:
        return 0.5 * (lhi + rhi)   # mid-hip (secondary center)

    nose = _joint_xy(frame, _F_NOSE)
    if nose is not None:
        return nose   # nose (last resort)

    return None   # all candidates missing


def _pick_scale(frame):
    """Choose the best available scale reference for a frame (priority order).

    The scale factor normalizes skeleton size so the model is scale-invariant.
    Priority:
        1. Shoulder width (distance between left and right shoulder) — most stable
        2. Left forearm length (elbow to wrist distance)
        3. Right forearm length
        4. 1.0 (no reference → skip scaling)

    Args:
        frame (np.ndarray): Flat feature vector.

    Returns:
        float: Positive scale factor.
    """
    lsh = _joint_xy(frame, _F_LSH)
    rsh = _joint_xy(frame, _F_RSH)
    if lsh is not None and rsh is not None:
        d = np.linalg.norm(lsh - rsh)   # Euclidean shoulder width
        if d > 1e-6:
            return d   # shoulder width (primary scale)

    lwr = _joint_xy(frame, _F_LWR)
    lel = _joint_xy(frame, _F_LEL)
    if lwr is not None and lel is not None:
        d = np.linalg.norm(lwr - lel)
        if d > 1e-6:
            return d   # left forearm length

    rwr = _joint_xy(frame, _F_RWR)
    rel = _joint_xy(frame, _F_REL)
    if rwr is not None and rel is not None:
        d = np.linalg.norm(rwr - rel)
        if d > 1e-6:
            return d   # right forearm length

    return 1.0   # degenerate frame — no scaling applied


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_skeleton(skeleton):
    """Normalize one skeleton sequence: center → scale → relative hands.

    Per-frame normalization makes the model invariant to:
        - Signer's position in the frame (centering)
        - Signer's distance from the camera (scale normalization)
        - Global arm position (hand re-anchoring)

    For each frame independently:
        1. Subtract the mid-shoulder point from all joint coordinates.
        2. Divide all coordinates by the shoulder width.
        3. Subtract each hand's wrist root from its own finger joints.

    Frames where no reference joints are found are left unchanged (zeros).

    Args:
        skeleton (np.ndarray): Shape (T, F) — one video's feature sequence.

    Returns:
        np.ndarray: Shape (T, F) — normalized skeleton, float32.
    """
    skeleton = skeleton.copy().astype(np.float32)
    for t in range(skeleton.shape[0]):
        frame = skeleton[t]

        # Determine the centering point for this frame
        center = _pick_center(frame)
        if center is None:
            continue   # no reference → skip this frame

        # Subtract center from all x-coordinates (even indices) and y-coordinates (odd)
        frame[0::2] -= center[0]   # all x values
        frame[1::2] -= center[1]   # all y values

        # Divide by scale to normalize skeleton size
        scale = _pick_scale(frame)
        frame /= scale

        # Re-anchor hands relative to their own wrist roots
        frame = make_relative_hands(frame)
        skeleton[t] = frame

    return skeleton


def normalize_dataset(X):
    """Apply normalize_skeleton to every sample in the dataset.

    Args:
        X (np.ndarray): Shape (N, T, F) — full dataset.

    Returns:
        np.ndarray: Shape (N, T, F) — normalized dataset, float32.
    """
    return np.array([normalize_skeleton(X[i]) for i in range(len(X))], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def smooth_skeleton(skeleton, sigma=0.8):
    """Apply Gaussian temporal smoothing to reduce detection jitter.

    The Gaussian filter is applied along the time axis (axis=0) independently
    for each feature. It reduces high-frequency noise from frame-to-frame
    detector inconsistencies without blurring the overall sign motion.

    Sigma controls smoothing strength:
        0.5 — very light smoothing
        0.8 — recommended balance (default)
        1.5 — heavy smoothing (may blur fast motion)

    Args:
        skeleton (np.ndarray): Shape (T, F).
        sigma    (float)     : Gaussian smoothing std dev (in frames).

    Returns:
        np.ndarray: Shape (T, F) — smoothed skeleton, float32.
    """
    return gaussian_filter1d(skeleton, sigma=sigma, axis=0).astype(np.float32)


def smooth_dataset(X, sigma=0.8):
    """Apply smooth_skeleton to every sample in the dataset.

    Args:
        X     (np.ndarray): Shape (N, T, F).
        sigma (float)     : Gaussian sigma.

    Returns:
        np.ndarray: Shape (N, T, F) — smoothed dataset, float32.
    """
    return np.array([smooth_skeleton(X[i], sigma) for i in range(len(X))], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# BONE AND MOTION FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_bone_vectors(X):
    """Compute bone vectors: each bone = child_joint_position - parent_joint_position.

    Bone vectors encode limb orientations and lengths independently of global
    position. They are complementary to raw joint positions: joints describe
    WHERE the body is, bones describe HOW the body is oriented.

    The bone edges are defined in BONE_EDGES (parent, child) pairs.
    The bone vector at a child joint is: child_xy - parent_xy.

    Args:
        X (np.ndarray): Normalized joint positions, shape (N, T, F).
                        F must be 102 (51 joints × 2).

    Returns:
        np.ndarray: Bone vectors, same shape (N, T, F), float32.
    """
    N, T, F = X.shape
    V = F // 2   # number of joints (51)
    # Reshape to (N, T, V, 2) for easier per-joint indexing
    joints = X.reshape(N, T, V, 2)
    bones  = np.zeros_like(joints)   # bone vectors initialized to zero
    for parent, child in BONE_EDGES:
        if parent < V and child < V:
            # Bone vector = displacement from parent to child
            bones[:, :, child, :] = joints[:, :, child, :] - joints[:, :, parent, :]
    return bones.reshape(N, T, F).astype(np.float32)


def compute_motion(X):
    """Compute frame-to-frame joint velocity (motion stream).

    For each frame t, motion[t] = X[t+1] - X[t].
    The last frame has motion = 0 (boundary condition).

    Motion features encode HOW FAST joints are moving and in what direction,
    providing temporal velocity information that complements static pose.

    Args:
        X (np.ndarray): Joint positions or bone vectors, shape (N, T, F).

    Returns:
        np.ndarray: Frame-to-frame delta, same shape (N, T, F), float32.
    """
    motion = np.zeros_like(X, dtype=np.float32)
    # motion[t] = X[t+1] - X[t] for t = 0 to T-2; last frame stays zero
    motion[:, :-1] = X[:, 1:] - X[:, :-1]
    return motion


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(ROOT, "output")

    X_raw_path = os.path.join(OUTPUT_DIR, "X_raw.npy")
    if not os.path.exists(X_raw_path):
        print("X_raw.npy not found. Run extract.py first.")
        raise SystemExit(1)

    print("Loading X_raw.npy ...")
    X_raw = np.load(X_raw_path)
    print(f"   Shape: {X_raw.shape}")

    # Step 1: Replace missing (zero) keypoints with interpolated values
    print("\nStep 1 - Cleaning missing keypoints with per-joint interpolation ...")
    X_clean = np.array([clean_missing_keypoints(X_raw[i]) for i in range(len(X_raw))], dtype=np.float32)

    # Step 2: Reduce from 75 joints (150 features) to 51 joints (102 features)
    print("\nStep 2 - Filtering joints (150 → 102) ...")
    X_filt = filter_joints(X_clean)
    print(f"   Shape: {X_filt.shape}")

    # Step 3: Center on mid-shoulder, scale by shoulder width, re-anchor hands
    print("\nStep 3 - Normalizing (center + scale + relative hands) ...")
    X_norm = normalize_dataset(X_filt)

    # Step 4: Gaussian temporal smoothing to reduce detector noise
    print("\nStep 4 - Temporal smoothing ...")
    X_smooth = smooth_dataset(X_norm, sigma=0.8)

    # Step 5: Compute bone vectors (limb orientations)
    print("\nStep 5 - Computing bone vectors ...")
    X_bones = compute_bone_vectors(X_smooth)

    # Step 6: Compute joint motion (velocity stream)
    print("\nStep 6 - Computing joint motion ...")
    X_motion = compute_motion(X_smooth)

    # Step 7: Compute bone motion (angular velocity stream)
    print("\nStep 7 - Computing bone motion ...")
    X_bone_motion = compute_motion(X_bones)

    # Save all four streams to disk for use by train.py and main.py
    np.save(os.path.join(OUTPUT_DIR, "X_normalized.npy"), X_smooth)
    np.save(os.path.join(OUTPUT_DIR, "X_bones.npy"), X_bones)
    np.save(os.path.join(OUTPUT_DIR, "X_motion.npy"), X_motion)
    np.save(os.path.join(OUTPUT_DIR, "X_bone_motion.npy"), X_bone_motion)

    print(f"\nSaved all 4 streams to {OUTPUT_DIR}")
    print(f"   X_normalized  : {X_smooth.shape}      (joint positions)")
    print(f"   X_bones       : {X_bones.shape}       (bone vectors)")
    print(f"   X_motion      : {X_motion.shape}      (joint motion)")
    print(f"   X_bone_motion : {X_bone_motion.shape} (bone motion)")
